use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use parking_lot::Mutex;
use serde_json::json;
use songbird::{
    CoreEvent, Event, EventContext, EventHandler,
    driver::{Channels, DecodeConfig, DecodeMode, SampleRate},
    input::{Input, RawAdapter},
};
use std::collections::{HashMap, HashSet};
use std::io::Cursor;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::{RwLock, mpsc, oneshot};
use zeroclaw_api::channel::ChannelMessage;

const VOICE_REPLY_TARGET_PREFIX: &str = "discord_voice:";
const DEFAULT_VOICE_SILENCE_MS: u64 = 1200;
const DEFAULT_VOICE_MIN_UTTERANCE_MS: u64 = 400;
const DEFAULT_VOICE_MAX_UTTERANCE_MS: u64 = 12_000;
const DEFAULT_VOICE_ENERGY_THRESHOLD: f32 = 0.0125;
const VOICE_TICK_MS: u64 = 20;
const DEFAULT_READY_TIMEOUT_MS: u64 = 10_000;
const DEFAULT_PLAYBACK_SETTLE_MS: u64 = 250;

#[derive(Debug, Clone)]
pub struct DiscordVoiceBridge {
    state: Arc<DiscordVoiceBridgeState>,
}

#[derive(Debug)]
struct DiscordVoiceBridgeState {
    bot_token: String,
    guild_id: String,
    channel_id: String,
    allowed_users: HashSet<String>,
    tts: zeroclaw_config::schema::TtsConfig,
    transcription: zeroclaw_config::schema::TranscriptionConfig,
    runtime: Mutex<Option<tokio::task::JoinHandle<()>>>,
    command_tx: Mutex<Option<mpsc::UnboundedSender<VoiceBridgeCommand>>>,
    running: AtomicBool,
    joined: AtomicBool,
    silence_ms: u64,
    min_utterance_ms: u64,
    max_utterance_ms: u64,
}

#[derive(Debug, Clone)]
struct VoiceTarget {
    guild_id: String,
    channel_id: String,
}

#[derive(Debug)]
enum VoiceBridgeCommand {
    Speaking {
        user_id: u64,
        ssrc: u32,
    },
    VoiceTick {
        speaking: Vec<(u32, Vec<i16>)>,
    },
    UserLeft {
        user_id: u64,
    },
    JoinReady {
        call: Arc<tokio::sync::Mutex<songbird::Call>>,
    },
    PlayText {
        text: String,
        ack: oneshot::Sender<std::result::Result<std::time::Duration, String>>,
    },
}

#[derive(Debug, Clone, Default)]
struct UserBuffer {
    samples: Vec<f32>,
    active_ticks: u32,
    silence_ticks: u32,
}

impl UserBuffer {
    fn push(&mut self, samples: &[i16], threshold: f32) -> bool {
        if samples.is_empty() {
            self.silence_ticks = self.silence_ticks.saturating_add(1);
            return false;
        }
        let normalized: Vec<f32> = samples
            .iter()
            .map(|sample| f32::from(*sample) / 32768.0)
            .collect();
        let energy = compute_rms_energy(&normalized);
        if energy < threshold {
            self.silence_ticks = self.silence_ticks.saturating_add(1);
            return false;
        }
        self.samples.extend(normalized);
        self.active_ticks = self.active_ticks.saturating_add(1);
        self.silence_ticks = 0;
        true
    }

    fn duration_ms(&self) -> u64 {
        u64::from(self.active_ticks) * VOICE_TICK_MS
    }

    fn silence_ms(&self) -> u64 {
        u64::from(self.silence_ticks) * VOICE_TICK_MS
    }
}

impl DiscordVoiceBridge {
    pub fn new(
        bot_token: String,
        guild_id: String,
        channel_id: String,
        allowed_users: Vec<String>,
        tts: zeroclaw_config::schema::TtsConfig,
        transcription: zeroclaw_config::schema::TranscriptionConfig,
    ) -> Option<Self> {
        if guild_id.trim().is_empty()
            || channel_id.trim().is_empty()
            || !tts.enabled
            || !transcription.enabled
        {
            return None;
        }

        let allowed_users = allowed_users.into_iter().collect();
        Some(Self {
            state: Arc::new(DiscordVoiceBridgeState {
                bot_token,
                guild_id,
                channel_id,
                allowed_users,
                tts,
                transcription,
                runtime: Mutex::new(None),
                command_tx: Mutex::new(None),
                running: AtomicBool::new(false),
                joined: AtomicBool::new(false),
                silence_ms: DEFAULT_VOICE_SILENCE_MS,
                min_utterance_ms: DEFAULT_VOICE_MIN_UTTERANCE_MS,
                max_utterance_ms: DEFAULT_VOICE_MAX_UTTERANCE_MS,
            }),
        })
    }

    pub fn with_detection_window(
        mut self,
        silence_ms: u64,
        min_utterance_ms: u64,
        max_utterance_ms: u64,
    ) -> Self {
        if let Some(state) = Arc::get_mut(&mut self.state) {
            state.silence_ms = silence_ms.max(VOICE_TICK_MS);
            state.min_utterance_ms = min_utterance_ms.max(VOICE_TICK_MS);
            state.max_utterance_ms = max_utterance_ms.max(state.min_utterance_ms);
        }
        self
    }

    pub fn configured_target(&self) -> String {
        format!(
            "{VOICE_REPLY_TARGET_PREFIX}{}:{}",
            self.state.guild_id, self.state.channel_id
        )
    }

    fn parse_reply_target(recipient: &str) -> Option<VoiceTarget> {
        let rest = recipient.strip_prefix(VOICE_REPLY_TARGET_PREFIX)?;
        let (guild_id, channel_id) = rest.split_once(':')?;
        if guild_id.is_empty() || channel_id.is_empty() {
            return None;
        }
        Some(VoiceTarget {
            guild_id: guild_id.to_string(),
            channel_id: channel_id.to_string(),
        })
    }

    pub async fn ensure_running(&self, tx: mpsc::Sender<ChannelMessage>) {
        if self.state.running.load(Ordering::SeqCst) {
            return;
        }

        let mut guard = self.state.runtime.lock();
        if guard.is_some() {
            return;
        }

        let state = self.state.clone();
        let handle = tokio::spawn(async move {
            state.running.store(true, Ordering::SeqCst);
            state.joined.store(false, Ordering::SeqCst);
            if let Err(err) = run_voice_bridge(state.clone(), tx).await {
                tracing::warn!("Discord voice bridge stopped: {err}");
            }
            state.running.store(false, Ordering::SeqCst);
            state.joined.store(false, Ordering::SeqCst);
            *state.command_tx.lock() = None;
            *state.runtime.lock() = None;
        });
        *guard = Some(handle);
    }

    pub async fn play_reply(&self, recipient: &str, text: &str) -> Result<bool> {
        let Some(target) = Self::parse_reply_target(recipient) else {
            return Ok(false);
        };
        if target.guild_id != self.state.guild_id || target.channel_id != self.state.channel_id {
            bail!(
                "Discord voice reply target {} does not match configured target {}",
                recipient,
                self.configured_target()
            );
        }

        let sender = self
            .command_sender()
            .await
            .context("Discord voice bridge is not running")?;
        let (ack_tx, ack_rx) = oneshot::channel();
        sender
            .send(VoiceBridgeCommand::PlayText {
                text: text.trim().to_string(),
                ack: ack_tx,
            })
            .context("Discord voice bridge command channel closed")?;
        let playback_duration = ack_rx
            .await
            .context("Discord voice bridge dropped playback acknowledgement")?
            .map_err(anyhow::Error::msg)?;
        tokio::time::sleep(
            playback_duration + std::time::Duration::from_millis(DEFAULT_PLAYBACK_SETTLE_MS),
        )
        .await;
        Ok(true)
    }

    async fn command_sender(&self) -> Option<mpsc::UnboundedSender<VoiceBridgeCommand>> {
        let existing_sender = { self.state.command_tx.lock().clone() };
        if let Some(sender) = existing_sender {
            if self
                .wait_until_joined(std::time::Duration::from_millis(DEFAULT_READY_TIMEOUT_MS))
                .await
            {
                return Some(sender);
            }
        }

        let (tx, rx) = mpsc::channel::<ChannelMessage>(1);
        self.ensure_running(tx).await;
        drop(rx);

        if !self
            .wait_until_joined(std::time::Duration::from_millis(DEFAULT_READY_TIMEOUT_MS))
            .await
        {
            return None;
        }
        self.state.command_tx.lock().clone()
    }

    async fn wait_until_joined(&self, timeout: std::time::Duration) -> bool {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            if self.state.joined.load(Ordering::SeqCst) {
                return true;
            }
            if tokio::time::Instant::now() >= deadline {
                return false;
            }
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
    }
}

async fn run_voice_bridge(
    state: Arc<DiscordVoiceBridgeState>,
    tx: mpsc::Sender<ChannelMessage>,
) -> Result<()> {
    let (command_tx, command_rx) = mpsc::unbounded_channel();
    *state.command_tx.lock() = Some(command_tx.clone());

    let handler = DiscordVoiceGatewayHandler {
        state: state.clone(),
        command_tx: command_tx.clone(),
    };
    let intents =
        serenity::all::GatewayIntents::GUILDS | serenity::all::GatewayIntents::GUILD_VOICE_STATES;
    let songbird_config = songbird::Config::default().decode_mode(DecodeMode::Decode(
        DecodeConfig::new(Channels::Stereo, SampleRate::Hz48000),
    ));

    let mut client = songbird::serenity::register_from_config(
        serenity::Client::builder(&state.bot_token, intents).event_handler(handler),
        songbird_config,
    )
    .await
    .context("failed to build Discord voice client")?;

    let client_task = tokio::spawn(async move {
        if let Err(err) = client.start().await {
            tracing::warn!("Discord voice client error: {err}");
        }
    });

    let worker_result = voice_bridge_worker(state.clone(), command_rx, tx).await;
    client_task.abort();
    worker_result
}

async fn voice_bridge_worker(
    state: Arc<DiscordVoiceBridgeState>,
    mut command_rx: mpsc::UnboundedReceiver<VoiceBridgeCommand>,
    tx: mpsc::Sender<ChannelMessage>,
) -> Result<()> {
    let transcription_manager = Arc::new(
        crate::transcription::TranscriptionManager::new(&state.transcription)
            .context("failed to initialize Discord voice transcription")?,
    );
    let tts_manager = Arc::new(crate::tts::TtsManager::new(&state.tts)?);
    let current_call: Arc<RwLock<Option<Arc<tokio::sync::Mutex<songbird::Call>>>>> =
        Arc::new(RwLock::new(None));
    let mut ssrc_to_user: HashMap<u32, u64> = HashMap::new();
    let mut user_buffers: HashMap<u64, UserBuffer> = HashMap::new();

    while let Some(command) = command_rx.recv().await {
        match command {
            VoiceBridgeCommand::Speaking { user_id, ssrc } => {
                if !is_voice_user_allowed(&state.allowed_users, user_id) {
                    continue;
                }
                ssrc_to_user.insert(ssrc, user_id);
            }
            VoiceBridgeCommand::UserLeft { user_id } => {
                ssrc_to_user.retain(|_, mapped_user| *mapped_user != user_id);
                user_buffers.remove(&user_id);
            }
            VoiceBridgeCommand::JoinReady { call } => {
                *current_call.write().await = Some(call);
                state.joined.store(true, Ordering::SeqCst);
            }
            VoiceBridgeCommand::PlayText { text, ack } => {
                let Some(call) = current_call.read().await.clone() else {
                    let _ = ack.send(Err("voice join has not completed yet".to_string()));
                    tracing::warn!(
                        "Discord voice bridge cannot play reply before voice join completes"
                    );
                    continue;
                };
                if text.trim().is_empty() {
                    let _ = ack.send(Ok(std::time::Duration::from_millis(0)));
                    continue;
                }
                let audio = match tts_manager.synthesize(text.trim()).await {
                    Ok(bytes) => bytes,
                    Err(err) => {
                        let _ = ack.send(Err(format!("tts failed: {err}")));
                        tracing::warn!("Discord voice bridge TTS failed: {err}");
                        continue;
                    }
                };
                let wav = match ensure_wav_audio(&audio) {
                    Ok(wav) => wav,
                    Err(err) => {
                        let _ = ack.send(Err(format!("expected WAV audio from TTS: {err}")));
                        tracing::warn!("Discord voice bridge expected WAV audio from TTS: {err}");
                        continue;
                    }
                };
                let playback_duration = wav_duration(&wav)?;
                let input = wav_bytes_to_songbird_input(&wav)?;
                let mut call = call.lock().await;
                call.play_only_input(input);
                let _ = ack.send(Ok(playback_duration));
                tracing::info!("Discord voice bridge played reply into voice channel");
            }
            VoiceBridgeCommand::VoiceTick { speaking } => {
                let mut flushed = Vec::new();
                let active_users: HashSet<u64> = speaking
                    .iter()
                    .filter_map(|(ssrc, _)| ssrc_to_user.get(ssrc).copied())
                    .collect();

                for (ssrc, pcm) in speaking {
                    let Some(user_id) = ssrc_to_user.get(&ssrc).copied() else {
                        continue;
                    };
                    let buffer = user_buffers.entry(user_id).or_default();
                    let _ = buffer.push(&pcm, DEFAULT_VOICE_ENERGY_THRESHOLD);
                }

                for (user_id, buffer) in &mut user_buffers {
                    if !active_users.contains(user_id) {
                        buffer.silence_ticks = buffer.silence_ticks.saturating_add(1);
                    }
                    if should_flush_buffer(
                        buffer,
                        state.min_utterance_ms,
                        state.silence_ms,
                        state.max_utterance_ms,
                    ) {
                        flushed.push((*user_id, std::mem::take(&mut buffer.samples)));
                        buffer.active_ticks = 0;
                        buffer.silence_ticks = 0;
                    }
                }

                user_buffers
                    .retain(|_, buffer| !buffer.samples.is_empty() || buffer.active_ticks > 0);

                for (user_id, samples) in flushed {
                    let file = encode_wav_from_f32(&samples, 48_000, 2);
                    let transcription_manager = transcription_manager.clone();
                    let tx = tx.clone();
                    let reply_target = format!(
                        "{VOICE_REPLY_TARGET_PREFIX}{}:{}",
                        state.guild_id, state.channel_id
                    );
                    tokio::spawn(async move {
                        match transcription_manager
                            .transcribe(&file, "discord-voice.wav")
                            .await
                        {
                            Ok(text) => {
                                let trimmed = text.trim();
                                if trimmed.is_empty() {
                                    return;
                                }
                                tracing::info!(
                                    user_id,
                                    chars = trimmed.len(),
                                    "Discord voice bridge transcribed utterance"
                                );
                                let message = ChannelMessage {
                                    id: format!(
                                        "discord_voice_{}_{}",
                                        user_id,
                                        uuid::Uuid::new_v4()
                                    ),
                                    sender: user_id.to_string(),
                                    reply_target,
                                    content: trimmed.to_string(),
                                    channel: "discord".to_string(),
                                    timestamp: std::time::SystemTime::now()
                                        .duration_since(std::time::UNIX_EPOCH)
                                        .unwrap_or_default()
                                        .as_secs(),
                                    thread_ts: None,
                                    interruption_scope_id: None,
                                    attachments: vec![],
                                };
                                let _ = tx.send(message).await;
                            }
                            Err(err) => {
                                tracing::warn!(
                                    user_id,
                                    "Discord voice bridge transcription failed: {err}"
                                );
                            }
                        }
                    });
                }
            }
        }
    }

    Ok(())
}

struct DiscordVoiceGatewayHandler {
    state: Arc<DiscordVoiceBridgeState>,
    command_tx: mpsc::UnboundedSender<VoiceBridgeCommand>,
}

#[async_trait]
impl serenity::all::EventHandler for DiscordVoiceGatewayHandler {
    async fn ready(&self, ctx: serenity::all::Context, _ready: serenity::all::Ready) {
        let Some(manager) = songbird::get(&ctx).await else {
            tracing::warn!("Discord voice bridge could not access Songbird manager");
            return;
        };

        let guild_id = match self.state.guild_id.parse::<u64>() {
            Ok(id) => serenity::all::GuildId::new(id),
            Err(err) => {
                tracing::warn!("Discord voice bridge invalid guild_id: {err}");
                return;
            }
        };
        let channel_id = match self.state.channel_id.parse::<u64>() {
            Ok(id) => serenity::all::ChannelId::new(id),
            Err(err) => {
                tracing::warn!("Discord voice bridge invalid channel_id: {err}");
                return;
            }
        };

        match manager.join(guild_id, channel_id).await {
            Ok(call) => {
                let mut handler = call.lock().await;
                handler.add_global_event(
                    Event::Core(CoreEvent::SpeakingStateUpdate),
                    DiscordVoiceSongbirdHandler {
                        command_tx: self.command_tx.clone(),
                    },
                );
                handler.add_global_event(
                    Event::Core(CoreEvent::VoiceTick),
                    DiscordVoiceSongbirdHandler {
                        command_tx: self.command_tx.clone(),
                    },
                );
                handler.add_global_event(
                    Event::Core(CoreEvent::ClientDisconnect),
                    DiscordVoiceSongbirdHandler {
                        command_tx: self.command_tx.clone(),
                    },
                );
                let _ = self
                    .command_tx
                    .send(VoiceBridgeCommand::JoinReady { call: call.clone() });
                tracing::info!(
                    guild_id = %self.state.guild_id,
                    channel_id = %self.state.channel_id,
                    "Discord voice bridge joined configured voice channel"
                );
            }
            Err(err) => {
                tracing::warn!("Discord voice bridge failed to join voice channel: {err}");
            }
        }
    }
}

struct DiscordVoiceSongbirdHandler {
    command_tx: mpsc::UnboundedSender<VoiceBridgeCommand>,
}

#[async_trait]
impl EventHandler for DiscordVoiceSongbirdHandler {
    async fn act(&self, ctx: &EventContext<'_>) -> Option<Event> {
        match ctx {
            EventContext::SpeakingStateUpdate(state) => {
                if let Some(user_id) = state.user_id {
                    let _ = self.command_tx.send(VoiceBridgeCommand::Speaking {
                        user_id: user_id.0,
                        ssrc: state.ssrc,
                    });
                }
            }
            EventContext::VoiceTick(tick) => {
                let speaking = tick
                    .speaking
                    .iter()
                    .filter_map(|(ssrc, data)| {
                        let samples = data.decoded_voice.as_ref()?;
                        Some((*ssrc, samples.clone()))
                    })
                    .collect::<Vec<_>>();
                if !speaking.is_empty() {
                    let _ = self
                        .command_tx
                        .send(VoiceBridgeCommand::VoiceTick { speaking });
                }
            }
            EventContext::ClientDisconnect(disconnect) => {
                let _ = self.command_tx.send(VoiceBridgeCommand::UserLeft {
                    user_id: disconnect.user_id.0,
                });
            }
            _ => {}
        }
        None
    }
}

fn is_voice_user_allowed(allowed_users: &HashSet<String>, user_id: u64) -> bool {
    let user_id = user_id.to_string();
    allowed_users.contains("*") || allowed_users.contains(&user_id)
}

fn should_flush_buffer(
    buffer: &UserBuffer,
    min_utterance_ms: u64,
    silence_ms: u64,
    max_utterance_ms: u64,
) -> bool {
    let min_duration_met = buffer.duration_ms() >= min_utterance_ms;
    let silence_met = buffer.silence_ms() >= silence_ms;
    let max_duration_met = buffer.duration_ms() >= max_utterance_ms;
    !buffer.samples.is_empty() && min_duration_met && (silence_met || max_duration_met)
}

fn compute_rms_energy(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = samples.iter().map(|sample| sample * sample).sum();
    (sum_sq / samples.len() as f32).sqrt()
}

fn encode_wav_from_f32(samples: &[f32], sample_rate: u32, channels: u16) -> Vec<u8> {
    let bits_per_sample: u16 = 16;
    let byte_rate = u32::from(channels) * sample_rate * u32::from(bits_per_sample) / 8;
    let block_align = channels * bits_per_sample / 8;
    let data_len = (samples.len() * 2) as u32;
    let file_len = 36 + data_len;
    let mut buf = Vec::with_capacity(file_len as usize + 8);

    buf.extend_from_slice(b"RIFF");
    buf.extend_from_slice(&file_len.to_le_bytes());
    buf.extend_from_slice(b"WAVE");
    buf.extend_from_slice(b"fmt ");
    buf.extend_from_slice(&16u32.to_le_bytes());
    buf.extend_from_slice(&1u16.to_le_bytes());
    buf.extend_from_slice(&channels.to_le_bytes());
    buf.extend_from_slice(&sample_rate.to_le_bytes());
    buf.extend_from_slice(&byte_rate.to_le_bytes());
    buf.extend_from_slice(&block_align.to_le_bytes());
    buf.extend_from_slice(&bits_per_sample.to_le_bytes());
    buf.extend_from_slice(b"data");
    buf.extend_from_slice(&data_len.to_le_bytes());

    for sample in samples {
        let pcm16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        buf.extend_from_slice(&pcm16.to_le_bytes());
    }

    buf
}

fn ensure_wav_audio(bytes: &[u8]) -> Result<Vec<u8>> {
    if bytes.len() < 12 {
        bail!("audio payload too small");
    }
    if bytes.starts_with(b"RIFF") && &bytes[8..12] == b"WAVE" {
        return Ok(bytes.to_vec());
    }
    bail!("audio payload is not WAV; set [tts].default_format = \"wav\"")
}

fn wav_bytes_to_songbird_input(bytes: &[u8]) -> Result<Input> {
    let parsed = parse_wav_pcm(bytes)?;
    let mut pcm = Vec::with_capacity(parsed.samples.len() * std::mem::size_of::<f32>());
    for sample in parsed.samples {
        pcm.extend_from_slice(&sample.to_le_bytes());
    }
    let cursor = Cursor::new(pcm);
    Ok(Input::from(RawAdapter::new(
        cursor,
        parsed.sample_rate,
        u32::from(parsed.channels),
    )))
}

fn wav_duration(bytes: &[u8]) -> Result<std::time::Duration> {
    let parsed = parse_wav_pcm(bytes)?;
    let channels = usize::from(parsed.channels).max(1);
    let frames = parsed.samples.len() / channels;
    let secs = frames as f64 / f64::from(parsed.sample_rate.max(1));
    Ok(std::time::Duration::from_secs_f64(secs.max(0.0)))
}

struct ParsedWav {
    sample_rate: u32,
    channels: u16,
    samples: Vec<f32>,
}

fn parse_wav_pcm(bytes: &[u8]) -> Result<ParsedWav> {
    if bytes.len() < 44 || !bytes.starts_with(b"RIFF") || &bytes[8..12] != b"WAVE" {
        bail!("invalid WAV header");
    }

    let mut cursor = 12usize;
    let mut sample_rate = None;
    let mut channels = None;
    let mut bits_per_sample = None;
    let mut audio_format = None;
    let mut data = None;

    while cursor + 8 <= bytes.len() {
        let chunk_id = &bytes[cursor..cursor + 4];
        let chunk_len =
            u32::from_le_bytes(bytes[cursor + 4..cursor + 8].try_into().unwrap()) as usize;
        cursor += 8;
        if cursor + chunk_len > bytes.len() {
            break;
        }
        let chunk = &bytes[cursor..cursor + chunk_len];
        match chunk_id {
            b"fmt " if chunk.len() >= 16 => {
                audio_format = Some(u16::from_le_bytes(chunk[0..2].try_into().unwrap()));
                channels = Some(u16::from_le_bytes(chunk[2..4].try_into().unwrap()));
                sample_rate = Some(u32::from_le_bytes(chunk[4..8].try_into().unwrap()));
                bits_per_sample = Some(u16::from_le_bytes(chunk[14..16].try_into().unwrap()));
            }
            b"data" => {
                data = Some(chunk.to_vec());
            }
            _ => {}
        }
        cursor += chunk_len + (chunk_len % 2);
    }

    let sample_rate = sample_rate.context("WAV fmt chunk missing sample rate")?;
    let channels = channels.context("WAV fmt chunk missing channels")?;
    let bits_per_sample = bits_per_sample.context("WAV fmt chunk missing bit depth")?;
    let audio_format = audio_format.context("WAV fmt chunk missing format")?;
    let data = data.context("WAV data chunk missing")?;

    let samples = match (audio_format, bits_per_sample) {
        (1, 16) => data
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .map(|sample| f32::from(sample) / 32768.0)
            .collect(),
        (3, 32) => data
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
        _ => bail!(
            "unsupported WAV encoding format={} bits_per_sample={}",
            audio_format,
            bits_per_sample
        ),
    };

    Ok(ParsedWav {
        sample_rate,
        channels,
        samples,
    })
}

pub fn maybe_build_voice_message(
    bridge: Option<&DiscordVoiceBridge>,
    sender: &str,
    content: &str,
) -> Option<ChannelMessage> {
    let bridge = bridge?;
    let target = bridge.configured_target();
    Some(ChannelMessage {
        id: format!("discord_voice_{}_{}", sender, uuid::Uuid::new_v4()),
        sender: sender.to_string(),
        reply_target: target,
        content: content.to_string(),
        channel: "discord".to_string(),
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
        thread_ts: None,
        interruption_scope_id: None,
        attachments: vec![],
    })
}

pub fn voice_ready_log_json(target: &str) -> serde_json::Value {
    json!({"discord_voice_target": target})
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_reply_target_accepts_expected_shape() {
        let parsed = DiscordVoiceBridge::parse_reply_target("discord_voice:1:2").unwrap();
        assert_eq!(parsed.guild_id, "1");
        assert_eq!(parsed.channel_id, "2");
    }

    #[test]
    fn parse_reply_target_rejects_invalid_shape() {
        assert!(DiscordVoiceBridge::parse_reply_target("discord_voice:1").is_none());
        assert!(DiscordVoiceBridge::parse_reply_target("discord:1:2").is_none());
    }

    #[test]
    fn ensure_wav_audio_rejects_non_wav() {
        let err = ensure_wav_audio(b"not wav").unwrap_err();
        assert!(
            err.to_string().contains("WAV") || err.to_string().contains("too small"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn wav_round_trip_encodes_and_parses_pcm16() {
        let wav = encode_wav_from_f32(&[0.0, 0.5, -0.5, 1.0], 24_000, 1);
        let parsed = parse_wav_pcm(&wav).unwrap();
        assert_eq!(parsed.sample_rate, 24_000);
        assert_eq!(parsed.channels, 1);
        assert_eq!(parsed.samples.len(), 4);
    }

    #[test]
    fn should_flush_after_min_duration_and_silence() {
        let mut buffer = UserBuffer::default();
        buffer.samples = vec![0.1; 960 * 2 * 25];
        buffer.active_ticks = 25;
        buffer.silence_ticks = 60;
        assert!(should_flush_buffer(
            &buffer,
            DEFAULT_VOICE_MIN_UTTERANCE_MS,
            DEFAULT_VOICE_SILENCE_MS,
            DEFAULT_VOICE_MAX_UTTERANCE_MS
        ));
    }

    #[test]
    fn wav_duration_matches_encoded_audio_length() {
        let wav = encode_wav_from_f32(&vec![0.0; 24_000], 24_000, 1);
        let duration = wav_duration(&wav).unwrap();
        assert_eq!(duration.as_secs_f64(), 1.0);
    }
}
