use anyhow::{Context, Result, bail};
use tokio::sync::mpsc;
use zeroclaw_api::channel::{Channel, SendMessage};
use zeroclaw_channels::discord::DiscordChannel;
use zeroclaw_channels::discord_voice::DiscordVoiceBridge;
use zeroclaw_channels::tts::TtsManager;
use zeroclaw_config::schema::{Config, StreamMode};

const DEFAULT_INPUT: &str =
    "this is a loopback discord voice bridge test please confirm streamed voice replies work";
const DEFAULT_REPLY_FIRST: &str = "Streamed voice reply confirmed working.";
const DEFAULT_REPLY_SECOND: &str = "I heard your message loud and clear through the realtime draft path.";

#[tokio::main]
async fn main() -> Result<()> {
    let _ = rustls::crypto::ring::default_provider().install_default();
    let config = Config::load_or_init().await?;
    let discord = config
        .channels
        .discord
        .as_ref()
        .context("channels.discord is not configured")?;
    if !discord.enabled {
        bail!("Discord channel is disabled in config");
    }
    let voice = discord
        .voice
        .as_ref()
        .context("channels.discord.voice is not configured")?;
    let guild_id = discord
        .guild_id
        .clone()
        .context("channels.discord.guild_id is not configured")?;
    if !voice.enabled {
        bail!("Discord voice bridge is disabled in config");
    }

    let bridge = DiscordVoiceBridge::new(
        discord.bot_token.clone(),
        guild_id.clone(),
        voice.channel_id.clone(),
        discord.allowed_users.clone(),
        config.tts.clone(),
        config.transcription.clone(),
    )
    .context("Discord voice bridge could not be constructed")?
    .with_detection_window(voice.silence_ms, voice.min_utterance_ms, voice.max_utterance_ms);

    let channel = DiscordChannel::new(
        discord.bot_token.clone(),
        Some(guild_id),
        discord.allowed_users.clone(),
        false,
        false,
    )
    .with_streaming(StreamMode::Partial, 0, 0)
    .with_tts(config.tts.clone())
    .with_voice_bridge(
        Some(voice.clone()),
        config.transcription.clone(),
        config.tts.clone(),
    );

    let tts = TtsManager::new(&config.tts)?;
    let input_text = std::env::args()
        .nth(1)
        .filter(|arg| !arg.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_INPUT.to_string());
    let reply_first = std::env::args()
        .nth(2)
        .filter(|arg| !arg.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_REPLY_FIRST.to_string());
    let reply_second = std::env::args()
        .nth(3)
        .filter(|arg| !arg.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_REPLY_SECOND.to_string());

    let wav = tts.synthesize(&input_text).await?;
    let test_user_id = discord
        .allowed_users
        .iter()
        .find(|value| value.as_str() != "*")
        .and_then(|value| value.parse::<u64>().ok())
        .context("Discord allowed_users does not contain a numeric user id for loopback")?;

    let (tx, mut rx) = mpsc::channel(8);
    bridge.ensure_running(tx).await;

    let message = tokio::time::timeout(std::time::Duration::from_secs(90), async {
        bridge
            .simulate_wav_utterance(test_user_id, &wav)
            .await
            .context("Failed to inject synthetic Discord voice utterance")?;
        rx.recv()
            .await
            .context("Discord voice message channel closed unexpectedly")
    })
    .await
    .context("Timed out waiting for transcribed Discord voice message")??;

    println!("TRANSCRIBED={}", message.content);

    let draft_id = channel
        .send_draft(&SendMessage::new("", &message.reply_target))
        .await?
        .context("Discord streaming draft did not start for voice reply target")?;

    channel
        .update_draft(&message.reply_target, &draft_id, &reply_first)
        .await?;
    let final_reply = format!("{reply_first} {reply_second}");
    channel
        .finalize_draft(&message.reply_target, &draft_id, &final_reply)
        .await?;

    // The voice queue runs in background tasks; keep the runtime alive long enough
    // for the queued reply chunks to synthesize and play into Discord.
    tokio::time::sleep(std::time::Duration::from_secs(12)).await;
    println!("STREAMED_REPLY={final_reply}");
    println!("STREAMED_VOICE_OK");
    Ok(())
}
