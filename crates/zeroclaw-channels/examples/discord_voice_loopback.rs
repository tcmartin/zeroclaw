use anyhow::{Context, Result, bail};
use tokio::sync::{mpsc, oneshot};
use zeroclaw_channels::discord_voice::DiscordVoiceBridge;
use zeroclaw_channels::tts::TtsManager;
use zeroclaw_config::schema::Config;

const DEFAULT_INPUT: &str =
    "this is a loopback discord voice bridge test please confirm the real time voice bridge works";
const DEFAULT_REPLY: &str =
    "Real-time voice bridge confirmed working. I heard your message loud and clear.";

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
        guild_id,
        voice.channel_id.clone(),
        discord.allowed_users.clone(),
        config.tts.clone(),
        config.transcription.clone(),
    )
    .context("Discord voice bridge could not be constructed")?
    .with_detection_window(voice.silence_ms, voice.min_utterance_ms, voice.max_utterance_ms);

    let tts = TtsManager::new(&config.tts)?;
    let input_text = std::env::args()
        .nth(1)
        .filter(|arg| !arg.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_INPUT.to_string());
    let reply_text = std::env::args()
        .nth(2)
        .filter(|arg| !arg.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_REPLY.to_string());

    let wav = tts.synthesize(&input_text).await?;
    let test_user_id = discord
        .allowed_users
        .iter()
        .find(|value| value.as_str() != "*")
        .and_then(|value| value.parse::<u64>().ok())
        .context("Discord allowed_users does not contain a numeric user id for loopback")?;

    let (tx, mut rx) = mpsc::channel(8);
    bridge.ensure_running(tx).await;

    let reply_bridge = bridge.clone();
    let expected_reply = reply_text.clone();
    let (done_tx, done_rx) = oneshot::channel::<Result<()>>();
    tokio::spawn(async move {
        let result = async {
            let message = tokio::time::timeout(std::time::Duration::from_secs(90), rx.recv())
                .await
                .context("Timed out waiting for transcribed Discord voice message")?
                .context("Discord voice message channel closed unexpectedly")?;
            println!("TRANSCRIBED={}", message.content);
            reply_bridge
                .play_reply(&message.reply_target, &expected_reply)
                .await
                .context("Failed to play reply into Discord voice channel")?;
            println!("REPLY={expected_reply}");
            println!("VOICE_SEND_OK");
            Ok(())
        }
        .await;
        let _ = done_tx.send(result);
    });

    bridge
        .simulate_wav_utterance(test_user_id, &wav)
        .await
        .context("Failed to inject synthetic Discord voice utterance")?;

    tokio::time::timeout(std::time::Duration::from_secs(120), done_rx)
        .await
        .context("Timed out waiting for loopback reply")?
        .context("Loopback task dropped completion signal")??;

    Ok(())
}
