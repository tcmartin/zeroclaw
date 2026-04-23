use anyhow::{Context, Result};
use zeroclaw_channels::tts::TtsManager;
use zeroclaw_config::schema::Config;

const DEFAULT_TEXT: &str = "MiniMax websocket TTS smoke test";

#[tokio::main]
async fn main() -> Result<()> {
    let _ = rustls::crypto::ring::default_provider().install_default();

    let config = Config::load_or_init().await?;
    let manager = TtsManager::new(&config.tts)?;

    let provider = std::env::args()
        .nth(1)
        .filter(|arg| !arg.trim().is_empty())
        .unwrap_or_else(|| config.tts.default_provider.clone());
    let voice = std::env::args()
        .nth(2)
        .filter(|arg| !arg.trim().is_empty())
        .unwrap_or_else(|| config.tts.default_voice.clone());
    let text = std::env::args()
        .nth(3)
        .filter(|arg| !arg.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_TEXT.to_string());

    let audio = manager
        .synthesize_with_provider(&text, &provider, &voice)
        .await
        .with_context(|| format!("TTS smoke failed for provider={provider} voice={voice}"))?;

    let preview_len = audio.len().min(16);
    println!("PROVIDER={provider}");
    println!("VOICE={voice}");
    println!("BYTES={}", audio.len());
    println!("PREFIX_HEX={}", hex::encode(&audio[..preview_len]));
    Ok(())
}
