//! Multi-provider Text-to-Speech (TTS) subsystem.
//!
//! Supports OpenAI, ElevenLabs, Google Cloud TTS, Edge TTS (free, subprocess-based),
//! and Piper TTS (local GPU-accelerated, OpenAI-compatible endpoint).
//! Provider selection is driven by [`TtsConfig`] in `config.toml`.

use std::collections::HashMap;

use anyhow::{Context, Result, bail};
use futures_util::{SinkExt, StreamExt};
use tokio_tungstenite::tungstenite::Message;

use zeroclaw_config::schema::TtsConfig;

/// Maximum text length before synthesis is rejected (default: 4096 chars).
const DEFAULT_MAX_TEXT_LENGTH: usize = 4096;

/// Default HTTP request timeout for TTS API calls.
const TTS_HTTP_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(60);

// ── TtsProvider trait ────────────────────────────────────────────

/// Trait for pluggable TTS backends.
#[async_trait::async_trait]
pub trait TtsProvider: Send + Sync {
    /// Provider identifier (e.g. `"openai"`, `"elevenlabs"`).
    fn name(&self) -> &str;

    /// Synthesize `text` using the given `voice`, returning raw audio bytes.
    async fn synthesize(&self, text: &str, voice: &str) -> Result<Vec<u8>>;

    /// Voices supported by this provider.
    fn supported_voices(&self) -> Vec<String>;

    /// Audio output formats supported by this provider.
    fn supported_formats(&self) -> Vec<String>;
}

// ── OpenAI TTS ───────────────────────────────────────────────────

/// OpenAI TTS provider (`POST /v1/audio/speech`).
pub struct OpenAiTtsProvider {
    api_key: String,
    model: String,
    speed: f64,
    output_format: String,
    client: reqwest::Client,
}

impl OpenAiTtsProvider {
    /// Create a new OpenAI TTS provider from config, resolving the API key
    /// from config or `OPENAI_API_KEY` env var.
    pub fn new(
        config: &zeroclaw_config::schema::OpenAiTtsConfig,
        output_format: &str,
    ) -> Result<Self> {
        let api_key = config
            .api_key
            .as_deref()
            .map(str::trim)
            .filter(|k| !k.is_empty())
            .map(ToOwned::to_owned)
            .or_else(|| {
                std::env::var("OPENAI_API_KEY")
                    .ok()
                    .map(|v| v.trim().to_string())
                    .filter(|v| !v.is_empty())
            })
            .context("Missing OpenAI TTS API key: set [tts.openai].api_key or OPENAI_API_KEY")?;

        Ok(Self {
            api_key,
            model: config.model.clone(),
            speed: config.speed,
            output_format: output_format.trim().to_ascii_lowercase(),
            client: reqwest::Client::builder()
                .timeout(TTS_HTTP_TIMEOUT)
                .build()
                .context("Failed to build HTTP client for OpenAI TTS")?,
        })
    }
}

#[async_trait::async_trait]
impl TtsProvider for OpenAiTtsProvider {
    fn name(&self) -> &str {
        "openai"
    }

    async fn synthesize(&self, text: &str, voice: &str) -> Result<Vec<u8>> {
        let body = serde_json::json!({
            "model": self.model,
            "input": text,
            "voice": voice,
            "speed": self.speed,
            "response_format": self.output_format,
        });

        let resp = self
            .client
            .post("https://api.openai.com/v1/audio/speech")
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .context("Failed to send OpenAI TTS request")?;

        let status = resp.status();
        if !status.is_success() {
            let error_body: serde_json::Value = resp
                .json()
                .await
                .unwrap_or_else(|_| serde_json::json!({"error": "unknown"}));
            let msg = error_body["error"]["message"]
                .as_str()
                .unwrap_or("unknown error");
            bail!("OpenAI TTS API error ({}): {}", status, msg);
        }

        let bytes = resp
            .bytes()
            .await
            .context("Failed to read OpenAI TTS response body")?;
        Ok(bytes.to_vec())
    }

    fn supported_voices(&self) -> Vec<String> {
        ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
            .iter()
            .map(|s| (*s).to_string())
            .collect()
    }

    fn supported_formats(&self) -> Vec<String> {
        ["mp3", "opus", "aac", "flac", "wav", "pcm"]
            .iter()
            .map(|s| (*s).to_string())
            .collect()
    }
}

// ── ElevenLabs TTS ───────────────────────────────────────────────

/// ElevenLabs TTS provider (`POST /v1/text-to-speech/{voice_id}`).
pub struct ElevenLabsTtsProvider {
    api_key: String,
    model_id: String,
    stability: f64,
    similarity_boost: f64,
    client: reqwest::Client,
}

impl ElevenLabsTtsProvider {
    /// Create a new ElevenLabs TTS provider from config, resolving the API key
    /// from config or `ELEVENLABS_API_KEY` env var.
    pub fn new(config: &zeroclaw_config::schema::ElevenLabsTtsConfig) -> Result<Self> {
        let api_key = config
            .api_key
            .as_deref()
            .map(str::trim)
            .filter(|k| !k.is_empty())
            .map(ToOwned::to_owned)
            .or_else(|| {
                std::env::var("ELEVENLABS_API_KEY")
                    .ok()
                    .map(|v| v.trim().to_string())
                    .filter(|v| !v.is_empty())
            })
            .context(
                "Missing ElevenLabs API key: set [tts.elevenlabs].api_key or ELEVENLABS_API_KEY",
            )?;

        Ok(Self {
            api_key,
            model_id: config.model_id.clone(),
            stability: config.stability,
            similarity_boost: config.similarity_boost,
            client: reqwest::Client::builder()
                .timeout(TTS_HTTP_TIMEOUT)
                .build()
                .context("Failed to build HTTP client for ElevenLabs TTS")?,
        })
    }
}

#[async_trait::async_trait]
impl TtsProvider for ElevenLabsTtsProvider {
    fn name(&self) -> &str {
        "elevenlabs"
    }

    async fn synthesize(&self, text: &str, voice: &str) -> Result<Vec<u8>> {
        if !voice
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            bail!("ElevenLabs voice ID contains invalid characters: {voice}");
        }
        let url = format!("https://api.elevenlabs.io/v1/text-to-speech/{voice}");
        let body = serde_json::json!({
            "text": text,
            "model_id": self.model_id,
            "voice_settings": {
                "stability": self.stability,
                "similarity_boost": self.similarity_boost,
            },
        });

        let resp = self
            .client
            .post(&url)
            .header("xi-api-key", &self.api_key)
            .json(&body)
            .send()
            .await
            .context("Failed to send ElevenLabs TTS request")?;

        let status = resp.status();
        if !status.is_success() {
            let error_body: serde_json::Value = resp
                .json()
                .await
                .unwrap_or_else(|_| serde_json::json!({"error": "unknown"}));
            let msg = error_body["detail"]["message"]
                .as_str()
                .or_else(|| error_body["detail"].as_str())
                .unwrap_or("unknown error");
            bail!("ElevenLabs TTS API error ({}): {}", status, msg);
        }

        let bytes = resp
            .bytes()
            .await
            .context("Failed to read ElevenLabs TTS response body")?;
        Ok(bytes.to_vec())
    }

    fn supported_voices(&self) -> Vec<String> {
        // ElevenLabs voices are user-specific; return empty (dynamic lookup).
        Vec::new()
    }

    fn supported_formats(&self) -> Vec<String> {
        ["mp3", "pcm", "ulaw"]
            .iter()
            .map(|s| (*s).to_string())
            .collect()
    }
}

// ── Google Cloud TTS ─────────────────────────────────────────────

/// Google Cloud TTS provider (`POST /v1/text:synthesize`).
pub struct GoogleTtsProvider {
    api_key: String,
    language_code: String,
    client: reqwest::Client,
}

impl GoogleTtsProvider {
    /// Create a new Google Cloud TTS provider from config, resolving the API key
    /// from config or `GOOGLE_TTS_API_KEY` env var.
    pub fn new(config: &zeroclaw_config::schema::GoogleTtsConfig) -> Result<Self> {
        let api_key = config
            .api_key
            .as_deref()
            .map(str::trim)
            .filter(|k| !k.is_empty())
            .map(ToOwned::to_owned)
            .or_else(|| {
                std::env::var("GOOGLE_TTS_API_KEY")
                    .ok()
                    .map(|v| v.trim().to_string())
                    .filter(|v| !v.is_empty())
            })
            .context(
                "Missing Google TTS API key: set [tts.google].api_key or GOOGLE_TTS_API_KEY",
            )?;

        Ok(Self {
            api_key,
            language_code: config.language_code.clone(),
            client: reqwest::Client::builder()
                .timeout(TTS_HTTP_TIMEOUT)
                .build()
                .context("Failed to build HTTP client for Google TTS")?,
        })
    }
}

#[async_trait::async_trait]
impl TtsProvider for GoogleTtsProvider {
    fn name(&self) -> &str {
        "google"
    }

    async fn synthesize(&self, text: &str, voice: &str) -> Result<Vec<u8>> {
        let url = "https://texttospeech.googleapis.com/v1/text:synthesize";
        let body = serde_json::json!({
            "input": { "text": text },
            "voice": {
                "languageCode": self.language_code,
                "name": voice,
            },
            "audioConfig": {
                "audioEncoding": "MP3",
            },
        });

        let resp = self
            .client
            .post(url)
            .header("x-goog-api-key", &self.api_key)
            .json(&body)
            .send()
            .await
            .context("Failed to send Google TTS request")?;

        let status = resp.status();
        let resp_body: serde_json::Value = resp
            .json()
            .await
            .context("Failed to parse Google TTS response")?;

        if !status.is_success() {
            let msg = resp_body["error"]["message"]
                .as_str()
                .unwrap_or("unknown error");
            bail!("Google TTS API error ({}): {}", status, msg);
        }

        let audio_b64 = resp_body["audioContent"]
            .as_str()
            .context("Google TTS response missing 'audioContent' field")?;

        use base64::Engine;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(audio_b64)
            .context("Failed to decode Google TTS base64 audio")?;
        Ok(bytes)
    }

    fn supported_voices(&self) -> Vec<String> {
        // Google voices vary by language; return common English defaults.
        [
            "en-US-Standard-A",
            "en-US-Standard-B",
            "en-US-Standard-C",
            "en-US-Standard-D",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect()
    }

    fn supported_formats(&self) -> Vec<String> {
        ["mp3", "wav", "ogg"]
            .iter()
            .map(|s| (*s).to_string())
            .collect()
    }
}

// ── Edge TTS (subprocess) ────────────────────────────────────────

/// Edge TTS provider — free, uses the `edge-tts` CLI subprocess.
pub struct EdgeTtsProvider {
    binary_path: String,
}

impl EdgeTtsProvider {
    /// Allowed basenames for the Edge TTS binary.
    const ALLOWED_BINARIES: &[&str] = &["edge-tts", "edge-playback"];

    /// Create a new Edge TTS provider from config.
    ///
    /// `binary_path` must be a bare command name (no path separators) matching
    /// one of [`Self::ALLOWED_BINARIES`]. This prevents arbitrary executable
    /// paths like `/tmp/malicious/edge-tts` from passing the basename check.
    pub fn new(config: &zeroclaw_config::schema::EdgeTtsConfig) -> Result<Self> {
        let path = &config.binary_path;
        if path.contains('/') || path.contains('\\') {
            bail!(
                "Edge TTS binary_path must be a bare command name without path separators, got: {path}"
            );
        }
        if !Self::ALLOWED_BINARIES.contains(&path.as_str()) {
            bail!(
                "Edge TTS binary_path must be one of {:?}, got: {path}",
                Self::ALLOWED_BINARIES,
            );
        }
        Ok(Self {
            binary_path: config.binary_path.clone(),
        })
    }
}

#[async_trait::async_trait]
impl TtsProvider for EdgeTtsProvider {
    fn name(&self) -> &str {
        "edge"
    }

    async fn synthesize(&self, text: &str, voice: &str) -> Result<Vec<u8>> {
        let temp_dir = std::env::temp_dir();
        let output_file = temp_dir.join(format!("zeroclaw_tts_{}.mp3", uuid::Uuid::new_v4()));
        let output_path = output_file
            .to_str()
            .context("Failed to build temp file path for Edge TTS")?;

        let output = tokio::time::timeout(
            TTS_HTTP_TIMEOUT,
            tokio::process::Command::new(&self.binary_path)
                .arg("--text")
                .arg(text)
                .arg("--voice")
                .arg(voice)
                .arg("--write-media")
                .arg(output_path)
                .output(),
        )
        .await
        .context("Edge TTS subprocess timed out")?
        .context("Failed to spawn edge-tts subprocess")?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            // Clean up temp file on failure.
            let _ = tokio::fs::remove_file(&output_file).await;
            bail!("edge-tts failed (exit {}): {}", output.status, stderr);
        }

        let bytes = tokio::fs::read(&output_file)
            .await
            .context("Failed to read edge-tts output file")?;

        // Clean up temp file.
        let _ = tokio::fs::remove_file(&output_file).await;

        Ok(bytes)
    }

    fn supported_voices(&self) -> Vec<String> {
        // Edge TTS has many voices; return common defaults.
        [
            "en-US-AriaNeural",
            "en-US-GuyNeural",
            "en-US-JennyNeural",
            "en-GB-SoniaNeural",
        ]
        .iter()
        .map(|s| (*s).to_string())
        .collect()
    }

    fn supported_formats(&self) -> Vec<String> {
        vec!["mp3".to_string()]
    }
}

// ── Piper TTS (local, OpenAI-compatible) ─────────────────────────

/// Piper TTS provider — local GPU-accelerated server with an OpenAI-compatible endpoint.
pub struct PiperTtsProvider {
    client: reqwest::Client,
    api_url: String,
    output_format: String,
}

impl PiperTtsProvider {
    /// Create a new Piper TTS provider pointing at the given API URL.
    pub fn new(api_url: &str, output_format: &str) -> Self {
        Self {
            client: reqwest::Client::builder()
                .timeout(TTS_HTTP_TIMEOUT)
                .build()
                .expect("Failed to build HTTP client for Piper TTS"),
            api_url: api_url.to_string(),
            output_format: output_format.trim().to_ascii_lowercase(),
        }
    }
}

#[async_trait::async_trait]
impl TtsProvider for PiperTtsProvider {
    fn name(&self) -> &str {
        "piper"
    }

    async fn synthesize(&self, text: &str, voice: &str) -> Result<Vec<u8>> {
        let body = serde_json::json!({
            "model": "tts-1",
            "input": text,
            "voice": voice,
            "response_format": self.output_format,
        });

        let resp = self
            .client
            .post(&self.api_url)
            .json(&body)
            .send()
            .await
            .context("Failed to send Piper TTS request")?;

        let status = resp.status();
        if !status.is_success() {
            let error_body: serde_json::Value = resp
                .json()
                .await
                .unwrap_or_else(|_| serde_json::json!({"error": "unknown"}));
            let msg = error_body["error"]["message"]
                .as_str()
                .unwrap_or("unknown error");
            bail!("Piper TTS API error ({}): {}", status, msg);
        }

        let bytes = resp
            .bytes()
            .await
            .context("Failed to read Piper TTS response body")?;
        Ok(bytes.to_vec())
    }

    fn supported_voices(&self) -> Vec<String> {
        // Piper voices depend on installed models; return empty (dynamic).
        Vec::new()
    }

    fn supported_formats(&self) -> Vec<String> {
        ["mp3", "wav", "opus"]
            .iter()
            .map(|s| (*s).to_string())
            .collect()
    }
}

// ── MiniMax WebSocket TTS ───────────────────────────────────────

/// MiniMax TTS provider using the documented WebSocket streaming API.
pub struct MiniMaxWsTtsProvider {
    api_key: String,
    websocket_url: String,
    model: String,
    language_boost: String,
    speed: f64,
    volume: f64,
    pitch: f64,
    sample_rate: u32,
    bitrate: u32,
    channel: u8,
    output_format: String,
}

impl MiniMaxWsTtsProvider {
    fn json_number(value: f64) -> serde_json::Value {
        if value.is_finite() && value.fract() == 0.0 {
            serde_json::Value::Number(serde_json::Number::from(value as i64))
        } else if let Some(number) = serde_json::Number::from_f64(value) {
            serde_json::Value::Number(number)
        } else {
            serde_json::Value::Number(serde_json::Number::from(0))
        }
    }

    pub fn new(
        config: &zeroclaw_config::schema::MiniMaxTtsConfig,
        output_format: &str,
    ) -> Result<Self> {
        let api_key = config
            .api_key
            .as_deref()
            .map(str::trim)
            .filter(|k| !k.is_empty())
            .map(ToOwned::to_owned)
            .or_else(|| {
                std::env::var("MINIMAX_API_KEY")
                    .ok()
                    .map(|v| v.trim().to_string())
                    .filter(|v| !v.is_empty())
            })
            .context("Missing MiniMax TTS API key: set [tts.minimax].api_key or MINIMAX_API_KEY")?;

        let output_format = output_format.trim().to_ascii_lowercase();
        if output_format == "wav" {
            bail!(
                "MiniMax WebSocket TTS does not support wav streaming output; use mp3, pcm, or flac"
            );
        }

        Ok(Self {
            api_key,
            websocket_url: config.websocket_url.trim().to_string(),
            model: config.model.trim().to_string(),
            language_boost: config.language_boost.trim().to_string(),
            speed: config.speed,
            volume: config.volume,
            pitch: config.pitch,
            sample_rate: config.sample_rate,
            bitrate: config.bitrate,
            channel: config.channel,
            output_format,
        })
    }

    fn task_start_payload(&self, voice: &str) -> serde_json::Value {
        let mut payload = serde_json::json!({
            "event": "task_start",
            "model": self.model,
            "voice_setting": {
                "voice_id": voice,
                "speed": Self::json_number(self.speed),
                "vol": Self::json_number(self.volume),
                "pitch": Self::json_number(self.pitch),
            },
            "audio_setting": {
                "sample_rate": self.sample_rate,
                "bitrate": self.bitrate,
                "format": self.output_format,
                "channel": self.channel,
            }
        });
        if !self.language_boost.is_empty() && !self.language_boost.eq_ignore_ascii_case("auto") {
            payload["language_boost"] = serde_json::Value::String(self.language_boost.clone());
        }
        payload
    }

    fn extract_status_msg(payload: &serde_json::Value) -> Option<String> {
        payload
            .get("base_resp")
            .and_then(|base| base.get("status_msg"))
            .and_then(serde_json::Value::as_str)
            .map(ToOwned::to_owned)
            .or_else(|| {
                payload
                    .get("message")
                    .and_then(serde_json::Value::as_str)
                    .map(ToOwned::to_owned)
            })
    }
}

#[async_trait::async_trait]
impl TtsProvider for MiniMaxWsTtsProvider {
    fn name(&self) -> &str {
        "minimax"
    }

    async fn synthesize(&self, text: &str, voice: &str) -> Result<Vec<u8>> {
        let auth_header = format!("Bearer {}", self.api_key);
        use tokio_tungstenite::tungstenite::client::IntoClientRequest as _;
        use tokio_tungstenite::tungstenite::http::HeaderValue;

        let mut request = self
            .websocket_url
            .as_str()
            .into_client_request()
            .context("Failed to build MiniMax TTS WebSocket request")?;
        request.headers_mut().insert(
            "Authorization",
            HeaderValue::from_str(&auth_header)
                .context("MiniMax TTS Authorization header is invalid")?,
        );

        let (mut ws, _) = tokio_tungstenite::connect_async(request)
            .await
            .context("Failed to connect to MiniMax TTS WebSocket")?;

        let mut audio = Vec::new();
        let synth_result = async {
            let connected = ws
                .next()
                .await
                .context("MiniMax TTS WebSocket closed before connected_success")??;
            let connected = match connected {
                Message::Text(text) => serde_json::from_str::<serde_json::Value>(&text)
                    .context("Invalid MiniMax connected_success payload")?,
                other => bail!("Unexpected MiniMax TTS frame before start: {other:?}"),
            };
            if connected.get("event").and_then(serde_json::Value::as_str)
                != Some("connected_success")
            {
                let detail =
                    Self::extract_status_msg(&connected).unwrap_or_else(|| connected.to_string());
                bail!("MiniMax TTS connection failed: {detail}");
            }

            ws.send(Message::Text(
                self.task_start_payload(voice).to_string().into(),
            ))
            .await
            .context("Failed to send MiniMax TTS task_start")?;

            loop {
                let frame = ws
                    .next()
                    .await
                    .context("MiniMax TTS WebSocket closed before task_started")??;
                match frame {
                    Message::Text(text) => {
                        let payload = serde_json::from_str::<serde_json::Value>(&text)
                            .context("Invalid MiniMax task_started payload")?;
                        match payload.get("event").and_then(serde_json::Value::as_str) {
                            Some("task_started") => break,
                            Some("task_failed") => {
                                let detail = Self::extract_status_msg(&payload)
                                    .unwrap_or_else(|| payload.to_string());
                                bail!("MiniMax TTS task_start failed: {detail}");
                            }
                            Some("connected_success") => continue,
                            _ => {}
                        }
                    }
                    Message::Ping(payload) => {
                        ws.send(Message::Pong(payload))
                            .await
                            .context("Failed to answer MiniMax TTS ping")?;
                    }
                    Message::Close(frame) => {
                        bail!("MiniMax TTS WebSocket closed before task_started: {frame:?}");
                    }
                    _ => {}
                }
            }

            ws.send(Message::Text(
                serde_json::json!({
                    "event": "task_continue",
                    "text": text,
                })
                .to_string()
                .into(),
            ))
            .await
            .context("Failed to send MiniMax TTS task_continue")?;

            loop {
                let frame = ws
                    .next()
                    .await
                    .context("MiniMax TTS WebSocket closed before synthesis completed")??;
                match frame {
                    Message::Text(text) => {
                        let payload = serde_json::from_str::<serde_json::Value>(&text)
                            .context("Invalid MiniMax streaming payload")?;
                        if payload.get("event").and_then(serde_json::Value::as_str)
                            == Some("task_failed")
                        {
                            let detail = Self::extract_status_msg(&payload)
                                .unwrap_or_else(|| payload.to_string());
                            bail!("MiniMax TTS synthesis failed: {detail}");
                        }
                        if let Some(hex_audio) = payload
                            .get("data")
                            .and_then(|data| data.get("audio"))
                            .and_then(serde_json::Value::as_str)
                            .filter(|value| !value.is_empty())
                        {
                            let chunk = hex::decode(hex_audio)
                                .context("MiniMax TTS returned invalid hex audio")?;
                            audio.extend_from_slice(&chunk);
                        }
                        if payload
                            .get("is_final")
                            .and_then(serde_json::Value::as_bool)
                            .unwrap_or(false)
                        {
                            break;
                        }
                    }
                    Message::Ping(payload) => {
                        ws.send(Message::Pong(payload))
                            .await
                            .context("Failed to answer MiniMax TTS ping")?;
                    }
                    Message::Close(frame) => {
                        bail!("MiniMax TTS WebSocket closed during synthesis: {frame:?}");
                    }
                    _ => {}
                }
            }

            Ok::<(), anyhow::Error>(())
        }
        .await;

        let _ = ws
            .send(Message::Text(
                serde_json::json!({
                    "event": "task_finish",
                })
                .to_string()
                .into(),
            ))
            .await;
        let _ = tokio::time::timeout(std::time::Duration::from_secs(1), ws.close(None)).await;

        synth_result?;

        if audio.is_empty() {
            bail!("MiniMax TTS returned no audio data");
        }

        Ok(audio)
    }

    fn supported_voices(&self) -> Vec<String> {
        Vec::new()
    }

    fn supported_formats(&self) -> Vec<String> {
        ["mp3", "pcm", "flac"]
            .iter()
            .map(|s| (*s).to_string())
            .collect()
    }
}

// ── TtsManager ───────────────────────────────────────────────────

/// Central manager for multi-provider TTS synthesis.
pub struct TtsManager {
    providers: HashMap<String, Box<dyn TtsProvider>>,
    default_provider: String,
    default_voice: String,
    max_text_length: usize,
}

impl TtsManager {
    /// Build a `TtsManager` from config, initializing all configured providers.
    pub fn new(config: &TtsConfig) -> Result<Self> {
        let mut providers: HashMap<String, Box<dyn TtsProvider>> = HashMap::new();

        if let Some(ref openai_cfg) = config.openai {
            match OpenAiTtsProvider::new(openai_cfg, &config.default_format) {
                Ok(p) => {
                    providers.insert("openai".to_string(), Box::new(p));
                }
                Err(e) => {
                    tracing::warn!("Skipping OpenAI TTS provider: {e}");
                }
            }
        }

        if let Some(ref elevenlabs_cfg) = config.elevenlabs {
            match ElevenLabsTtsProvider::new(elevenlabs_cfg) {
                Ok(p) => {
                    providers.insert("elevenlabs".to_string(), Box::new(p));
                }
                Err(e) => {
                    tracing::warn!("Skipping ElevenLabs TTS provider: {e}");
                }
            }
        }

        if let Some(ref google_cfg) = config.google {
            match GoogleTtsProvider::new(google_cfg) {
                Ok(p) => {
                    providers.insert("google".to_string(), Box::new(p));
                }
                Err(e) => {
                    tracing::warn!("Skipping Google TTS provider: {e}");
                }
            }
        }

        if let Some(ref edge_cfg) = config.edge {
            match EdgeTtsProvider::new(edge_cfg) {
                Ok(p) => {
                    providers.insert("edge".to_string(), Box::new(p));
                }
                Err(e) => {
                    tracing::warn!("Skipping Edge TTS provider: {e}");
                }
            }
        }

        if let Some(ref piper_cfg) = config.piper {
            let provider = PiperTtsProvider::new(&piper_cfg.api_url, &config.default_format);
            providers.insert("piper".to_string(), Box::new(provider));
        }

        if let Some(ref minimax_cfg) = config.minimax {
            match MiniMaxWsTtsProvider::new(minimax_cfg, &config.default_format) {
                Ok(p) => {
                    providers.insert("minimax".to_string(), Box::new(p));
                }
                Err(e) => {
                    tracing::warn!("Skipping MiniMax TTS provider: {e}");
                }
            }
        }

        let max_text_length = if config.max_text_length == 0 {
            DEFAULT_MAX_TEXT_LENGTH
        } else {
            config.max_text_length
        };

        Ok(Self {
            providers,
            default_provider: config.default_provider.clone(),
            default_voice: config.default_voice.clone(),
            max_text_length,
        })
    }

    /// Synthesize text using the default provider and voice.
    pub async fn synthesize(&self, text: &str) -> Result<Vec<u8>> {
        self.synthesize_with_provider(text, &self.default_provider, &self.default_voice)
            .await
    }

    fn resolve_provider_key<'a>(&'a self, requested: &str) -> Option<&'a str> {
        let trimmed = requested.trim();
        if let Some((key, _)) = self.providers.get_key_value(trimmed) {
            return Some(key.as_str());
        }

        let lowered = trimmed.to_ascii_lowercase();
        if let Some((key, _)) = self.providers.get_key_value(lowered.as_str()) {
            return Some(key.as_str());
        }

        // Compatibility alias: users frequently configure local Kokoro-style
        // OpenAI-compatible endpoints via [tts.piper].
        if matches!(lowered.as_str(), "kokoro" | "kokoro-local" | "local-kokoro")
            && let Some((key, _)) = self.providers.get_key_value("piper")
        {
            tracing::warn!(
                requested_provider = trimmed,
                resolved_provider = "piper",
                "TTS provider alias applied"
            );
            return Some(key.as_str());
        }

        if matches!(
            lowered.as_str(),
            "minimax-ws" | "minimax-websocket" | "minimax-tts"
        ) && let Some((key, _)) = self.providers.get_key_value("minimax")
        {
            return Some(key.as_str());
        }

        // Last-resort resilience: if exactly one provider is configured, use it.
        if self.providers.len() == 1
            && let Some((key, _)) = self.providers.iter().next()
        {
            tracing::warn!(
                requested_provider = trimmed,
                resolved_provider = key,
                "TTS provider not configured; using the only available provider"
            );
            return Some(key.as_str());
        }

        None
    }

    /// Synthesize text using a specific provider and voice.
    pub async fn synthesize_with_provider(
        &self,
        text: &str,
        provider: &str,
        voice: &str,
    ) -> Result<Vec<u8>> {
        if text.is_empty() {
            bail!("TTS text must not be empty");
        }
        let char_count = text.chars().count();
        if char_count > self.max_text_length {
            bail!(
                "TTS text too long ({} chars, max {})",
                char_count,
                self.max_text_length
            );
        }

        let provider_key = self.resolve_provider_key(provider).ok_or_else(|| {
            anyhow::anyhow!(
                "TTS provider '{}' not configured (available: {})",
                provider,
                self.available_providers().join(", ")
            )
        })?;
        let tts = self.providers.get(provider_key).ok_or_else(|| {
            anyhow::anyhow!(
                "Resolved TTS provider '{}' is unavailable (available: {})",
                provider_key,
                self.available_providers().join(", ")
            )
        })?;

        tts.synthesize(text, voice).await
    }

    /// List names of all initialized providers.
    pub fn available_providers(&self) -> Vec<String> {
        let mut names: Vec<_> = self.providers.keys().cloned().collect();
        names.sort();
        names
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_tts_config() -> TtsConfig {
        TtsConfig::default()
    }

    #[test]
    fn tts_manager_creation_with_defaults() {
        let config = default_tts_config();
        let manager = TtsManager::new(&config).unwrap();
        // No providers configured by default, so list is empty.
        assert!(manager.available_providers().is_empty());
    }

    #[test]
    fn tts_manager_with_edge_provider() {
        let mut config = default_tts_config();
        config.default_provider = "edge".to_string();
        config.edge = Some(zeroclaw_config::schema::EdgeTtsConfig {
            binary_path: "edge-tts".into(),
        });

        let manager = TtsManager::new(&config).unwrap();
        assert_eq!(manager.available_providers(), vec!["edge"]);
    }

    #[tokio::test]
    async fn tts_rejects_empty_text() {
        let mut config = default_tts_config();
        config.default_provider = "edge".to_string();
        config.edge = Some(zeroclaw_config::schema::EdgeTtsConfig {
            binary_path: "edge-tts".into(),
        });

        let manager = TtsManager::new(&config).unwrap();
        let err = manager
            .synthesize_with_provider("", "edge", "en-US-AriaNeural")
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("must not be empty"),
            "expected empty-text error, got: {err}"
        );
    }

    #[tokio::test]
    async fn tts_rejects_text_exceeding_max_length() {
        let mut config = default_tts_config();
        config.default_provider = "edge".to_string();
        config.max_text_length = 10;
        config.edge = Some(zeroclaw_config::schema::EdgeTtsConfig {
            binary_path: "edge-tts".into(),
        });

        let manager = TtsManager::new(&config).unwrap();
        let long_text = "a".repeat(11);
        let err = manager
            .synthesize_with_provider(&long_text, "edge", "en-US-AriaNeural")
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("too long"),
            "expected too-long error, got: {err}"
        );
    }

    #[tokio::test]
    async fn tts_rejects_unknown_provider() {
        let config = default_tts_config();
        let manager = TtsManager::new(&config).unwrap();
        let err = manager
            .synthesize_with_provider("hello", "nonexistent", "voice")
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("not configured"),
            "expected not-configured error, got: {err}"
        );
    }

    #[test]
    fn piper_provider_creation() {
        let provider = PiperTtsProvider::new("http://127.0.0.1:5000/v1/audio/speech", "wav");
        assert_eq!(provider.name(), "piper");
        assert_eq!(provider.api_url, "http://127.0.0.1:5000/v1/audio/speech");
        assert_eq!(provider.output_format, "wav");
        assert_eq!(provider.supported_formats(), vec!["mp3", "wav", "opus"]);
        // Piper voices depend on installed models; list is empty.
        assert!(provider.supported_voices().is_empty());
    }

    #[test]
    fn openai_provider_creation_uses_requested_output_format() {
        let provider = OpenAiTtsProvider::new(
            &zeroclaw_config::schema::OpenAiTtsConfig {
                api_key: Some("test-key".into()),
                model: "tts-1".into(),
                speed: 1.0,
            },
            "wav",
        )
        .unwrap();
        assert_eq!(provider.output_format, "wav");
    }

    #[test]
    fn tts_manager_with_piper_provider() {
        let mut config = default_tts_config();
        config.default_provider = "piper".to_string();
        config.piper = Some(zeroclaw_config::schema::PiperTtsConfig {
            api_url: "http://127.0.0.1:5000/v1/audio/speech".into(),
        });

        let manager = TtsManager::new(&config).unwrap();
        assert_eq!(manager.available_providers(), vec!["piper"]);
    }

    #[test]
    fn tts_manager_with_minimax_provider() {
        let mut config = default_tts_config();
        config.default_provider = "minimax".to_string();
        config.minimax = Some(zeroclaw_config::schema::MiniMaxTtsConfig {
            api_key: Some("test-key".into()),
            ..Default::default()
        });

        let manager = TtsManager::new(&config).unwrap();
        assert_eq!(manager.available_providers(), vec!["minimax"]);
    }

    #[test]
    fn resolve_provider_key_maps_kokoro_alias_to_piper() {
        let mut config = default_tts_config();
        config.default_provider = "kokoro".to_string();
        config.piper = Some(zeroclaw_config::schema::PiperTtsConfig {
            api_url: "http://127.0.0.1:5000/v1/audio/speech".into(),
        });
        let manager = TtsManager::new(&config).unwrap();
        assert_eq!(manager.resolve_provider_key("kokoro"), Some("piper"));
    }

    #[test]
    fn resolve_provider_key_maps_minimax_ws_alias_to_minimax() {
        let mut config = default_tts_config();
        config.default_provider = "minimax".to_string();
        config.minimax = Some(zeroclaw_config::schema::MiniMaxTtsConfig {
            api_key: Some("test-key".into()),
            ..Default::default()
        });
        let manager = TtsManager::new(&config).unwrap();
        assert_eq!(manager.resolve_provider_key("minimax-ws"), Some("minimax"));
    }

    #[test]
    fn resolve_provider_key_falls_back_to_only_provider() {
        let mut config = default_tts_config();
        config.default_provider = "something-else".to_string();
        config.piper = Some(zeroclaw_config::schema::PiperTtsConfig {
            api_url: "http://127.0.0.1:5000/v1/audio/speech".into(),
        });
        let manager = TtsManager::new(&config).unwrap();
        assert_eq!(manager.resolve_provider_key("missing"), Some("piper"));
    }

    #[tokio::test]
    async fn tts_rejects_empty_text_for_piper() {
        let mut config = default_tts_config();
        config.default_provider = "piper".to_string();
        config.piper = Some(zeroclaw_config::schema::PiperTtsConfig {
            api_url: "http://127.0.0.1:5000/v1/audio/speech".into(),
        });

        let manager = TtsManager::new(&config).unwrap();
        let err = manager
            .synthesize_with_provider("", "piper", "default")
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("must not be empty"),
            "expected empty-text error, got: {err}"
        );
    }

    #[test]
    fn piper_not_registered_when_config_is_none() {
        let config = default_tts_config();
        let manager = TtsManager::new(&config).unwrap();
        assert!(!manager.available_providers().contains(&"piper".to_string()));
    }

    #[test]
    fn tts_config_defaults() {
        let config = TtsConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.default_provider, "openai");
        assert_eq!(config.default_voice, "alloy");
        assert_eq!(config.default_format, "mp3");
        assert_eq!(config.max_text_length, DEFAULT_MAX_TEXT_LENGTH);
        assert!(config.openai.is_none());
        assert!(config.elevenlabs.is_none());
        assert!(config.google.is_none());
        assert!(config.edge.is_none());
        assert!(config.piper.is_none());
        assert!(config.minimax.is_none());
    }

    #[test]
    fn tts_manager_max_text_length_zero_uses_default() {
        let mut config = default_tts_config();
        config.max_text_length = 0;
        let manager = TtsManager::new(&config).unwrap();
        assert_eq!(manager.max_text_length, DEFAULT_MAX_TEXT_LENGTH);
    }

    #[test]
    fn tts_manager_rejects_minimax_wav_streaming() {
        let mut config = default_tts_config();
        config.default_provider = "minimax".to_string();
        config.default_format = "wav".to_string();
        config.minimax = Some(zeroclaw_config::schema::MiniMaxTtsConfig {
            api_key: Some("test-key".into()),
            ..Default::default()
        });

        let manager = TtsManager::new(&config).unwrap();
        assert!(manager.available_providers().is_empty());
    }

    #[tokio::test]
    async fn minimax_ws_provider_synthesizes_audio_from_mock_server() {
        use tokio::net::TcpListener;
        use tokio_tungstenite::accept_async;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut ws = accept_async(stream).await.unwrap();

            ws.send(Message::Text(
                serde_json::json!({
                    "event": "connected_success",
                    "base_resp": {"status_code": 0, "status_msg": "success"},
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();

            let task_start = ws.next().await.unwrap().unwrap();
            let task_start = match task_start {
                Message::Text(text) => serde_json::from_str::<serde_json::Value>(&text).unwrap(),
                other => panic!("unexpected task_start frame: {other:?}"),
            };
            assert_eq!(task_start["event"], "task_start");
            assert_eq!(task_start["model"], "speech-2.8-turbo");

            ws.send(Message::Text(
                serde_json::json!({
                    "event": "task_started",
                    "base_resp": {"status_code": 0, "status_msg": "success"},
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();

            let task_continue = ws.next().await.unwrap().unwrap();
            let task_continue = match task_continue {
                Message::Text(text) => serde_json::from_str::<serde_json::Value>(&text).unwrap(),
                other => panic!("unexpected task_continue frame: {other:?}"),
            };
            assert_eq!(task_continue["event"], "task_continue");
            assert_eq!(task_continue["text"], "hello from minimax ws");

            ws.send(Message::Text(
                serde_json::json!({
                    "data": {"audio": hex::encode(b"audio-one")},
                    "is_final": false,
                    "base_resp": {"status_code": 0, "status_msg": "success"},
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();
            ws.send(Message::Text(
                serde_json::json!({
                    "data": {"audio": hex::encode(b"audio-two")},
                    "is_final": true,
                    "base_resp": {"status_code": 0, "status_msg": "success"},
                })
                .to_string()
                .into(),
            ))
            .await
            .unwrap();

            let finish = ws.next().await.unwrap().unwrap();
            let finish = match finish {
                Message::Text(text) => serde_json::from_str::<serde_json::Value>(&text).unwrap(),
                other => panic!("unexpected task_finish frame: {other:?}"),
            };
            assert_eq!(finish["event"], "task_finish");
        });

        let provider = MiniMaxWsTtsProvider::new(
            &zeroclaw_config::schema::MiniMaxTtsConfig {
                api_key: Some("test-key".into()),
                websocket_url: format!("ws://{addr}"),
                ..Default::default()
            },
            "mp3",
        )
        .unwrap();

        let audio = provider
            .synthesize("hello from minimax ws", "male-qn-qingse")
            .await
            .unwrap();
        assert_eq!(audio, b"audio-oneaudio-two");

        server.await.unwrap();
    }
}
