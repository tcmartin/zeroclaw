# Host Media Bridge

Host Media Bridge exposes one local OpenAI-compatible HTTP service for ZeroClaw
running inside a macOS VM:

- `POST /v1/audio/transcriptions` -> Parakeet MLX on the host
- `POST /v1/audio/speech` -> Kokoro TTS on the host

The VM can keep its existing in-VM Parakeet/Kokoro services as fallbacks. Point
ZeroClaw at the host service from inside the VM:

```toml
[transcription]
enabled = true
default_provider = "local_whisper"
api_url = "http://192.168.64.1:5010/v1/audio/transcriptions"
model = "parakeet-tdt-0.6b-v2"

[transcription.local_whisper]
url = "http://192.168.64.1:5010/v1/audio/transcriptions"
max_audio_bytes = 524288000
timeout_secs = 900

[tts]
enabled = true
default_provider = "piper"
default_voice = "af_heart"
default_format = "wav"

[tts.piper]
api_url = "http://192.168.64.1:5010/v1/audio/speech"
```

From the host:

```bash
./scripts/host-media-bridge/start.sh
```

From inside the VM, use the helper before falling back to any in-VM Parakeet
server:

```bash
~/Projects/zeroclaw/scripts/host-media-bridge/transcribe-file.sh input.m4a transcript.txt
```

The helper posts multipart audio to
`http://192.168.64.1:5010/v1/audio/transcriptions`, writes both text and JSON
outputs, and reports the backend used.

To install it as a host LaunchAgent:

```bash
HOST_MEDIA_STT_FALLBACK_URL=http://192.168.64.3:5010/v1/audio/transcriptions \
HOST_MEDIA_TTS_FALLBACK_URL=http://192.168.64.3:5020/v1/audio/speech \
./scripts/host-media-bridge/install-launch-agent.sh
```

Useful environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `HOST_MEDIA_HOST` | `0.0.0.0` | Bind address reachable from the VM |
| `HOST_MEDIA_PORT` | `5010` | HTTP port for both STT and TTS |
| `HOST_MEDIA_STT_MODEL` | `mlx-community/parakeet-tdt-0.6b-v2` | Parakeet MLX model |
| `HOST_MEDIA_STT_CHUNK_SECS` | `90` | Chunk size for long audio |
| `HOST_MEDIA_TTS_VOICE` | `af_heart` | Default Kokoro voice |
| `HOST_MEDIA_STT_FALLBACK_URL` | unset | Optional fallback STT endpoint |
| `HOST_MEDIA_TTS_FALLBACK_URL` | unset | Optional fallback TTS endpoint |
| `HOST_MEDIA_FALLBACK_TIMEOUT_SECS` | `30` | Timeout for optional fallback calls |

With the standard Tart network, host-to-guest fallback URLs are usually:

```bash
export HOST_MEDIA_STT_FALLBACK_URL=http://192.168.64.3:5010/v1/audio/transcriptions
export HOST_MEDIA_TTS_FALLBACK_URL=http://192.168.64.3:5020/v1/audio/speech
```

The bridge returns `X-Host-Media-Backend` on successful TTS responses and
includes backend metadata in STT JSON responses so agents can tell whether host
or fallback handled a request.
