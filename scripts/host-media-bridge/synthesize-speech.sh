#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  cat >&2 <<'USAGE'
Usage: synthesize-speech.sh INPUT_TEXT_OR_@FILE OUTPUT_AUDIO [VOICE]

Synthesizes speech through the host media bridge first. This is intended for
ZeroClaw running inside a macOS VM; prefer this helper or the configured
[tts.piper].api_url over calling an in-VM Kokoro/Piper server directly.

Environment:
  HOST_MEDIA_URL      default: http://192.168.64.1:5010
  HOST_MEDIA_TTS_MODEL default: kokoro-82m
  HOST_MEDIA_TTS_VOICE default: af_heart
  HOST_MEDIA_TTS_FORMAT default: inferred from OUTPUT_AUDIO extension, then wav
  HOST_MEDIA_TIMEOUT  default: 120
USAGE
  exit 2
fi

input=$1
output_audio=$2
voice=${3:-${HOST_MEDIA_TTS_VOICE:-af_heart}}

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

host_media_url=${HOST_MEDIA_URL:-http://192.168.64.1:5010}
model=${HOST_MEDIA_TTS_MODEL:-kokoro-82m}
timeout=${HOST_MEDIA_TIMEOUT:-120}

extension=${output_audio##*.}
if [[ "$extension" == "$output_audio" ]]; then
  format=${HOST_MEDIA_TTS_FORMAT:-wav}
else
  format=${HOST_MEDIA_TTS_FORMAT:-$extension}
fi
format=$(printf '%s' "$format" | tr '[:upper:]' '[:lower:]')

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/zeroclaw-host-media-tts.XXXXXX")
cleanup() {
  rm -rf "$work_dir"
}
trap cleanup EXIT

request="$work_dir/request.json"
headers="$work_dir/headers.txt"

python3 - "$input" "$request" "$model" "$voice" "$format" <<'PY'
import json
import sys
from pathlib import Path

raw_input, request_path, model, voice, response_format = sys.argv[1:6]
if raw_input.startswith("@"):
    text = Path(raw_input[1:]).read_text(encoding="utf-8")
else:
    text = raw_input

Path(request_path).write_text(
    json.dumps(
        {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": response_format,
        }
    ),
    encoding="utf-8",
)
PY

curl -fsS --max-time "$timeout" \
  -H "Content-Type: application/json" \
  -d "@$request" \
  -D "$headers" \
  "$host_media_url/v1/audio/speech" \
  -o "$output_audio"

backend=$(python3 - "$headers" <<'PY'
import sys
from pathlib import Path

for line in Path(sys.argv[1]).read_text(errors="replace").splitlines():
    name, sep, value = line.partition(":")
    if sep and name.strip().lower() == "x-host-media-backend":
        print(value.strip())
        break
PY
)
bytes=$(wc -c <"$output_audio" | tr -d ' ')
echo "WROTE $output_audio bytes=$bytes"
echo "BACKEND ${backend:-unknown}"
