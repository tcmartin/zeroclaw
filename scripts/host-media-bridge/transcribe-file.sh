#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  cat >&2 <<'USAGE'
Usage: transcribe-file.sh INPUT_MEDIA OUTPUT_TXT [OUTPUT_JSON]

Transcribes audio/video through the host media bridge first. This is intended
for ZeroClaw running inside a macOS VM; do not call the old in-VM
http://127.0.0.1:5010/v1/transcribe JSON endpoint unless the host bridge is
unavailable.

Environment:
  HOST_MEDIA_URL      default: http://192.168.64.1:5010
  HOST_MEDIA_MODEL    default: parakeet-tdt-0.6b-v2
  HOST_MEDIA_TIMEOUT  default: 900
USAGE
  exit 2
fi

input=$1
output_txt=$2
output_json=${3:-"${output_txt%.*}.json"}

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"

host_media_url=${HOST_MEDIA_URL:-http://192.168.64.1:5010}
model=${HOST_MEDIA_MODEL:-parakeet-tdt-0.6b-v2}
timeout=${HOST_MEDIA_TIMEOUT:-900}

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/zeroclaw-host-media.XXXXXX")
cleanup() {
  rm -rf "$work_dir"
}
trap cleanup EXIT

wav="$work_dir/input.wav"
response="$work_dir/transcript.json"

ffmpeg -hide_banner -loglevel error -y \
  -i "$input" \
  -vn -ac 1 -ar 16000 -f wav "$wav"

curl -fsS --max-time "$timeout" \
  -F "model=$model" \
  -F "response_format=json" \
  -F "file=@$wav;filename=$(basename "${input%.*}").wav" \
  "$host_media_url/v1/audio/transcriptions" \
  -o "$response"

python3 - "$response" "$output_txt" "$output_json" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
out_txt = Path(sys.argv[2])
out_json = Path(sys.argv[3])
data = json.loads(src.read_text())
text = data.get("text", "")
out_txt.write_text(text, encoding="utf-8")
out_json.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"WROTE {out_txt} chars={len(text)}")
print(f"WROTE {out_json}")
print(f"BACKEND {data.get('backend', 'unknown')}")
PY
