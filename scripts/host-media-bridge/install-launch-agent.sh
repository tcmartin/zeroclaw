#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LABEL="${HOST_MEDIA_LAUNCH_LABEL:-com.zeroclaw.host-media-bridge}"
PLIST_DIR="$HOME/Library/LaunchAgents"
PLIST="$PLIST_DIR/$LABEL.plist"
LOG_DIR="$HOME/.zeroclaw/logs"
VENV_DIR="${HOST_MEDIA_VENV:-$HOME/.cache/zeroclaw-host-media/venv}"
HOST="${HOST_MEDIA_HOST:-0.0.0.0}"
PORT="${HOST_MEDIA_PORT:-5010}"
STT_MODEL="${HOST_MEDIA_STT_MODEL:-mlx-community/parakeet-tdt-0.6b-v2}"
STT_CHUNK_SECS="${HOST_MEDIA_STT_CHUNK_SECS:-90}"
TTS_VOICE="${HOST_MEDIA_TTS_VOICE:-af_heart}"
STT_FALLBACK_URL="${HOST_MEDIA_STT_FALLBACK_URL:-}"
TTS_FALLBACK_URL="${HOST_MEDIA_TTS_FALLBACK_URL:-}"
FALLBACK_TIMEOUT_SECS="${HOST_MEDIA_FALLBACK_TIMEOUT_SECS:-30}"

plist_escape() {
  /usr/bin/sed \
    -e 's/&/\&amp;/g' \
    -e 's/</\&lt;/g' \
    -e 's/>/\&gt;/g' \
    -e 's/"/\&quot;/g' \
    <<<"$1"
}

mkdir -p "$PLIST_DIR" "$LOG_DIR"

cat >"$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$(plist_escape "$LABEL")</string>
  <key>ProgramArguments</key>
  <array>
    <string>$(plist_escape "$SCRIPT_DIR/start.sh")</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$(plist_escape "$REPO_ROOT")</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOST_MEDIA_VENV</key>
    <string>$(plist_escape "$VENV_DIR")</string>
    <key>HOST_MEDIA_HOST</key>
    <string>$(plist_escape "$HOST")</string>
    <key>HOST_MEDIA_PORT</key>
    <string>$(plist_escape "$PORT")</string>
    <key>HOST_MEDIA_STT_MODEL</key>
    <string>$(plist_escape "$STT_MODEL")</string>
    <key>HOST_MEDIA_STT_CHUNK_SECS</key>
    <string>$(plist_escape "$STT_CHUNK_SECS")</string>
    <key>HOST_MEDIA_TTS_VOICE</key>
    <string>$(plist_escape "$TTS_VOICE")</string>
    <key>HOST_MEDIA_STT_FALLBACK_URL</key>
    <string>$(plist_escape "$STT_FALLBACK_URL")</string>
    <key>HOST_MEDIA_TTS_FALLBACK_URL</key>
    <string>$(plist_escape "$TTS_FALLBACK_URL")</string>
    <key>HOST_MEDIA_FALLBACK_TIMEOUT_SECS</key>
    <string>$(plist_escape "$FALLBACK_TIMEOUT_SECS")</string>
    <key>PATH</key>
    <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>StandardOutPath</key>
  <string>$(plist_escape "$LOG_DIR/host-media-bridge.stdout.log")</string>
  <key>StandardErrorPath</key>
  <string>$(plist_escape "$LOG_DIR/host-media-bridge.stderr.log")</string>
</dict>
</plist>
PLIST

launchctl bootout "gui/$(id -u)/$LABEL" >/dev/null 2>&1 || true
launchctl bootstrap "gui/$(id -u)" "$PLIST"
launchctl kickstart -k "gui/$(id -u)/$LABEL"
launchctl print "gui/$(id -u)/$LABEL" | sed -n '1,80p'

echo "Installed $PLIST"
