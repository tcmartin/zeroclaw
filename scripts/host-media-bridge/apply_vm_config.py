#!/usr/bin/env python3
"""Patch an existing ZeroClaw config.toml to use Host Media Bridge.

This intentionally performs a narrow table/key update instead of regenerating
the whole TOML file so encrypted secrets and user formatting are preserved.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


def set_key(lines: list[str], table: str, key: str, value: str) -> list[str]:
    header = f"[{table}]"
    table_start = None
    for i, line in enumerate(lines):
        if line.strip() == header:
            table_start = i
            break

    if table_start is None:
        if lines and lines[-1].strip():
            lines.append("\n")
        lines.extend([f"{header}\n", f"{key} = {value}\n"])
        return lines

    table_end = len(lines)
    for i in range(table_start + 1, len(lines)):
        stripped = lines[i].strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            table_end = i
            break

    for i in range(table_start + 1, table_end):
        stripped = lines[i].lstrip()
        if stripped.startswith(f"{key} ") or stripped.startswith(f"{key}="):
            prefix_len = len(lines[i]) - len(stripped)
            lines[i] = " " * prefix_len + f"{key} = {value}\n"
            return lines

    lines.insert(table_end, f"{key} = {value}\n")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="~/.zeroclaw/config.toml")
    parser.add_argument("--host-url", required=True, help="Base URL, e.g. http://192.168.64.1:5010")
    parser.add_argument("--max-audio-bytes", type=int, default=500 * 1024 * 1024)
    parser.add_argument("--timeout-secs", type=int, default=900)
    parser.add_argument("--voice", default="af_heart")
    parser.add_argument("--format", default="wav")
    args = parser.parse_args()

    config = Path(args.config).expanduser()
    lines = config.read_text().splitlines(keepends=True)
    host_url = args.host_url.rstrip("/")

    updates = [
        ("transcription", "enabled", "true"),
        ("transcription", "default_provider", quote("local_whisper")),
        ("transcription", "api_url", quote(f"{host_url}/v1/audio/transcriptions")),
        ("transcription", "model", quote("parakeet-tdt-0.6b-v2")),
        (
            "transcription.local_whisper",
            "url",
            quote(f"{host_url}/v1/audio/transcriptions"),
        ),
        ("transcription.local_whisper", "max_audio_bytes", str(args.max_audio_bytes)),
        ("transcription.local_whisper", "timeout_secs", str(args.timeout_secs)),
        ("tts", "enabled", "true"),
        ("tts", "default_provider", quote("piper")),
        ("tts", "default_voice", quote(args.voice)),
        ("tts", "default_format", quote(args.format)),
        ("tts.piper", "api_url", quote(f"{host_url}/v1/audio/speech")),
    ]

    for table, key, value in updates:
        lines = set_key(lines, table, key, value)

    backup = config.with_suffix(config.suffix + ".host-media-backup")
    if not backup.exists():
        backup.write_text(config.read_text())
    config.write_text("".join(lines))
    print(f"Updated {config}")
    print(f"Backup: {backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
