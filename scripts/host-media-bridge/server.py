#!/usr/bin/env python3
"""Host-side STT/TTS bridge for ZeroClaw VMs.

The bridge is deliberately OpenAI-compatible so existing ZeroClaw config can
point `[transcription.local_whisper].url` and `[tts.piper].api_url` at it.
"""

from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Optional

import requests
import soundfile as sf
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel


HOST = os.environ.get("HOST_MEDIA_HOST", "0.0.0.0")
PORT = int(os.environ.get("HOST_MEDIA_PORT", "5010"))
STT_MODEL = os.environ.get("HOST_MEDIA_STT_MODEL", "mlx-community/parakeet-tdt-0.6b-v2")
STT_CHUNK_SECS = int(os.environ.get("HOST_MEDIA_STT_CHUNK_SECS", "90"))
STT_FALLBACK_URL = os.environ.get("HOST_MEDIA_STT_FALLBACK_URL", "").strip()
TTS_FALLBACK_URL = os.environ.get("HOST_MEDIA_TTS_FALLBACK_URL", "").strip()
FALLBACK_TIMEOUT_SECS = int(os.environ.get("HOST_MEDIA_FALLBACK_TIMEOUT_SECS", "30"))
TTS_VOICE = os.environ.get("HOST_MEDIA_TTS_VOICE", "af_heart")
TMP_ROOT = Path(os.environ.get("HOST_MEDIA_TMPDIR", "~/.cache/zeroclaw-host-media/tmp")).expanduser()

app = FastAPI(title="ZeroClaw Host Media Bridge", version="0.1.0")
_stt_model = None
_tts_pipeline = None


def _ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required and was not found on PATH")


def _load_stt():
    global _stt_model
    if _stt_model is None:
        from parakeet_mlx import from_pretrained

        print(f"[host-media] loading STT {STT_MODEL}", flush=True)
        _stt_model = from_pretrained(STT_MODEL)
        print("[host-media] STT loaded", flush=True)
    return _stt_model


def _load_tts():
    global _tts_pipeline
    if _tts_pipeline is None:
        from kokoro import KPipeline

        print("[host-media] loading Kokoro TTS", flush=True)
        _tts_pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
        print("[host-media] Kokoro TTS loaded", flush=True)
    return _tts_pipeline


def _split_to_wav(audio_path: Path, out_dir: Path) -> list[Path]:
    _ensure_ffmpeg()
    pattern = out_dir / "chunk_%04d.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(audio_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "segment",
            "-segment_time",
            str(STT_CHUNK_SECS),
            "-reset_timestamps",
            "1",
            "-c:a",
            "pcm_s16le",
            str(pattern),
        ],
        check=True,
    )
    return sorted(out_dir.glob("chunk_*.wav"))


def _duration(path: Path) -> float:
    _ensure_ffmpeg()
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "csv=p=0",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def _fallback_stt(file_bytes: bytes, filename: str, model_name: str, language: Optional[str]):
    if not STT_FALLBACK_URL:
        raise
    files = {"file": (filename, file_bytes)}
    data = {"model": model_name, "response_format": "json"}
    if language:
        data["language"] = language
    resp = requests.post(STT_FALLBACK_URL, files=files, data=data, timeout=FALLBACK_TIMEOUT_SECS)
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/json"),
        headers={"X-Host-Media-Backend": "stt-fallback"},
    )


@app.post("/v1/audio/transcriptions")
async def transcribe_upload(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(default="parakeet-tdt-0.6b-v2", alias="model"),
    language: Optional[str] = Form(default=None),
    response_format: Optional[str] = Form(default="json"),
):
    file_bytes = await file.read()
    filename = file.filename or "audio.wav"
    suffix = Path(filename).suffix or ".wav"
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="stt_", dir=TMP_ROOT))
    source = work_dir / f"audio{suffix}"
    source.write_bytes(file_bytes)
    t0 = time.time()

    try:
        model = _load_stt()
        duration = _duration(source)
        chunks = _split_to_wav(source, work_dir)
        texts = []
        for idx, chunk in enumerate(chunks, start=1):
            ct0 = time.time()
            result = model.transcribe(str(chunk))
            text = getattr(result, "text", str(result)).strip()
            texts.append(text)
            print(
                f"[host-media] STT {filename} chunk {idx}/{len(chunks)} "
                f"chars={len(text)} elapsed={time.time() - ct0:.1f}s",
                flush=True,
            )
        text = "\n\n".join(part for part in texts if part).strip()
        elapsed = time.time() - t0
        print(
            f"[host-media] STT {filename}: {duration:.1f}s -> {len(text)} chars "
            f"in {elapsed:.1f}s",
            flush=True,
        )
    except Exception as exc:
        print(f"[host-media] STT host failed for {filename}: {exc}", flush=True)
        if STT_FALLBACK_URL:
            return _fallback_stt(file_bytes, filename, model_name or "parakeet", language)
        raise HTTPException(status_code=500, detail=f"STT error: {exc}") from exc
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    if response_format == "text":
        return Response(
            content=text,
            media_type="text/plain",
            headers={"X-Host-Media-Backend": "parakeet-mlx"},
        )
    return JSONResponse(
        {
            "text": text,
            "duration": round(duration, 2),
            "language": language or "en",
            "model": STT_MODEL,
            "backend": "parakeet-mlx",
            "chunks": len(chunks),
        },
        headers={"X-Host-Media-Backend": "parakeet-mlx"},
    )


class SpeechRequest(BaseModel):
    model: Optional[str] = "kokoro-82m"
    input: str
    voice: Optional[str] = None
    response_format: Optional[str] = "wav"
    speed: Optional[float] = 1.0


def _fallback_tts(req: SpeechRequest):
    if not TTS_FALLBACK_URL:
        raise
    resp = requests.post(TTS_FALLBACK_URL, json=req.model_dump(), timeout=FALLBACK_TIMEOUT_SECS)
    return Response(
        content=resp.content,
        status_code=resp.status_code,
        media_type=resp.headers.get("content-type", "application/octet-stream"),
        headers={"X-Host-Media-Backend": "tts-fallback"},
    )


@app.post("/v1/audio/speech")
async def synthesize(req: SpeechRequest):
    t0 = time.time()
    voice = req.voice or TTS_VOICE
    fmt = (req.response_format or "wav").lower()
    try:
        pipeline = _load_tts()
        chunks = []
        for result in pipeline(req.input, voice=voice, speed=req.speed or 1.0):
            if result.output is not None:
                audio = result.output.audio
                if hasattr(audio, "detach"):
                    audio = audio.detach().cpu().numpy()
                elif hasattr(audio, "cpu"):
                    audio = audio.cpu().numpy()
                chunks.append(audio)
        if not chunks:
            raise RuntimeError("Kokoro returned no audio")

        import numpy as np

        audio_data = np.concatenate(chunks)
        wav = io.BytesIO()
        sf.write(wav, audio_data, 24000, format="WAV")
        wav.seek(0)
        wav_bytes = wav.read()

        if fmt == "wav":
            body = wav_bytes
            media_type = "audio/wav"
        elif fmt == "mp3":
            _ensure_ffmpeg()
            TMP_ROOT.mkdir(parents=True, exist_ok=True)
            work_dir = Path(tempfile.mkdtemp(prefix="tts_", dir=TMP_ROOT))
            wav_path = work_dir / "speech.wav"
            mp3_path = work_dir / "speech.mp3"
            try:
                wav_path.write_bytes(wav_bytes)
                subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        str(wav_path),
                        "-codec:a",
                        "libmp3lame",
                        "-qscale:a",
                        "2",
                        str(mp3_path),
                    ],
                    check=True,
                )
                body = mp3_path.read_bytes()
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)
            media_type = "audio/mpeg"
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported TTS format: {fmt}")

        print(
            f"[host-media] TTS {len(req.input)} chars -> {len(body)} bytes "
            f"in {time.time() - t0:.1f}s",
            flush=True,
        )
        return Response(
            content=body,
            media_type=media_type,
            headers={"X-Host-Media-Backend": "kokoro"},
        )
    except HTTPException:
        raise
    except Exception as exc:
        print(f"[host-media] TTS host failed: {exc}", flush=True)
        if TTS_FALLBACK_URL:
            return _fallback_tts(req)
        raise HTTPException(status_code=500, detail=f"TTS error: {exc}") from exc


@app.get("/v1/models")
async def models():
    return {
        "models": [
            {"id": STT_MODEL, "type": "stt"},
            {"id": "kokoro-82m", "type": "tts"},
        ]
    }


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "stt_model": STT_MODEL,
        "tts_model": "kokoro-82m",
        "stt_fallback": bool(STT_FALLBACK_URL),
        "tts_fallback": bool(TTS_FALLBACK_URL),
    }


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
