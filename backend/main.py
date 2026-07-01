"""FastAPI application: enrollment REST API + real-time diarization WebSocket."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import settings, TUNABLE_FIELDS, BASE_DIR
from . import audio as audio_utils
from . import embeddings, vad, transcription
from .enrollment import store
from .diarization import StreamingDiarizer
from .schemas import (
    SpeakerListResponse, SpeakerSummary, EnrollResponse, ConfigUpdate,
)

FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="SpeechBrain Speaker Recognition", version="1.0.0")


@app.on_event("startup")
async def _startup() -> None:
    # Warm up models off the event loop so the first request/connection is fast.
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _warmup_all)


def _warmup_all() -> None:
    print("[startup] loading models...")
    vad.warmup()
    embeddings.warmup()
    transcription.warmup()
    print(f"[startup] ready. enrolled speakers: {len(store.list())}")


# ---------------------------------------------------------------- REST: speakers

@app.get("/api/speakers", response_model=SpeakerListResponse)
async def list_speakers() -> SpeakerListResponse:
    return SpeakerListResponse(
        speakers=[SpeakerSummary(**s.summary()) for s in store.list()]
    )


@app.post("/api/speakers/enroll", response_model=EnrollResponse)
async def enroll(name: str = Form(...), files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "At least one voice sample is required.")
    wavs = []
    for f in files:
        raw = await f.read()
        try:
            wav = await asyncio.get_event_loop().run_in_executor(
                None, audio_utils.decode_file, raw, f.filename or ""
            )
        except Exception as e:
            raise HTTPException(400, f"Could not decode '{f.filename}': {e}")
        wavs.append(wav)

    try:
        spk, added, skipped = await asyncio.get_event_loop().run_in_executor(
            None, store.enroll, name, wavs
        )
    except ValueError as e:
        raise HTTPException(400, str(e))

    return EnrollResponse(
        speaker=SpeakerSummary(**spk.summary()),
        added_samples=added, skipped_samples=skipped,
        message=f"Enrolled '{spk.name}' with {added} sample(s)"
                + (f", skipped {skipped} unusable" if skipped else ""),
    )


@app.delete("/api/speakers/{speaker_id}")
async def delete_speaker(speaker_id: str):
    if not store.delete(speaker_id):
        raise HTTPException(404, "Speaker not found")
    return {"deleted": speaker_id}


@app.patch("/api/speakers/{speaker_id}")
async def rename_speaker(speaker_id: str, name: str = Form(...)):
    spk = store.rename(speaker_id, name)
    if spk is None:
        raise HTTPException(404, "Speaker not found")
    return SpeakerSummary(**spk.summary())


# ---------------------------------------------------------------- REST: config

@app.get("/api/config")
async def get_config():
    return settings.public()


@app.post("/api/config")
async def update_config(update: ConfigUpdate):
    changed = {}
    for k, v in update.model_dump(exclude_none=True).items():
        if k in TUNABLE_FIELDS:
            setattr(settings, k, v)
            changed[k] = v
    return {"updated": changed, "config": settings.public()}


# ---------------------------------------------------------------- WebSocket: live

@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_event_loop()
    client_sr = settings.sample_rate
    diarizer = StreamingDiarizer(store.snapshot())
    await ws.send_json({"type": "ready", "speakers": len(store.snapshot())})

    async def emit(events: list[dict]) -> None:
        for ev in events:
            if ev["type"] == "segment":
                seg_audio = ev.pop("_audio", None)
                text = ""
                if seg_audio is not None and settings.enable_transcription:
                    text = await loop.run_in_executor(
                        None, transcription.transcribe, seg_audio
                    )
                ev["text"] = text
                if not text and settings.enable_transcription:
                    # nothing intelligible; skip empty finalized turn
                    continue
            await ws.send_json(ev)

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if "text" in msg and msg["text"] is not None:
                data = json.loads(msg["text"])
                if data.get("type") == "start":
                    client_sr = int(data.get("sample_rate", settings.sample_rate))
                    diarizer = StreamingDiarizer(store.snapshot())
                elif data.get("type") == "stop":
                    await emit(await loop.run_in_executor(None, diarizer.flush))
                continue
            if "bytes" in msg and msg["bytes"] is not None:
                wav = audio_utils.pcm16_to_float32(msg["bytes"])
                if client_sr != settings.sample_rate:
                    wav = audio_utils.resample(wav, client_sr, settings.sample_rate)
                events = await loop.run_in_executor(None, diarizer.process, wav)
                await emit(events)
    except WebSocketDisconnect:
        pass
    except Exception as e:  # pragma: no cover
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


# ---------------------------------------------------------------- static frontend

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
