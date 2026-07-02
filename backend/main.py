"""FastAPI application: enrollment REST API + real-time diarization WebSocket."""
from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import settings, TUNABLE_FIELDS, BASE_DIR
from . import audio as audio_utils
from . import embeddings, vad, exporters, sherpa_asr
from .enrollment import store
from .session import LiveSession
from .batch import iter_process_file
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
    sherpa_asr.warmup()   # single ASR engine: Sherpa-ONNX (live + file upload)
    print(f"[startup] ready (Sherpa-ONNX ASR). "
          f"enrolled speakers: {len(store.list())}")


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


# ---------------------------------------------------------------- REST: file upload

@app.post("/api/transcribe-file")
async def transcribe_file(file: UploadFile = File(...)):
    """Diarize + transcribe an uploaded recording, streaming NDJSON progress.

    Each line is one JSON object: progress updates ({stage,pct,...}) and finally
    {stage:"done", segments:[...]}. The client reads the stream incrementally.
    """
    raw = await file.read()
    filename = file.filename or "upload"

    def gen():
        try:
            for msg in iter_process_file(raw, filename):
                yield json.dumps(msg) + "\n"
        except Exception as e:  # pragma: no cover
            yield json.dumps({"stage": "error", "message": str(e)}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


# ---------------------------------------------------------------- REST: export

class ExportRequest(BaseModel):
    format: str
    segments: list[dict[str, Any]]
    filename: str | None = None


@app.post("/api/export")
async def export_transcript(req: ExportRequest):
    try:
        content, media = exporters.export(req.segments, req.format)
    except KeyError:
        raise HTTPException(400, f"Unknown export format '{req.format}'")
    base = (req.filename or "transcript").rsplit(".", 1)[0]
    name = f"{base}.{req.format.lower()}"
    return Response(
        content=content, media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


# ---------------------------------------------------------------- WebSocket: live

@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket):
    """Real-time streaming diarization + transcription.

    Two cooperating executor tasks: the receive loop runs the diarizer (fast) and
    the ASR worker runs the live ASR engine (Sherpa-ONNX by default) on the open
    speaker block (heavy). Outbound sends are serialized. See backend/session.py.
    """
    await ws.accept()
    loop = asyncio.get_event_loop()
    client_sr = settings.sample_rate
    session: LiveSession | None = None
    stop_flag = asyncio.Event()
    send_lock = asyncio.Lock()
    asr_task: asyncio.Task | None = None

    async def send(events: list[dict]) -> None:
        if not events:
            return
        async with send_lock:
            for ev in events:
                await ws.send_json(ev)

    async def asr_worker() -> None:
        try:
            while not stop_flag.is_set():
                await asyncio.sleep(settings.asr_tick_sec)
                if session is None:
                    continue
                evs = await loop.run_in_executor(None, session.asr_tick)
                await send(evs)
        except Exception:  # pragma: no cover
            pass

    await ws.send_json({"type": "ready", "speakers": len(store.snapshot())})
    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("text") is not None:
                data = json.loads(msg["text"])
                if data.get("type") == "start":
                    client_sr = int(data.get("sample_rate", settings.sample_rate))
                    session = LiveSession(store.snapshot())
                    stop_flag.clear()
                    if asr_task is None or asr_task.done():
                        asr_task = asyncio.create_task(asr_worker())
                elif data.get("type") == "stop" and session is not None:
                    await send(await loop.run_in_executor(None, session.flush))
                    for _ in range(200):  # drain finalize queue
                        evs = await loop.run_in_executor(None, session.asr_tick)
                        if not evs:
                            break
                        await send(evs)
                    await ws.send_json({"type": "stopped"})
                continue
            if msg.get("bytes") is not None and session is not None:
                wav = audio_utils.pcm16_to_float32(msg["bytes"])
                if client_sr != settings.sample_rate:
                    wav = audio_utils.resample(wav, client_sr, settings.sample_rate)
                await send(await loop.run_in_executor(None, session.feed, wav))
    except WebSocketDisconnect:
        pass
    except Exception as e:  # pragma: no cover
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
    finally:
        stop_flag.set()
        if asr_task is not None:
            try:
                await asr_task
            except Exception:
                pass


# ---------------------------------------------------------------- static frontend

if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
