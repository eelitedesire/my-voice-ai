# VoiceID — Real-time Speaker Diarization & Recognition

Production-grade speaker diarization + speaker identification with live
transcription, built on **SpeechBrain (ECAPA-TDNN)**, **Silero VAD**, and
**faster-whisper**. Enroll two or more speakers through the web UI, then stream
your microphone and see who is speaking, in real time, with their name attached
to the transcript. Anyone not enrolled (or below the confidence threshold) is
labeled **“Unknown Speaker.”**

## Features

- **Web enrollment** — record or upload multiple voice samples per person, named.
- **Robust embeddings** — ECAPA-TDNN 192-d embeddings, L2-normalized, with
  loudness normalization so different mics/levels stay comparable.
- **Real-time diarization** — Silero VAD + sliding-window embedding extraction
  detects speaker turns as you talk.
- **Speaker identification** — cosine similarity vs. every enrolled profile
  (`max`-over-samples by default; `centroid`/`mean` also available).
- **Unknown handling** — a configurable `id_threshold` labels low-confidence
  speech as *Unknown Speaker* instead of forcing a wrong match.
- **Stability / anti-flicker** — EMA score smoothing + hysteresis (switch margin
  and consecutive-window requirement) prevent identity switching mid-conversation.
- **Live transcription** — each finalized speaker turn is transcribed and shown
  with the speaker’s name.
- **N speakers, no rearchitecting** — enroll 2, 5, 50; the identifier just scores
  against however many profiles exist.
- **Fully tunable at runtime** — every threshold is editable from the UI / API.

## Requirements

- **Python 3.11** (torch/speechbrain/faster-whisper have **no** wheels for 3.14).
  `brew install python@3.11` if needed.
- macOS/Linux. CPU inference works out of the box (this repo targets Intel Mac).
- A microphone + a **secure context** for the browser mic API: `localhost` is
  fine; a remote host needs HTTPS.

## Setup & run

```bash
./setup.sh      # creates .venv (py3.11) and installs deps  (downloads torch)
./run.sh        # starts the server on http://127.0.0.1:8000
```

First launch downloads the models (ECAPA ~80 MB, Silero small, Whisper base ~140 MB)
into `models/`. Then:

1. Open **http://127.0.0.1:8000/enroll.html** and enroll ≥ 2 speakers.
2. Open **http://127.0.0.1:8000/live.html**, press **Start**, and speak.

## Configuration

All knobs live in [`backend/config.py`](backend/config.py) and can be set via
environment variables or the `/api/config` endpoint (and the live UI “Tuning”
panel). Key ones:

| Setting | Default | Meaning |
|---|---|---|
| `ID_THRESHOLD` | `0.35` | Below this cosine similarity → *Unknown Speaker*. |
| `SCORING` | `max` | `max` / `mean` over enrolled samples, or `centroid`. |
| `SWITCH_MARGIN` | `0.06` | Challenger must beat incumbent by this to switch. |
| `MIN_SWITCH_WINDOWS` | `2` | …for this many consecutive windows. |
| `EMA_ALPHA` | `0.6` | Score smoothing (lower = smoother/stickier). |
| `WINDOW_SEC` / `HOP_SEC` | `1.5` / `0.75` | Embedding window / step. |
| `MIN_SEGMENT_SEC` | `0.8` | Shortest turn that gets emitted. |
| `VAD_THRESHOLD` | `0.5` | Silero speech sensitivity. |
| `WHISPER_MODEL` | `base.en` | `tiny(.en)`/`base(.en)`/`small(.en)`. |
| `ENABLE_ASR` | `1` | Set `0` to disable transcription (lower latency). |

Example: stricter identity + faster ASR:
```bash
ID_THRESHOLD=0.45 WHISPER_MODEL=tiny.en ./run.sh
```

## Architecture

```
Browser (AudioWorklet, 16 kHz PCM) ──WS──► FastAPI
  enroll.html / live.html                    │
                                             ├─ Silero VAD  (speech/turn detection)
                                             ├─ ECAPA-TDNN  (speaker embeddings)
                                             ├─ Diarizer    (windows + hysteresis ID)
                                             └─ faster-whisper (per-turn ASR)
```

- [`backend/embeddings.py`](backend/embeddings.py) — ECAPA singleton, cosine utils.
- [`backend/vad.py`](backend/vad.py) — Silero VAD (offline + streaming).
- [`backend/enrollment.py`](backend/enrollment.py) — profile store, centroids, persistence.
- [`backend/diarization.py`](backend/diarization.py) — streaming turn detection, ID, stability.
- [`backend/transcription.py`](backend/transcription.py) — faster-whisper wrapper.
- [`backend/main.py`](backend/main.py) — REST + WebSocket + static frontend.

## Tips for best accuracy

- Enroll **3–5 varied samples** per person (10–30 s total), ideally on the same
  mic they’ll use live, but include some variation in speaking style.
- If two enrolled voices get confused, raise `SWITCH_MARGIN` / `MIN_SWITCH_WINDOWS`.
- If real speakers are wrongly marked *Unknown*, lower `ID_THRESHOLD` (or add more
  enrollment samples). If strangers get matched, raise it.
- `small.en` Whisper improves transcription at the cost of latency on CPU.

## Notes

- Enrollments persist under `data/speakers/`. Delete a speaker from the UI.
- The WebSocket runs one diarizer per connection; heavy work (embeddings, ASR)
  runs in a thread pool so the event loop stays responsive.
# my-voice-ai
