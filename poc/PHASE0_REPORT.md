# Phase 0 — Sherpa-ONNX Streaming Zipformer coexistence spike

**Question:** Can Sherpa-ONNX Streaming Zipformer coexist with the SpeechBrain
diarizer and deliver significantly lower-latency live transcription **without
affecting diarization**?

**Answer: YES.** Coexistence works, diarization output is byte-identical, and
Sherpa is ~36× more CPU-efficient than the current Whisper streaming path with
sub-second time-to-first-word.

Everything here is isolated in `poc/` — no change to `backend/`, `frontend/`,
or the Whisper pipeline. Sherpa is loaded only in `--mode combined`.

## Environment findings

- **sherpa-onnx must be pinned to `1.10.46` on this machine.** Newer wheels
  (1.11+) bundle an onnxruntime built for **macOS ≥15**; this box is **macOS
  13.7 (Ventura)** and they fail to load (`Symbol not found: _MLComputePlan`).
  1.10.46 imports and runs fine.
- `KMP_DUPLICATE_LIB_OK=TRUE` set as a precaution; no OpenMP crash observed with
  torch 2.2.2 + sherpa 1.10.46 in one process.
- Model: `sherpa-onnx-streaming-zipformer-en-2023-06-26`, int8 (same as the
  reference project).

## 1. Coexistence — PASS

`--mode combined` loaded SpeechBrain (torch/ECAPA/Silero) **and** Sherpa
(onnxruntime) in one process and ran the full pipeline to completion. No OMP
clash, no segfault.

## 2. Diarization unchanged — IDENTICAL ✅

Same 14.5 s / 3-speaker conversation, diarizer-only vs diarizer+Sherpa:

```
BASELINE                              COMBINED
[0.06-4.26] Daniel   conf 0.663       [0.06-4.26] Daniel   conf 0.663
[4.61-9.47] Samantha conf 0.610       [4.61-9.47] Samantha conf 0.610
[9.82-14.4] Unknown  conf 0.350       [9.82-14.4] Unknown  conf 0.350
```

Byte-for-byte identical (incl. Unknown detection + confidence). Sherpa running in
the same process does not perturb diarization.

## 3. Benchmarks (same machine, same audio)

| Metric | **Whisper base.en (current)** | **Sherpa Zipformer int8 (POC)** |
|---|---:|---:|
| ASR real-time factor (CPU / audio-sec) | **5.20** (5× *slower* than realtime) | **0.144** (7× *faster*) |
| Per-decode latency — mean | **2062 ms** | ~5–15 ms (per 100 ms chunk) |
| Per-decode latency — p95 / max | 4426 / 5537 ms | — |
| Time to First Word (perceived) | ~1.0 s audio **+ ~2 s decode ≈ 3 s** | ~0.8 s audio **+ negligible ≈ 0.8 s** |
| Partial cadence | ~2 s, decode-bound, falls behind | ~0.3–0.4 s, smooth, monotonic |
| Final text | `let me walk you through the numbers before we make any final decision.` (cased + punctuated; minor dup) | `LET ME WALK YOU THROUGH THE NUMBERS BEFORE WE MAKE ANY FINAL DECISION` (UPPERCASE, no punctuation) |
| Combined RTF (diar + ASR) | diar 0.245 + Whisper 5.20 ≈ **5.4 (cannot keep up)** | diar 0.245 + Sherpa 0.144 = **0.389 (2.6× realtime headroom)** |
| End-to-end per 100 ms chunk (combined) | — | mean **38 ms**, p95 285 ms, max 341 ms* |
| Peak RSS | ~430 MB (diar) + Whisper | **645.8 MB** (diar + Sherpa + ORT) |

\* the 285–341 ms spikes are the **diarizer's ECAPA embedding** windows (every
~0.75 s), *not* Sherpa. Sherpa's own per-chunk cost is a few ms.

### Headline
On this Intel-Mac CPU the **current Whisper live path is 5× slower than
real-time** — each partial re-decodes the whole growing buffer (~2 s per update),
so text lags by seconds and accumulates delay on long turns. **Sherpa is 7×
faster than real-time** with **~0.8 s TTFW** and fluid token-by-token growth —
the ChatGPT-Voice feel, e.g.:

```
0.80s  LET
1.20s  LET ME WALK
1.50s  LET ME WALK YOU THROUGH
1.80s  LET ME WALK YOU THROUGH THE NUM
2.10s  LET ME WALK YOU THROUGH THE NUMBERS
...
```

## Caveats / notes

- **Single-stream POC artifact:** to measure raw ASR latency, the POC fed the
  whole conversation to **one** Sherpa stream, so its partial text ran across all
  three speakers. This is *not* an integration problem — Phase 1 uses **one
  Sherpa stream per diarizer block** (the block model already in `session.py`),
  which resets at each speaker boundary and keeps text per-speaker.
- **No punctuation / casing** from streaming Zipformer (as predicted). Phase 1
  should add a punctuation-restoration pass on finalize for polish.
- **Accuracy:** Zipformer transcribed cleanly here; Whisper still likely edges it
  on noisy/accented audio and gives punctuation — keep Whisper for the offline
  file/export path.
- **Memory:** +~216 MB for Sherpa/ORT; +~150–200 MB more if Whisper stays loaded
  for batch.

## Verdict

**Proceed to Phase 1.** Coexistence is proven, diarization is untouched, and the
latency win is large and decisive on this hardware. Recommended constraints:
pin `sherpa-onnx==1.10.46` (or move to macOS ≥15), add punctuation restoration,
and keep Whisper for offline batch/export.

## How to reproduce

```bash
.venv/bin/python poc/phase0_poc.py --mode baseline --out poc/baseline.json
.venv/bin/python poc/phase0_poc.py --mode combined --out poc/combined.json
.venv/bin/python poc/phase0_poc.py --mode compare  --a poc/baseline.json --b poc/combined.json
.venv/bin/python poc/whisper_bench.py   # current Whisper streaming numbers
```
