# Live transcription — latency profile (measured, no changes applied)

Real-time-paced measurement on this Intel-Mac CPU. "Emission lag" = wall time a
token appears on the wire minus the audio time of that token = what the user
perceives as "behind real speech".

## Headline numbers (current config)

| Regime | Current | Model-inherent floor |
|---|---:|---:|
| Continuous speech — emission lag (mean) | **0.63 s** (p50 0.60, p95 0.91) | ~0.41 s |
| Turn start — first word of a new speaker | **~1.9 s** | = diarizer `window_sec` (1.5 s) + 1 tick |
| Sherpa decode compute / tick | 54 ms (p95 ~96 ms) | — |

Two different problems: (a) a steady ~0.6 s lag while someone talks, and (b) a
~1.9 s wait for the first word of each turn. (a) is pure ASR and safely tunable;
(b) is gated by the SpeechBrain speaker-ID window (out of scope to change).

## Decode-interval + chunk sweep (measured)

| client chunk | asr_tick | mean lag | p50 | p95 |
|---:|---:|---:|---:|---:|
| 64 ms | **0.45 s (current)** | 0.656 | 0.642 | 0.917 |
| 64 ms | 0.30 s | 0.519 | 0.526 | 0.684 |
| 64 ms | 0.20 s | 0.488 | 0.503 | 0.629 |
| 64 ms | 0.15 s | 0.468 | 0.474 | 0.609 |
| 64 ms | 0.10 s | 0.446 | 0.444 | 0.607 |
| 32 ms | 0.20 s | 0.460 | 0.472 | 0.607 |
| 32 ms | 0.10 s | **0.411** | 0.424 | 0.570 |

Floor ≈ **0.41 s** = model lookahead (chunk-16-left-128) + decode compute. Cannot
go lower without a different (smaller-lookahead) model.

## Per-source breakdown

| # | Source | Current | Delay it adds | Why | Lowest safe | Est. improvement |
|---|---|---|---|---|---|---|
| 1 | **Decode interval** `asr_tick_sec` | 0.45 s | up to 0.45 s staleness (avg ½ tick) | Whisper needed big chunks; Sherpa doesn't | **0.20 s** (0.15 aggressive) | **−0.15…0.19 s** |
| 2 | **`asr_min_chunk_sec` gate** | 0.30 s | blocks decodes when tick < 0.30 → skips | legacy Whisper gate | **0.10 s** | enables #1 (else low tick does nothing) |
| 3 | **`asr_worker` sleep-BEFORE-decode** | sleeps a full tick first | +≤1 tick on first/each emission | loop orders `sleep` then `decode` | decode-then-sleep | −up to 0.1–0.2 s tail |
| 4 | **Client chunk** (worklet) | 1024 spl = 64 ms | up to 64 ms buffering | postMessage batching | **512 spl = 32 ms** | **−0.03…0.05 s** |
| 5 | **Model left/right context** | chunk-16-left-128 | ~0.41 s floor | baked into ONNX export | keep (swap = new model) | 0 (floor) |
| 6 | **Endpoint detection** | OFF | 0 | already disabled | keep OFF | 0 |
| 7 | **VAD buffering** (diarizer) | 512-frame = 32 ms | 32 ms, fast path only | Silero requires 512 | keep | 0 (doesn't gate ASR) |
| 8 | **Ring buffer** (RollingAudio) | slices immediately | 0 | no re-buffering | keep | 0 |
| 9 | **WebSocket emission** | = `asr_tick` cadence | tied to #1 | localhost < 1 ms | falls out of #1 | (included in #1) |
| 10 | **Frontend render** | no throttle, upsert on message | ≤1 frame (~16 ms) | direct DOM update | keep | ~0 |
| 11 | **Block-open gate** (turn start) | `window_sec` 1.5 s | ~1.5 s to first word of a turn | diarizer needs a full window to ID the speaker | keep (diarization) | optional/out of scope |

## Recommendation (ASR-only, diarization untouched)

Apply together (2 must drop with 1):

- `asr_tick_sec`: 0.45 → **0.20**
- `asr_min_chunk_sec`: 0.30 → **0.10**
- client worklet chunk: 1024 → **512** samples
- `asr_worker`: reorder to **decode-then-sleep**

**Expected result:** continuous-speech lag **0.63 s → ~0.46–0.49 s** (≈ −0.15–0.19 s,
25–30%), p95 **0.91 s → ~0.61 s** (−0.30 s). Turn-start first word **~1.9 s → ~1.65 s**
(from the smaller tick). Accuracy unchanged (same model, same audio). CPU rises
modestly (more frequent 54 ms decodes); 0.20 s keeps comfortable headroom, 0.10 s
approaches the floor but uses more CPU — 0.15–0.20 s is the sweet spot.

Not recommended (would trade accuracy/stability): lowering `window_sec` (faster
turn-start but weaker speaker ID), or swapping to a smaller-lookahead model.

---

# APPLIED optimizations — before/after (measured)

Changes: `asr_tick_sec` 0.45→**0.20**, `asr_min_chunk_sec` 0.30→**0.10**, client
chunk 1024→**512** (32 ms), ASR worker **decode-then-sleep**. All env-overridable
(`ASR_TICK_SEC`, `ASR_MIN_CHUNK_SEC`, `CLIENT_CHUNK_SAMPLES`). `window_sec`
unchanged (1.5). Same model → identical transcription output.

| Metric | Before | After | Change |
|---|---:|---:|---:|
| **Time to First Word** (turn start) | 1.9 s | **1.5 s** | −0.4 s |
| **Mean emission latency** | 0.63 s | **0.47 s** | −26 % |
| **P95 emission latency** | 0.91 s | **0.62 s** | −32 % |
| **Decode compute / tick** | 54 ms | 26 ms | smaller chunks/tick |
| **ASR real-time factor** | 0.144 | **0.128** | 7.8× realtime headroom |
| **Peak RSS** | ~582 MB | ~582 MB | unchanged |
| Live partial updates (8 s clip) | 3 | 16 | smoother |

The TTFW win (−0.4 s) is a free bonus from decode-then-sleep: the first partial of
a turn now fires at block-open (1.6 s) instead of waiting an extra tick. Emission
lag is now ~0.05 s above the model floor (~0.42 s).

## Optional further optimizations (NOT applied — no quality impact)

1. **Dedicated single-thread executor for the ASR worker.** — TESTED & REVERTED.
   A concurrency benchmark (`poc/executor_bench.py`, feed on default pool vs
   asr_tick on shared pool vs a dedicated 1-thread executor, real-time paced under
   full diarizer load) showed **no measurable difference** (asr_tick p95 and
   emission-lag p95 varied more run-to-run than between the two modes). Reason:
   the default pool has ample threads AND both torch (ECAPA) and onnxruntime
   (Sherpa) release the GIL during native compute, so ASR already runs truly
   parallel to the diarizer — there is nothing to isolate on this hardware. Per the
   "keep only if measurable" rule, reverted. Would only help on a saturated pool
   (many concurrent sessions/uploads) or a GIL-bound/single-core host.
2. **Client chunk 512→256 (16 ms).** ~−0.02 s mean, at 2× WebSocket message rate.
   Marginal; only worth it if you want the last few ms.
3. **`SHERPA_NUM_THREADS` 2→3.** Cuts decode compute (~26→~18 ms), trimming the
   tail slightly; costs a core shared with diarization. Hardware-dependent.
4. **Lower-latency ASR model** (e.g. chunk-8 streaming Zipformer) would cut the
   ~0.42 s floor — but it changes the model/output, so it is out of scope by your
   accuracy constraint. Listed for completeness only.
