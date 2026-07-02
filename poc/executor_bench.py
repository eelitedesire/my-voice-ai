#!/usr/bin/env python3
"""Measure the dedicated-ASR-executor effect: does the diarizer's ECAPA embedding
work delay Sherpa decoding when they share the default thread pool?

Mirrors the server's asyncio concurrency: feed (diarizer) on the default executor,
asr_tick (Sherpa) on either the SAME default pool ("shared") or a dedicated
single-thread executor ("dedicated"). Real-time paced. Reports, per mode:
  * asr_tick wall latency (submit->result) mean / p95   <- the pure jitter metric
  * emission lag mean / p95 (token timestamps under load)
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys, time, asyncio, statistics, resource
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
VOICES = Path("/private/tmp/claude-501/-Users-elite-Downloads-my-voice-ai/"
              "a4452a18-9ee2-431c-99e7-ee1f665b189f/scratchpad/voices")
SR = 16000

from backend.config import settings
from backend import sherpa_asr, embeddings
from backend.enrollment import ProfileSnapshot
from backend.session import LiveSession
from backend.vad import trim_to_speech
from backend.embeddings import l2_normalize


def load(f):
    w, sr = sf.read(str(VOICES / f), dtype="float32")
    return (w if w.ndim == 1 else w.mean(1)).astype(np.float32)


def profiles():
    def enr(fs):
        return np.stack([e for e in (embeddings.embed(trim_to_speech(load(f))) for f in fs) if e is not None])
    dan, sam = enr(["danA.wav", "danB.wav"]), enr(["samA.wav", "samB.wav"])
    return ProfileSnapshot(["dan", "sam"], ["Daniel", "Samantha"],
                           [l2_normalize(dan.mean(0)), l2_normalize(sam.mean(0))], [dan, sam])


def pct(xs, p):
    return round(float(np.percentile(xs, p)), 1) if xs else None


async def run(mode, conv, prof):
    session = LiveSession(prof)
    asr_exec = None if mode == "shared" else ThreadPoolExecutor(max_workers=1)
    loop = asyncio.get_event_loop()
    stop = asyncio.Event()
    tick_lat, lags = [], []
    prev_tok = {}
    start = time.perf_counter()
    rec = sherpa_asr._load()

    async def asr_loop():
        while not stop.is_set():
            t0 = time.perf_counter()
            await loop.run_in_executor(asr_exec, session.asr_tick)
            tick_lat.append((time.perf_counter() - t0) * 1000)
            blk = session.open_block
            if blk is not None and hasattr(blk.transcriber, "_stream"):
                try:
                    ts = list(rec.timestamps(blk.transcriber._stream) or [])
                except Exception:
                    ts = []
                wall = time.perf_counter() - start
                base = blk.start_sample / SR
                for k in range(prev_tok.get(blk.id, 0), len(ts)):
                    lags.append(wall - (base + ts[k]))
                prev_tok[blk.id] = len(ts)
            await asyncio.sleep(settings.asr_tick_sec)

    async def feed_loop():
        CH = settings.client_chunk_samples
        pos = 0.0
        for i in range(0, len(conv), CH):
            chunk = conv[i:i + CH]; pos += len(chunk) / SR
            while (time.perf_counter() - start) < pos:
                await asyncio.sleep(0.001)
            await loop.run_in_executor(None, session.feed, chunk)
        stop.set()

    at = asyncio.create_task(asr_loop())
    await feed_loop()
    await at
    if asr_exec:
        asr_exec.shutdown(wait=False)
    return tick_lat, lags


async def main():
    sherpa_asr.warmup()
    prof = profiles()
    gap = np.zeros(int(0.4 * SR), np.float32)
    conv = np.concatenate([load("dan_test.wav"), gap, load("sam_test.wav")])

    print(f"config: tick={settings.asr_tick_sec}s chunk={settings.client_chunk_samples} "
          f"threads={settings.sherpa_num_threads}  audio={len(conv)/SR:.1f}s\n")
    print(f"{'mode':>10} | {'asr_tick ms mean':>16} {'p95':>7} {'max':>7} | "
          f"{'emit lag mean':>13} {'p95':>7}")
    for mode in ["shared", "dedicated", "shared", "dedicated"]:  # interleave to average out noise
        tl, lg = await run(mode, conv, prof)
        print(f"{mode:>10} | {statistics.mean(tl):>16.1f} {pct(tl,95):>7} {max(tl):>7.1f} | "
              f"{statistics.mean(lg):>12.3f}s {pct([x*1000 for x in lg],95)/1000 if lg else 0:>6}s")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    print(f"\npeak RSS: {rss:.0f} MB")


if __name__ == "__main__":
    asyncio.run(main())
