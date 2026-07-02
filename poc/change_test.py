#!/usr/bin/env python3
"""Speaker-change detection test. Concatenates two speakers' audio with NO silence
gap (so VAD sees one turn) and checks the diarizer splits it acoustically — for
every combination of known/unknown — while NOT over-splitting a single speaker."""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
from pathlib import Path
import numpy as np, soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
VOICES = Path("/private/tmp/claude-501/-Users-elite-Downloads-my-voice-ai/"
              "a4452a18-9ee2-431c-99e7-ee1f665b189f/scratchpad/voices")
SR = 16000

from backend.config import settings
from backend.enrollment import ProfileSnapshot
from backend.diarization import StreamingDiarizer
from backend import embeddings
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


PROF = profiles()


def run(clips):
    """Feed trimmed, gapless concatenation; return list of segment speakers."""
    audio = np.concatenate([trim_to_speech(load(c)) for c in clips])
    diar = StreamingDiarizer(PROF)
    segs = []
    for i in range(0, len(audio), 1600):
        for ev in diar.process(audio[i:i + 1600]):
            if ev["type"] == "segment":
                segs.append((ev["speaker"], round(ev["end"] - ev["start"], 1)))
    for ev in diar.flush():
        if ev["type"] == "segment":
            segs.append((ev["speaker"], round(ev["end"] - ev["start"], 1)))
    return segs


CASES = [
    ("K->U   Daniel then unknown(Fred)",   ["dan_test.wav", "fred_test.wav"], ["Daniel", "Unknown Speaker"]),
    ("U->K   unknown(Fred) then Daniel",   ["fred_test.wav", "dan_test.wav"], ["Unknown Speaker", "Daniel"]),
    ("U->U   Fred then Kathy (2 unknowns)", ["fred_test.wav", "kathy_test.wav"], ["Unknown Speaker", "Unknown Speaker"]),
    ("K->K   Daniel then Samantha",        ["dan_test.wav", "sam_test.wav"], ["Daniel", "Samantha"]),
    ("same   Daniel only (no false split)", ["dan_test.wav", "danB.wav"], ["Daniel"]),
    ("same   Fred only (no false split)",   ["fred_test.wav"], ["Unknown Speaker"]),
]

print(f"change_sim_threshold={settings.change_sim_threshold}  min_switch_windows={settings.min_switch_windows}\n")
passed = 0
for name, clips, expect in CASES:
    segs = run(clips)
    got = [s for s, _ in segs]
    ok = got == expect
    passed += ok
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"        segments: {segs}")
    if not ok:
        print(f"        expected: {expect}")
print(f"\n{passed}/{len(CASES)} scenarios pass")
