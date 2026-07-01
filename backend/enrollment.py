"""Speaker enrollment store.

Each enrolled speaker keeps:
  * every per-sample embedding (enables "max over samples" scoring, which is more
    robust than a single centroid to varying speaking style / mic),
  * a centroid (mean of L2-normalized sample embeddings, re-normalized),
  * metadata (name, timestamps, sample count).

Persisted under ``data/speakers/<id>/`` as ``embeddings.npy`` + ``meta.json`` so
enrollments survive restarts. Thread-safe; the scoring path is lock-free by
snapshotting an immutable in-memory view.

Adding a 3rd, 4th ... Nth speaker requires no code change — the identifier scores
against however many centroids exist.
"""
from __future__ import annotations

import json
import time
import uuid
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import SPEAKERS_DIR
from .embeddings import embed, l2_normalize
from .vad import trim_to_speech


@dataclass
class Speaker:
    id: str
    name: str
    created_at: float
    updated_at: float
    embeddings: np.ndarray            # (N, D) L2-normalized per-sample embeddings
    centroid: np.ndarray             # (D,) L2-normalized

    @property
    def num_samples(self) -> int:
        return int(self.embeddings.shape[0])

    def summary(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "num_samples": self.num_samples,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def _compute_centroid(embs: np.ndarray) -> np.ndarray:
    if embs.shape[0] == 0:
        return np.zeros(embs.shape[1] if embs.ndim == 2 else 0, dtype=np.float32)
    return l2_normalize(embs.mean(axis=0))


class EnrollmentStore:
    def __init__(self, root: Path = SPEAKERS_DIR):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._speakers: dict[str, Speaker] = {}
        self._load_all()

    # ---------- persistence ----------

    def _load_all(self) -> None:
        for d in sorted(self.root.iterdir()) if self.root.exists() else []:
            if not d.is_dir():
                continue
            meta_p, emb_p = d / "meta.json", d / "embeddings.npy"
            if not (meta_p.exists() and emb_p.exists()):
                continue
            try:
                meta = json.loads(meta_p.read_text())
                embs = np.load(emb_p).astype(np.float32)
                self._speakers[meta["id"]] = Speaker(
                    id=meta["id"], name=meta["name"],
                    created_at=meta["created_at"], updated_at=meta["updated_at"],
                    embeddings=embs, centroid=_compute_centroid(embs),
                )
            except Exception as e:  # pragma: no cover - corrupt dir
                print(f"[enrollment] skipping {d.name}: {e}")

    def _persist(self, spk: Speaker) -> None:
        d = self.root / spk.id
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "embeddings.npy", spk.embeddings)
        (d / "meta.json").write_text(json.dumps({
            "id": spk.id, "name": spk.name,
            "created_at": spk.created_at, "updated_at": spk.updated_at,
        }, indent=2))

    # ---------- mutations ----------

    def enroll(self, name: str, wavs: list[np.ndarray]) -> tuple[Speaker, int, int]:
        """Create or extend a speaker from a list of 16 kHz mono clips.

        Returns (speaker, added, skipped). Clips too short (after silence-trimming)
        to embed reliably are skipped rather than polluting the profile.
        """
        name = name.strip()
        if not name:
            raise ValueError("Speaker name is required")

        new_embs: list[np.ndarray] = []
        skipped = 0
        for wav in wavs:
            speech = trim_to_speech(wav)
            emb = embed(speech)
            if emb is None:
                skipped += 1
                continue
            new_embs.append(emb)

        if not new_embs:
            raise ValueError(
                "No usable speech found in the provided samples. "
                "Please record clearer / longer audio."
            )

        with self._lock:
            existing = self._find_by_name(name)
            now = time.time()
            if existing is None:
                spk = Speaker(
                    id=uuid.uuid4().hex[:12], name=name,
                    created_at=now, updated_at=now,
                    embeddings=np.stack(new_embs),
                    centroid=_compute_centroid(np.stack(new_embs)),
                )
            else:
                stacked = np.concatenate([existing.embeddings, np.stack(new_embs)])
                spk = Speaker(
                    id=existing.id, name=existing.name,
                    created_at=existing.created_at, updated_at=now,
                    embeddings=stacked, centroid=_compute_centroid(stacked),
                )
            self._speakers[spk.id] = spk
            self._persist(spk)
        return spk, len(new_embs), skipped

    def delete(self, speaker_id: str) -> bool:
        with self._lock:
            if speaker_id not in self._speakers:
                return False
            del self._speakers[speaker_id]
            shutil.rmtree(self.root / speaker_id, ignore_errors=True)
            return True

    def rename(self, speaker_id: str, new_name: str) -> Speaker | None:
        with self._lock:
            spk = self._speakers.get(speaker_id)
            if spk is None:
                return None
            spk.name = new_name.strip()
            spk.updated_at = time.time()
            self._persist(spk)
            return spk

    # ---------- reads ----------

    def _find_by_name(self, name: str) -> Speaker | None:
        low = name.lower()
        for s in self._speakers.values():
            if s.name.lower() == low:
                return s
        return None

    def list(self) -> list[Speaker]:
        with self._lock:
            return list(self._speakers.values())

    def snapshot(self) -> "ProfileSnapshot":
        """Immutable, lock-free view for the real-time scoring path."""
        with self._lock:
            ids, names, centroids, per_sample = [], [], [], []
            for s in self._speakers.values():
                ids.append(s.id)
                names.append(s.name)
                centroids.append(s.centroid)
                per_sample.append(s.embeddings)
        return ProfileSnapshot(ids, names, centroids, per_sample)


@dataclass
class ProfileSnapshot:
    ids: list[str]
    names: list[str]
    centroids: list[np.ndarray]          # each (D,)
    per_sample: list[np.ndarray]         # each (N_i, D)

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def centroid_matrix(self) -> np.ndarray:
        if not self.centroids:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack(self.centroids)


# process-wide singleton
store = EnrollmentStore()
