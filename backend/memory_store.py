"""Session memory — per-speaker facts the LLM extracts from conversations.

Faithful Python port of voice-ai-master's JsonMemoryRepository: facts are kept
per speaker name and persisted to ``data/memory/memory.json`` so the assistant
can be personalised across sessions. Thread-safe; de-dupes by content.
"""
from __future__ import annotations

import json
import time
import uuid
import threading
from pathlib import Path

from .config import MEMORY_DIR

CATEGORIES = ("personal", "relationship", "emotional", "goal",
              "preference", "history", "other")
_DB_PATH = MEMORY_DIR / "memory.json"
_lock = threading.RLock()


def _load() -> dict:
    if _DB_PATH.exists():
        try:
            return json.loads(_DB_PATH.read_text())
        except Exception:
            pass
    return {"speakers": {}}


def _save(db: dict) -> None:
    _DB_PATH.write_text(json.dumps(db, indent=2))


def get_all() -> dict:
    with _lock:
        return _load()


def get_for_speaker(name: str) -> dict | None:
    with _lock:
        return _load()["speakers"].get(name)


def add_facts(name: str, facts: list[dict]) -> list[dict]:
    """Add {content, category} facts for a speaker (skips duplicates). Returns
    the facts actually stored (with id + timestamp)."""
    with _lock:
        db = _load()
        spk = db["speakers"].setdefault(name, {"name": name, "facts": [], "updatedAt": time.time()})
        have = {f["content"].strip().lower() for f in spk["facts"]}
        added = []
        for f in facts:
            content = (f.get("content") or "").strip()
            if not content or content.lower() in have:
                continue
            cat = f.get("category") if f.get("category") in CATEGORIES else "other"
            fact = {"id": uuid.uuid4().hex[:12], "content": content,
                    "category": cat, "extractedAt": time.time()}
            spk["facts"].append(fact)
            have.add(content.lower())
            added.append(fact)
        if added:
            spk["updatedAt"] = time.time()
            _save(db)
        return added


def delete_fact(name: str, fact_id: str) -> bool:
    with _lock:
        db = _load()
        spk = db["speakers"].get(name)
        if not spk:
            return False
        n = len(spk["facts"])
        spk["facts"] = [f for f in spk["facts"] if f["id"] != fact_id]
        if len(spk["facts"]) == n:
            return False
        if not spk["facts"]:
            db["speakers"].pop(name, None)
        _save(db)
        return True


def clear_speaker(name: str) -> bool:
    with _lock:
        db = _load()
        if name not in db["speakers"]:
            return False
        db["speakers"].pop(name)
        _save(db)
        return True


def format_for_context(names: list[str]) -> str:
    """Compact 'what we know about these speakers' block for the LLM prompt."""
    with _lock:
        db = _load()
    parts = []
    for name in names:
        spk = db["speakers"].get(name)
        if spk and spk["facts"]:
            lines = "\n".join(f"- {f['content']}" for f in spk["facts"])
            parts.append(f"What we know about {name}:\n{lines}")
    return ("Relevant background from past sessions:\n" + "\n\n".join(parts)) if parts else ""
