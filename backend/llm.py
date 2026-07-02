"""Groq LLM client — the single place the AI layer talks to a hosted model.

CPU-only server, so the model is a hosted API (Groq / Llama). Two primitives,
mirroring the Vercel AI SDK usage in voice-ai-master:

  * ``generate_text``  -> free-form text (the live assistant)
  * ``generate_json``  -> a JSON object validated against a schema hint
                          (session analysis + memory extraction)

Every call is guarded: if no key is configured the caller gets a clear
``LLMUnavailable`` error, which the API turns into a 503 rather than a crash.
"""
from __future__ import annotations

import json
import threading

from .config import settings

_lock = threading.Lock()
_client = None


class LLMUnavailable(RuntimeError):
    """Raised when the AI features are used without a configured Groq key."""


def available() -> bool:
    return settings.ai_enabled()


def _get():
    global _client
    if not settings.groq_api_key.strip():
        raise LLMUnavailable(
            "AI features are disabled: set GROQ_API_KEY (and ENABLE_ASSISTANT!=0)."
        )
    if _client is not None:
        return _client
    with _lock:
        if _client is None:
            from groq import Groq
            _client = Groq(api_key=settings.groq_api_key.strip())
    return _client


def generate_text(system: str, prompt: str, temperature: float = 0.7,
                  max_tokens: int = 700) -> str:
    resp = _get().chat.completions.create(
        model=settings.groq_model,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()


def generate_json(system: str, prompt: str, schema_hint: str,
                  temperature: float = 0.3, max_tokens: int = 1200) -> dict:
    """Return a parsed JSON object. ``schema_hint`` is a compact description of
    the expected shape appended to the system prompt (Groq JSON mode guarantees
    valid JSON; we still validate/normalize downstream)."""
    sys = (f"{system}\n\nRespond with ONLY a single valid JSON object, no prose, "
           f"matching this shape:\n{schema_hint}")
    resp = _get().chat.completions.create(
        model=settings.groq_model,
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # last-ditch: pull the outermost {...}
        a, b = raw.find("{"), raw.rfind("}")
        return json.loads(raw[a:b + 1]) if a >= 0 and b > a else {}
