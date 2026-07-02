"""AI layer — faithful port of voice-ai-master's LLM features onto the Sanuvia
transcript. All of it consumes the transcript this app already produces:

  * assistant()  — live in-session therapist reply (generate_text)
  * analyze()    — structured clinical session analysis (generate_json)
  * memory       — LLM-extracted per-speaker facts, stored + injected as context
  * safety_scan  — deterministic crisis detection that overrides the AI reply

LLM calls go through backend.llm (Groq). Memory extraction is fire-and-forget.
"""
from __future__ import annotations

import re
import threading

from . import llm
from . import memory_store as memory

# ─────────────────────────────── personas ───────────────────────────────

THERAPIST_SYSTEM_PROMPT = """You are an experienced couples therapist participating in a live therapy session. You are observing a conversation between partners and can interject with therapeutic guidance.

Your role:
- Provide empathetic, professional therapeutic responses
- Ask clarifying or reflective questions to individuals or the couple
- Offer suggestions, reframes, or observations when helpful
- Address messages to a specific person by name, or to the couple together
- Keep responses concise and conversational (2-4 sentences typically)
- Use trauma-informed, culturally sensitive language

You can see the full session transcript and the live chat history. Respond naturally as a therapist would in session.

IMPORTANT formatting rules:
- When addressing a specific person, start with their name followed by a comma (e.g. "Sarah, I notice that...")
- When addressing the couple, you can start directly (e.g. "I'd like both of you to consider...")
- Be warm but professional"""

# Supervisor "lenses" (ported from prompt-templates.ts).
SUPERVISOR_TEMPLATES: list[dict] = [
    {"id": "clinical-supervisor", "name": "Clinical Supervisor",
     "description": "General clinical supervision: breakthroughs, mood, homework",
     "prompt": """You are a clinical supervisor analyzing a therapeutic session between a Therapist and a Client.

Your role is to:
1. Identify key emotional breakthroughs and patterns
2. Assess the client's emotional state and mood
3. Suggest actionable homework assignments that build on session insights
4. Flag any concerns that require immediate attention (e.g., safety issues, crisis indicators)

Guidelines:
- Be compassionate yet objective
- Focus on evidence from the transcript
- Provide specific, actionable recommendations
- Use trauma-informed language
- Consider cultural sensitivity

Analyze the following transcript and provide a structured clinical assessment."""},
    {"id": "cbt-focused", "name": "CBT-Focused Supervisor",
     "description": "Identifies cognitive distortions and behavioral patterns",
     "prompt": """You are a clinical supervisor specializing in Cognitive Behavioral Therapy (CBT), analyzing a therapeutic session between a Therapist and a Client.

Your role is to:
1. Identify cognitive distortions in the client's statements (catastrophizing, black-and-white thinking, mind reading, overgeneralization)
2. Assess the client's emotional state and underlying core beliefs
3. Suggest CBT-based homework (thought records, behavioral experiments, activity scheduling)
4. Flag any concerns that require immediate attention

Focus on the connection between thoughts, feelings, and behaviors. Analyze the transcript through a CBT lens."""},
    {"id": "psychodynamic", "name": "Psychodynamic Supervisor",
     "description": "Explores unconscious patterns, defenses, and the alliance",
     "prompt": """You are a clinical supervisor with a psychodynamic orientation, analyzing a therapeutic session between a Therapist and a Client.

Your role is to:
1. Identify recurring relational patterns and unconscious themes
2. Assess the client's defense mechanisms (projection, denial, intellectualization, displacement)
3. Evaluate the therapeutic alliance and any transference/countertransference
4. Suggest reflective exercises or journaling prompts that deepen self-awareness

Pay attention to what is not said. Analyze the transcript through a psychodynamic lens."""},
    {"id": "trauma-informed", "name": "Trauma-Informed Supervisor",
     "description": "Trauma response, safety, and stabilization",
     "prompt": """You are a clinical supervisor specializing in trauma-informed care, analyzing a therapeutic session between a Therapist and a Client.

Your role is to:
1. Identify trauma responses and triggers (hyperarousal, dissociation, avoidance, emotional flooding)
2. Assess the client's window of tolerance and emotional regulation
3. Suggest grounding and stabilization homework (breathing, safe-place visualization, body scan)
4. Flag any safety concerns, re-traumatization risks, or crisis indicators with HIGH priority

Prioritize safety and stabilization. Analyze the transcript through a trauma-informed lens."""},
    {"id": "brief-solution-focused", "name": "Solution-Focused Supervisor",
     "description": "Strengths, goals, exceptions, scaling progress",
     "prompt": """You are a clinical supervisor using Solution-Focused Brief Therapy (SFBT), analyzing a therapeutic session between a Therapist and a Client.

Your role is to:
1. Identify client strengths, resources, and exceptions to the problem
2. Assess progress toward the client's goals on a 1-10 scale
3. Suggest solution-focused homework (miracle question reflection, exception tracking, scaling)
4. Reframe concerns in terms of what the client needs to move forward

Focus on what is working. Keep the assessment future-oriented. Analyze the transcript through a solution-focused lens."""},
]
SUPERVISOR_BY_ID = {t["id"]: t for t in SUPERVISOR_TEMPLATES}
DEFAULT_SUPERVISOR = "clinical-supervisor"

# ─────────────────────────────── safety ─────────────────────────────────

_SAFETY_PATTERNS = {
    "suicidal-ideation": [r"\bkill myself\b", r"\bwant to die\b", r"\bend my life\b",
                          r"\bsuicid", r"\bbetter off dead\b", r"\bno reason to live\b",
                          r"\btake my own life\b", r"\bdon'?t want to (?:be here|live)\b"],
    "self-harm": [r"\bhurt myself\b", r"\bcut myself\b", r"\bself[- ]harm", r"\bharming myself\b"],
    "domestic-violence": [r"\b(?:he|she|they|my partner) hits? me\b", r"\bbeats? me\b",
                          r"\bafraid of my partner\b", r"\bthreatens? (?:to hurt|me)\b"],
    "child-abuse": [r"\bhit my (?:child|kid|son|daughter)\b", r"\bhurt(?:ing)? the (?:kids|children)\b",
                    r"\bchild abuse\b"],
}
_CRISIS_TEXT = {
    "suicidal-ideation": "988 Suicide & Crisis Lifeline (call or text 988), Crisis Text Line (text HOME to 741741).",
    "self-harm": "988 Suicide & Crisis Lifeline (call or text 988), Crisis Text Line (text HOME to 741741).",
    "domestic-violence": "National Domestic Violence Hotline 1-800-799-7233 (text START to 88788).",
    "child-abuse": "Childhelp National Child Abuse Hotline 1-800-422-4453.",
}


def safety_scan(text: str) -> dict | None:
    """Deterministic crisis check. Returns an override dict if triggered, else None."""
    low = (text or "").lower()
    flags = [k for k, pats in _SAFETY_PATTERNS.items() if any(re.search(p, low) for p in pats)]
    if not flags:
        return None
    resources = "  •  ".join(dict.fromkeys(_CRISIS_TEXT[f] for f in flags))
    reply = ("It sounds like there may be something serious happening, and your safety "
             "matters most right now. Please consider reaching out to trained crisis "
             f"support immediately:  {resources}  If anyone is in immediate danger, call "
             "your local emergency number.")
    return {"reply": reply, "safety_override": True, "flags": flags}


# ─────────────────────────────── helpers ────────────────────────────────

def _format_transcript(entries: list[dict]) -> str:
    return "\n".join(f"[{e.get('speaker') or 'Unknown'}]: {e.get('text','')}" for e in entries)


def _speaker_names(transcript: list[dict], chat: list[dict]) -> list[str]:
    names: list[str] = []
    for e in transcript:
        s = e.get("speaker")
        if s and s not in names:
            names.append(s)
    for m in chat:
        s = m.get("speaker")
        if s and s not in names:
            names.append(s)
    return names


# ─────────────────────────────── assistant ──────────────────────────────

def assistant(message: str, transcript: list[dict] | None = None,
              chat_history: list[dict] | None = None,
              system_prompt: str | None = None) -> dict:
    transcript = transcript or []
    chat_history = chat_history or []

    # 1) deterministic safety override (runs before the LLM)
    over = safety_scan(message)
    if over:
        return over

    # 2) assemble context (transcript + chat + stored memories)
    ctx = ""
    if transcript:
        ctx += "Session transcript so far:\n" + _format_transcript(transcript) + "\n\n"
    if chat_history:
        lines = []
        for m in chat_history:
            if m.get("role") == "therapist":
                lines.append(f"[Therapist]: {m.get('text','')}")
            else:
                lines.append(f"[{m.get('speaker') or 'Unknown'}]: {m.get('text','')}")
        ctx += "Live chat history:\n" + "\n".join(lines) + "\n\n"
    mem = memory.format_for_context(_speaker_names(transcript, chat_history))
    if mem:
        ctx += mem + "\n\n"

    system = (system_prompt or "").strip() or THERAPIST_SYSTEM_PROMPT
    reply = llm.generate_text(
        system,
        f"{ctx}The following message was just sent in the live chat. Respond as the therapist.\n\nMessage: {message}",
    )

    # 3) fire-and-forget memory extraction from this message
    m = re.match(r"^\[([^\]]+)\]:\s*(.+)$", message, re.S)
    if m:
        extract_from_message(m.group(2), m.group(1))

    return {"reply": reply, "safety_override": False}


# ─────────────────────────────── supervisor ─────────────────────────────

_ANALYSIS_SHAPE = (
    '{"summary": string, "mood": string, "keyBreakthroughs": string[], '
    '"homework": string, "concerns": string[]}'
)


def analyze(transcript: list[dict], system_prompt: str | None = None,
            template_id: str | None = None) -> dict:
    system = (system_prompt or "").strip()
    if not system:
        tpl = SUPERVISOR_BY_ID.get(template_id or DEFAULT_SUPERVISOR, SUPERVISOR_BY_ID[DEFAULT_SUPERVISOR])
        system = tpl["prompt"]

    obj = llm.generate_json(
        system,
        f"Analyze this therapeutic session transcript:\n\n{_format_transcript(transcript)}",
        _ANALYSIS_SHAPE,
    )
    result = {
        "summary": str(obj.get("summary", "")).strip(),
        "mood": str(obj.get("mood", "")).strip(),
        "keyBreakthroughs": [str(x) for x in (obj.get("keyBreakthroughs") or [])],
        "homework": str(obj.get("homework", "")).strip(),
        "concerns": [str(x) for x in (obj.get("concerns") or [])],
    }

    # fire-and-forget memory extraction from the whole session
    names = list(dict.fromkeys(e.get("speaker") for e in transcript if e.get("speaker")))
    extract_from_transcript(transcript, names)
    return result


# ─────────────────────────────── memory extraction ──────────────────────

_TX_MEM_SHAPE = ('{"speakers": [{"name": string, "facts": '
                 '[{"content": string, "category": '
                 '"personal"|"relationship"|"emotional"|"goal"|"preference"|"history"|"other"}]}]}')
_MSG_MEM_SHAPE = ('{"facts": [{"content": string, "category": '
                  '"personal"|"relationship"|"emotional"|"goal"|"preference"|"history"|"other"}]}')


def extract_from_transcript(transcript: list[dict], names: list[str]) -> None:
    if not llm.available() or not transcript:
        return
    threading.Thread(target=_do_extract_transcript, args=(transcript, names), daemon=True).start()


def extract_from_message(text: str, speaker: str) -> None:
    if not llm.available() or not text.strip():
        return
    threading.Thread(target=_do_extract_message, args=(text, speaker), daemon=True).start()


def _existing_block(names: list[str]) -> str:
    parts = []
    for n in names:
        mem = memory.get_for_speaker(n)
        if mem and mem["facts"]:
            parts.append(f"Already known about {n}:\n" + "\n".join(f"- {f['content']}" for f in mem["facts"]))
    return ("\n\nThese are already stored — do NOT re-extract them:\n" + "\n\n".join(parts)) if parts else ""


def _do_extract_transcript(transcript: list[dict], names: list[str]) -> None:
    try:
        obj = llm.generate_json(
            "You extract important facts about people from therapy transcripts. Extract personal "
            "details, relationship dynamics, emotional patterns, goals, preferences, and history. "
            "Each fact is one concise sentence. Only extract genuinely new information."
            + _existing_block(names),
            f"Extract facts about each person from this transcript:\n\n{_format_transcript(transcript)}",
            _TX_MEM_SHAPE,
        )
        for spk in obj.get("speakers", []):
            if spk.get("name") and spk.get("facts"):
                memory.add_facts(spk["name"], spk["facts"])
    except Exception as e:  # pragma: no cover
        print(f"[ai] transcript memory extraction failed: {e}")


def _do_extract_message(text: str, speaker: str) -> None:
    try:
        obj = llm.generate_json(
            "Extract important new facts about the person from this therapy chat message. Only "
            "extract genuinely valuable information; return an empty array if nothing noteworthy. "
            "Each fact is one concise sentence." + _existing_block([speaker]),
            f'Speaker "{speaker}" said: {text}',
            _MSG_MEM_SHAPE,
        )
        if obj.get("facts"):
            memory.add_facts(speaker, obj["facts"])
    except Exception as e:  # pragma: no cover
        print(f"[ai] message memory extraction failed: {e}")
