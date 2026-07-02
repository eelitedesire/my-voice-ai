// Chat-style transcript renderer (WhatsApp-like) used by the live page.
// Public API is unchanged: new TranscriptView(el,{autoscroll}), upsert(ev),
// clear(), search(q), segments(), plainText(); plus colorFor/fmtClock/export.
//
// Events (block_open / transcript_partial / transcript_final) carry a block_id =
// one diarizer turn. Consecutive turns from the SAME speaker within MERGE_GAP are
// merged into one bubble group; a different speaker starts a new group.

const COLORS = ['#059669', '#d97706', '#e11d48', '#0d9488', '#db2777', '#ca8a04', '#be185d', '#15803d'];
const _assigned = {};
export function colorFor(id, unknown) {
  if (unknown || !id) return '#64748b';
  if (!(id in _assigned)) _assigned[id] = COLORS[Object.keys(_assigned).length % COLORS.length];
  return _assigned[id];
}
export const fmtClock = (s) => {
  s = Math.max(0, Math.floor(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), ss = s % 60;
  const p = (n) => String(n).padStart(2, '0');
  return h > 0 ? `${h}:${p(m)}:${p(ss)}` : `${p(m)}:${p(ss)}`;
};

const MERGE_GAP = 3.0;   // seconds: same speaker within this gap → same bubble
const TYPING = '<span class="typing"><i></i><i></i><i></i></span>';

function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
function highlight(text, q) {
  if (!q) return esc(text);
  if (text.toLowerCase().indexOf(q.toLowerCase()) < 0) return esc(text);
  const re = new RegExp(q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
  return esc(text).replace(re, (m) => `<mark>${m}</mark>`);
}
function initialsOf(name, unknown) {
  if (unknown) return '?';
  return (name || '?').split(/\s+/).map((w) => w[0]).join('').slice(0, 2).toUpperCase();
}
function rgba(hex, a) {
  const m = /^#?([\da-f]{2})([\da-f]{2})([\da-f]{2})$/i.exec(hex || '');
  if (!m) return `rgba(100,116,139,${a})`;
  return `rgba(${parseInt(m[1], 16)},${parseInt(m[2], 16)},${parseInt(m[3], 16)},${a})`;
}

export class TranscriptView {
  constructor(container, { autoscroll = true } = {}) {
    this.box = container;
    this.box.classList.add('chat');
    this.autoscroll = autoscroll;
    this.blocks = new Map();   // block_id -> { data, lineEl, group }
    this.groups = [];          // ordered bubble groups
    this._sides = [];          // speaker-key order → side assignment
    this.query = '';
    this._stick = true;
    this.box.addEventListener('scroll', () => {
      this._stick = this.box.scrollHeight - this.box.scrollTop - this.box.clientHeight < 80;
    });
  }

  clear() {
    this.blocks.clear(); this.groups = []; this._sides = [];
    this.box.innerHTML = '';
  }

  _sideFor(key) {
    let i = this._sides.indexOf(key);
    if (i < 0) { this._sides.push(key); i = this._sides.length - 1; }
    return i % 2 === 0 ? 'left' : 'right';
  }

  _newGroup(key, ev) {
    const col = colorFor(ev.speaker_id, ev.unknown);
    const side = this._sideFor(key);
    const g = document.createElement('div');
    g.className = `msg ${side}${ev.unknown ? ' unknown' : ''}`;
    g.innerHTML =
      `<div class="msg-head"><span class="avatar" style="background:${col}">${initialsOf(ev.speaker, ev.unknown)}</span>` +
      `<span class="who" style="color:${col}">${esc(ev.speaker)}</span></div>` +
      `<div class="bubble"><span class="btime"></span></div>`;
    this.box.appendChild(g);
    const bubble = g.querySelector('.bubble');
    // translucent tints so bubbles read correctly over both light and dark themes
    bubble.style.background = ev.unknown ? 'rgba(148,163,184,0.14)' : rgba(col, 0.10);
    bubble.style.borderColor = ev.unknown ? 'rgba(148,163,184,0.45)' : rgba(col, 0.30);
    const group = {
      key, speaker: ev.speaker, unknown: !!ev.unknown, startTime: ev.start,
      lastEnd: ev.end ?? ev.start, groupEl: g, bubbleEl: bubble,
      timeEl: g.querySelector('.btime'), blockIds: [],
    };
    this.groups.push(group);
    return group;
  }

  upsert(ev) {
    const id = ev.block_id;
    const key = ev.unknown ? 'unknown' : (ev.speaker_id || 'unknown');
    const hasText = (ev.text || '').trim().length > 0;
    let entry = this.blocks.get(id);

    if (!entry) {
      const last = this.groups[this.groups.length - 1];
      const gap = last ? (ev.start - last.lastEnd) : Infinity;
      const group = (last && last.key === key && gap <= MERGE_GAP) ? last : this._newGroup(key, ev);
      const lineEl = document.createElement('div');
      lineEl.className = 'line';
      group.bubbleEl.insertBefore(lineEl, group.timeEl);
      entry = { data: {}, lineEl, group };
      this.blocks.set(id, entry);
      group.blockIds.push(id);
    }

    // drop an empty finalized block (and its group if it becomes empty)
    if (ev.is_final && !hasText) {
      entry.lineEl.remove();
      const g = entry.group;
      g.blockIds = g.blockIds.filter((x) => x !== id);
      this.blocks.delete(id);
      if (!g.blockIds.length) { g.groupEl.remove(); this.groups = this.groups.filter((x) => x !== g); }
      return;
    }

    entry.data = { ...entry.data, ...ev };
    if (hasText) {
      entry.lineEl.innerHTML = highlight(ev.text, this.query);
      entry.lineEl.classList.toggle('streaming', ev.is_final === false);
    } else {
      entry.lineEl.innerHTML = TYPING;      // block opened, no words yet
      entry.lineEl.classList.remove('streaming');
    }
    const g = entry.group;
    g.lastEnd = Math.max(g.lastEnd, ev.end ?? ev.start);
    g.timeEl.textContent = fmtClock(g.startTime);
    this._scroll();
    this._applyFilter(g);
  }

  _scroll() {
    if (this.autoscroll && this._stick) this.box.scrollTop = this.box.scrollHeight;
  }

  search(q) {
    this.query = q.trim();
    for (const g of this.groups) {
      let hit = false;
      for (const id of g.blockIds) {
        const e = this.blocks.get(id); if (!e) continue;
        const txt = e.data.text || '';
        if (txt.trim()) e.lineEl.innerHTML = highlight(txt, this.query);
        if (this.query && txt.toLowerCase().includes(this.query.toLowerCase())) hit = true;
      }
      this._applyFilter(g, hit);
    }
  }
  _applyFilter(g, hit = null) {
    if (!this.query) { g.groupEl.classList.remove('dim'); return; }
    if (hit === null) hit = g.blockIds.some((id) => (this.blocks.get(id)?.data.text || '').toLowerCase().includes(this.query.toLowerCase()));
    g.groupEl.classList.toggle('dim', !hit);
  }

  segments() {
    return [...this.blocks.values()]
      .map((b) => b.data)
      .filter((d) => (d.text || '').trim())
      .sort((a, b) => a.start - b.start)
      .map((d) => ({
        start: d.start, end: d.end ?? d.start, speaker: d.speaker,
        speaker_id: d.speaker_id, unknown: !!d.unknown, text: (d.text || '').trim(),
        confidence: d.confidence ?? null, asr_confidence: d.asr_confidence ?? null,
      }));
  }

  plainText() {
    return this.segments().map((s) => `${fmtClock(s.start)}\n${s.speaker}:\n${s.text}`).join('\n\n');
  }
}

export async function exportTranscript(segments, format, filename = 'transcript') {
  if (!segments.length) throw new Error('Nothing to export yet');
  const r = await fetch('/api/export', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ format, segments, filename }),
  });
  if (!r.ok) throw new Error('Export failed');
  const blob = await r.blob();
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `${filename}.${format}`;
  a.click();
  URL.revokeObjectURL(a.href);
}
