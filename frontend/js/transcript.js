// Shared, order-independent transcript renderer used by the live and upload pages.
// Blocks are keyed by block_id and upserted, so events may arrive in any order
// (e.g. a streaming partial before its block_open).

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
function esc(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }
function highlight(text, q) {
  if (!q) return esc(text);
  const i = text.toLowerCase().indexOf(q.toLowerCase());
  if (i < 0) return esc(text);
  const re = new RegExp(q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'gi');
  return esc(text).replace(new RegExp(re.source, 'gi'), (m) => `<mark>${m}</mark>`);
}

export class TranscriptView {
  constructor(container, { autoscroll = true } = {}) {
    this.box = container;
    this.autoscroll = autoscroll;
    this.blocks = new Map();     // block_id -> {el, bodyEl, tsEl, badgeEl, data}
    this.query = '';
    this.box.addEventListener('scroll', () => {
      const nearBottom = this.box.scrollHeight - this.box.scrollTop - this.box.clientHeight < 60;
      this._stick = nearBottom;
    });
    this._stick = true;
  }

  clear() { this.blocks.clear(); this.box.innerHTML = ''; }

  upsert(ev) {
    const id = ev.block_id;
    let b = this.blocks.get(id);
    if (!b) {
      const el = document.createElement('div');
      el.className = 'block';
      el.innerHTML = `<div class="gutter"><span class="badge-spk"></span><span class="ts"></span></div><div class="body"></div>`;
      // insert in ascending block_id order
      const after = [...this.blocks.values()].filter((x) => x.data.block_id < id).pop();
      if (after) after.el.after(el); else this.box.appendChild(el);
      b = { el, bodyEl: el.querySelector('.body'), tsEl: el.querySelector('.ts'),
            badgeEl: el.querySelector('.badge-spk'), data: {} };
      this.blocks.set(id, b);
    }
    b.data = { ...b.data, ...ev };
    const col = colorFor(ev.speaker_id, ev.unknown);
    b.badgeEl.style.background = col;
    b.badgeEl.textContent = ev.speaker;
    b.el.classList.toggle('unknown', !!ev.unknown);
    b.el.classList.toggle('streaming', ev.is_final === false);
    const range = ev.end != null ? `${fmtClock(ev.start)} – ${fmtClock(ev.end)}` : fmtClock(ev.start);
    b.tsEl.textContent = range;
    b.bodyEl.innerHTML = highlight(ev.text || '', this.query);
    if (this.autoscroll && this._stick) this.box.scrollTop = this.box.scrollHeight;
    this._applyFilter(b);
  }

  search(q) {
    this.query = q.trim();
    for (const b of this.blocks.values()) {
      b.bodyEl.innerHTML = highlight(b.data.text || '', this.query);
      this._applyFilter(b);
    }
  }
  _applyFilter(b) {
    if (!this.query) { b.el.classList.remove('dim'); return; }
    const hit = (b.data.text || '').toLowerCase().includes(this.query.toLowerCase());
    b.el.classList.toggle('dim', !hit);
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
