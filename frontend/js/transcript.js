// Chat-style transcript renderer (WhatsApp-like), all-left group style.
// Public API unchanged: new TranscriptView(el,{autoscroll}), upsert(ev), clear(),
// search(q), segments(), plainText(); plus colorFor/fmtClock/exportTranscript.
//
// Evidence-based labeling: a turn opens as a PENDING (unlabeled) bubble that
// streams text immediately; when the backend confidently identifies the speaker
// it sends {type:'block_label', ...} (or a non-pending transcript_* event) and the
// bubble is relabeled IN PLACE (no reposition, no flicker). A turn that is never
// identified finalizes as "Unknown Speaker".

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
const keyOf = (ev, id) => ev.pending ? `pending#${id}`
  : (ev.unknown ? 'unknown' : (ev.speaker_id || 'unknown'));

export class TranscriptView {
  constructor(container, { autoscroll = true } = {}) {
    this.box = container;
    this.box.classList.add('chat');
    this.autoscroll = autoscroll;
    this.blocks = new Map();   // block_id -> { data, lineEl, group }
    this.groups = [];          // ordered bubble groups
    this.query = '';
    this._stick = true;
    this.box.addEventListener('scroll', () => {
      this._stick = this.box.scrollHeight - this.box.scrollTop - this.box.clientHeight < 80;
    });
  }

  clear() { this.blocks.clear(); this.groups = []; this.box.innerHTML = ''; }

  _newGroup(key, ev) {
    const g = document.createElement('div');
    g.className = 'msg left';
    g.innerHTML =
      `<div class="msg-head"><span class="avatar"></span><span class="who"></span></div>` +
      `<div class="bubble"><span class="btime"></span></div>`;
    this.box.appendChild(g);
    const group = {
      key, speaker: ev.speaker, unknown: !!ev.unknown, pending: !!ev.pending,
      startTime: ev.start ?? 0, lastEnd: ev.end ?? ev.start ?? 0,
      groupEl: g, bubbleEl: g.querySelector('.bubble'), timeEl: g.querySelector('.btime'),
      avatarEl: g.querySelector('.avatar'), whoEl: g.querySelector('.who'), blockIds: [],
    };
    this._style(group, ev);
    this.groups.push(group);
    return group;
  }

  // apply header + bubble styling for the group's current label state
  _style(group, ev) {
    const pend = !!ev.pending;
    group.pending = pend;
    group.groupEl.classList.toggle('pending', pend);
    group.groupEl.classList.toggle('unknown', !pend && !!ev.unknown);
    if (pend) {
      group.avatarEl.style.background = '#94a3b8';
      group.avatarEl.textContent = '';
      group.whoEl.textContent = 'Identifying…';
      group.whoEl.style.color = 'var(--faint)';
      group.bubbleEl.style.background = 'rgba(148,163,184,0.12)';
      group.bubbleEl.style.borderColor = 'rgba(148,163,184,0.35)';
    } else {
      const col = colorFor(ev.speaker_id, ev.unknown);
      group.avatarEl.style.background = col;
      group.avatarEl.textContent = initialsOf(ev.speaker, ev.unknown);
      group.whoEl.textContent = ev.speaker;
      group.whoEl.style.color = ev.unknown ? '' : col;
      group.bubbleEl.style.background = ev.unknown ? 'rgba(148,163,184,0.14)' : rgba(col, 0.10);
      group.bubbleEl.style.borderColor = ev.unknown ? 'rgba(148,163,184,0.45)' : rgba(col, 0.30);
      group.key = ev.unknown ? 'unknown' : (ev.speaker_id || 'unknown');
      group.speaker = ev.speaker; group.unknown = !!ev.unknown;
    }
  }

  upsert(ev) {
    if (ev.type === 'block_label') return this._relabel(ev);

    const id = ev.block_id;
    const hasText = (ev.text || '').trim().length > 0;
    let entry = this.blocks.get(id);

    if (!entry) {
      const key = keyOf(ev, id);
      const last = this.groups[this.groups.length - 1];
      // never merge across the 'unknown' key: two consecutive unknowns may be
      // different people, so keep them as separate bubbles.
      const mergeable = last && !ev.pending && key !== 'unknown' && last.key === key
        && !last.pending && ((ev.start ?? 0) - last.lastEnd) <= MERGE_GAP;
      const group = mergeable ? last : this._newGroup(key, ev);
      const lineEl = document.createElement('div');
      lineEl.className = 'line';
      group.bubbleEl.insertBefore(lineEl, group.timeEl);
      entry = { data: {}, lineEl, group };
      this.blocks.set(id, entry);
      group.blockIds.push(id);
    }

    // empty finalized block → drop it (and its group if now empty)
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
      entry.lineEl.innerHTML = TYPING;
      entry.lineEl.classList.remove('streaming');
    }
    const g = entry.group;
    g.lastEnd = Math.max(g.lastEnd, ev.end ?? ev.start ?? g.lastEnd);
    g.timeEl.textContent = fmtClock(g.startTime);

    // a pending bubble that just became labeled (via a non-pending event) → relabel
    if (g.pending && ev.pending === false) { this._style(g, ev); this._tryMerge(g); }
    this._scroll();
    this._applyFilter(g);
  }

  _relabel(ev) {
    const e = this.blocks.get(ev.block_id);
    if (!e) return;
    e.data = { ...e.data, speaker: ev.speaker, speaker_id: ev.speaker_id,
               unknown: ev.unknown, pending: false, confidence: ev.confidence };
    this._style(e.group, { ...ev, pending: false });
    this._tryMerge(e.group);
  }

  // merge a just-labeled group into the previous same-speaker group if close in time
  _tryMerge(g) {
    const i = this.groups.indexOf(g);
    if (i <= 0 || g.pending || g.key === 'unknown') return;
    const prev = this.groups[i - 1];
    if (prev.pending || prev.key !== g.key || (g.startTime - prev.lastEnd) > MERGE_GAP) return;
    for (const line of [...g.bubbleEl.querySelectorAll('.line')]) prev.bubbleEl.insertBefore(line, prev.timeEl);
    for (const id of g.blockIds) { const e = this.blocks.get(id); if (e) e.group = prev; prev.blockIds.push(id); }
    prev.lastEnd = Math.max(prev.lastEnd, g.lastEnd);
    g.groupEl.remove();
    this.groups.splice(i, 1);
  }

  _scroll() { if (this.autoscroll && this._stick) this.box.scrollTop = this.box.scrollHeight; }

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
      .sort((a, b) => (a.start ?? 0) - (b.start ?? 0))
      .map((d) => ({
        start: d.start ?? 0, end: d.end ?? d.start ?? 0,
        speaker: d.pending ? 'Unknown Speaker' : (d.speaker || 'Unknown Speaker'),
        speaker_id: d.speaker_id ?? null, unknown: !!d.unknown || !!d.pending,
        text: (d.text || '').trim(),
        confidence: d.confidence ?? null, asr_confidence: d.asr_confidence ?? null,
      }));
  }

  plainText() {
    return this.segments().map((s) => `${fmtClock(s.start)}\n${s.speaker}:\n${s.text}`).join('\n\n');
  }
}
