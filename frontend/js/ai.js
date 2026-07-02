// AI Session panels — faithful to voice-ai-master: Assistant (therapist chat with
// speaker dropdown), Analysis (structured + TTS), Memory (category chips, clear-all,
// auto-refresh). Reads the live transcript via window.__transcript() (from live.js).
const $ = (id) => document.getElementById(id);
const getTranscript = () => (window.__transcript ? window.__transcript() : []);
const esc = (s) => { const d = document.createElement('div'); d.textContent = s ?? ''; return d.innerHTML; };
function toast(msg, kind = '') {
  const t = $('toast'); if (!t) return;
  t.textContent = msg; t.className = `toast show ${kind}`;
  setTimeout(() => (t.className = 'toast'), 2600);
}

let chatHistory = [];
let aiEnabled = false;
let enrolled = [];
let memTimer = null;
let speaking = false;

const CAT_LABELS = { personal: 'Personal', relationship: 'Relationship', emotional: 'Emotional',
  goal: 'Goal', preference: 'Preference', history: 'History', other: 'Other' };
const CAT_COLORS = { personal: ['#dbeafe', '#1e40af'], relationship: ['#fce7f3', '#9d174d'],
  emotional: ['#fef9c3', '#854d0e'], goal: ['#dcfce7', '#166534'], preference: ['#f3e8ff', '#6b21a8'],
  history: ['#e5e7eb', '#374151'], other: ['#f1f5f9', '#334155'] };

async function init() {
  let data = null;
  try { data = await (await fetch('/api/prompt-templates')).json(); } catch {}
  aiEnabled = !!(data && data.ai_enabled);

  // If the AI layer isn't configured, hide the whole card — no technical detail.
  if (!aiEnabled) { const c = $('aiCard'); if (c) c.remove(); return; }

  $('aiCard').style.display = '';
  $('aiStatus').textContent = 'ready';
  document.querySelectorAll('.ai-tab').forEach((t) => (t.onclick = () => showTab(t.dataset.tab)));
  const sel = $('aiPersona'); sel.innerHTML = '';
  (data.templates || []).forEach((tp) => {
    const o = document.createElement('option'); o.value = tp.id; o.textContent = tp.name; o.title = tp.description || '';
    sel.appendChild(o);
  });
  if (data.default) sel.value = data.default;

  await loadEnrolled();
  updateSpeakers();
  setInterval(updateSpeakers, 3000);   // pick up speakers as they're detected

  $('aiSend').onclick = send;
  $('aiInput').addEventListener('keydown', (e) => { if (e.key === 'Enter') send(); });
  $('aiInterject').onclick = () => {
    const seg = getTranscript(); const last = seg[seg.length - 1];
    if (!last) return toast('No transcript yet', 'err');
    sendMessage(`[${last.speaker}]: ${last.text}`);
  };
  $('aiAnalyze').onclick = analyze;
  $('aiMemRefresh').onclick = loadMemory;
}

function showTab(name) {
  document.querySelectorAll('.ai-tab').forEach((t) => t.classList.toggle('active', t.dataset.tab === name));
  document.querySelectorAll('.ai-pane').forEach((p) => (p.hidden = p.id !== `pane-${name}`));
  clearInterval(memTimer); memTimer = null;
  if (name === 'memory') { loadMemory(); memTimer = setInterval(loadMemory, 10000); }
}

// ---------- speaker dropdown ----------
async function loadEnrolled() {
  try { const { speakers } = await (await fetch('/api/speakers')).json(); enrolled = speakers.map((s) => s.name); } catch {}
}
function updateSpeakers() {
  const fromTx = [...new Set(getTranscript().map((s) => s.speaker).filter((x) => x && x !== 'Unknown Speaker'))];
  const all = [...new Set([...enrolled, ...fromTx])];
  const sel = $('aiSpeaker'); if (!sel) return;
  if (sel.dataset.list === JSON.stringify(all)) return;
  const cur = sel.value; sel.dataset.list = JSON.stringify(all);
  sel.innerHTML = all.length ? '' : '<option value="">No speakers</option>';
  all.forEach((n) => { const o = document.createElement('option'); o.value = n; o.textContent = n; sel.appendChild(o); });
  if (all.includes(cur)) sel.value = cur; else if (all.length) sel.value = all[0];
}

// ---------- assistant ----------
function appendChat(role, who, text) {
  const hint = $('aiChat').querySelector('.hint'); if (hint) hint.remove();
  const d = document.createElement('div');
  d.className = `ai-msg ${role}`;
  d.innerHTML = `<div class="ai-who">${role === 'therapist' ? '🧑‍⚕️ Therapist' : esc(who || 'You')}</div>` +
                `<div class="ai-text">${esc(text)}</div>`;
  $('aiChat').appendChild(d); $('aiChat').scrollTop = $('aiChat').scrollHeight;
}
function send() {
  const v = $('aiInput').value.trim(); if (!v) return;
  const spk = $('aiSpeaker').value;
  $('aiInput').value = '';
  sendMessage(spk ? `[${spk}]: ${v}` : v);
}
async function sendMessage(message) {
  const m = message.match(/^\[([^\]]+)\]:\s*(.+)$/);
  appendChat('user', m ? m[1] : 'You', m ? m[2] : message);
  chatHistory.push({ role: 'user', speaker: m ? m[1] : undefined, text: m ? m[2] : message });
  $('aiSend').disabled = true; $('aiInterject').disabled = true;
  const typing = document.createElement('div'); typing.className = 'ai-msg therapist';
  typing.innerHTML = '<div class="ai-who">🧑‍⚕️ Therapist</div><div class="ai-text"><span class="typing"><i></i><i></i><i></i></span></div>';
  $('aiChat').appendChild(typing); $('aiChat').scrollTop = $('aiChat').scrollHeight;
  try {
    const r = await fetch('/api/assistant', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message, transcript: getTranscript(), chat_history: chatHistory }),
    });
    const data = await r.json(); typing.remove();
    if (!r.ok) throw new Error(data.detail || 'assistant error');
    appendChat('therapist', '', data.reply);
    chatHistory.push({ role: 'therapist', text: data.reply });
    if (data.safety_override) toast('⚠️ Safety override triggered', 'err');
  } catch (e) { typing.remove(); appendChat('therapist', '', '⚠️ ' + e.message); }
  finally { $('aiSend').disabled = false; $('aiInterject').disabled = false; }
}

// ---------- analysis ----------
async function analyze() {
  const seg = getTranscript();
  if (!seg.length) return toast('No transcript to analyze yet', 'err');
  const btn = $('aiAnalyze'); btn.disabled = true; btn.textContent = 'Analyzing…';
  $('aiAnalysis').innerHTML = '';
  try {
    const r = await fetch('/api/analyze', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ transcript: seg, template_id: $('aiPersona').value }),
    });
    const a = await r.json();
    if (!r.ok) throw new Error(a.detail || 'analyze error');
    renderAnalysis(a);
  } catch (e) { $('aiAnalysis').innerHTML = `<p class="hint">⚠️ ${esc(e.message)}</p>`; }
  finally { btn.disabled = false; btn.textContent = 'Analyze session'; }
}
function renderAnalysis(a) {
  const list = (arr) => (arr && arr.length) ? '<ul>' + arr.map((x) => `<li>${esc(x)}</li>`).join('') + '</ul>' : '<p class="hint">None noted.</p>';
  $('aiAnalysis').innerHTML =
    `<div class="an-block"><h4>Summary</h4><p>${esc(a.summary)}</p></div>` +
    `<div class="an-block"><h4>Mood</h4><p><span class="mood-pill">${esc(a.mood)}</span></p></div>` +
    `<div class="an-block"><h4>Key breakthroughs</h4>${list(a.keyBreakthroughs)}</div>` +
    `<div class="an-block"><h4>Homework</h4><p class="homework">${esc(a.homework)}</p></div>` +
    `<div class="an-block ${(a.concerns && a.concerns.length) ? 'concern' : ''}"><h4>Areas of concern</h4>${list(a.concerns)}</div>`;
  const btn = document.createElement('button');
  btn.className = 'secondary'; btn.style.marginTop = '12px'; btn.id = 'aiListen';
  btn.textContent = '🔊 Listen to Analysis';
  btn.onclick = () => speakAnalysis(a, btn);
  $('aiAnalysis').appendChild(btn);
}
function speakAnalysis(a, btn) {
  if (!('speechSynthesis' in window)) return toast('Text-to-speech not supported by this browser', 'err');
  if (speaking) { window.speechSynthesis.cancel(); speaking = false; btn.textContent = '🔊 Listen to Analysis'; return; }
  const text = `Session summary. ${a.summary}. Mood assessment: ${a.mood}. ` +
    `Key breakthroughs: ${(a.keyBreakthroughs || []).join('. ')}. Homework: ${a.homework}. ` +
    ((a.concerns && a.concerns.length) ? `Areas of concern: ${a.concerns.join('. ')}.` : '');
  const u = new SpeechSynthesisUtterance(text); u.rate = 1.0;
  u.onend = () => { speaking = false; btn.textContent = '🔊 Listen to Analysis'; };
  speaking = true; btn.textContent = '⏹ Stop';
  window.speechSynthesis.cancel(); window.speechSynthesis.speak(u);
}

// ---------- memory ----------
async function loadMemory() {
  const box = $('aiMemory');
  try {
    const db = await (await fetch('/api/memory')).json();
    const speakers = Object.values(db.speakers || {}).filter((s) => s.facts && s.facts.length);
    speakers.sort((a, b) => (b.updatedAt || 0) - (a.updatedAt || 0));
    if (!speakers.length) {
      box.innerHTML = '<div class="empty-state"><div class="e-icon">🧠</div><p>No memories yet. They\'re extracted automatically after chat messages and session analysis.</p></div>';
      return;
    }
    box.innerHTML = '';
    for (const s of speakers) {
      const el = document.createElement('div'); el.className = 'mem-spk';
      const head = document.createElement('div'); head.className = 'mem-head';
      head.innerHTML = `<span class="mem-name">${esc(s.name)}</span>` +
        `<span class="mem-meta">${s.facts.length} fact${s.facts.length !== 1 ? 's' : ''}` +
        ` · <a class="mem-clear">Clear all</a></span>`;
      head.querySelector('.mem-clear').onclick = async () => {
        if (!confirm(`Clear all memories for ${s.name}?`)) return;
        await fetch(`/api/memory/${encodeURIComponent(s.name)}`, { method: 'DELETE' }); loadMemory();
      };
      el.appendChild(head);
      const facts = document.createElement('div'); facts.className = 'mem-facts';
      for (const f of s.facts) {
        const [bg, fg] = CAT_COLORS[f.category] || CAT_COLORS.other;
        const chip = document.createElement('span'); chip.className = 'mem-fact';
        chip.innerHTML = `<span class="cat" style="background:${bg};color:${fg}">${esc(CAT_LABELS[f.category] || f.category)}</span> ${esc(f.content)} <span class="x" title="delete">✕</span>`;
        chip.querySelector('.x').onclick = async () => {
          await fetch(`/api/memory/${encodeURIComponent(s.name)}/${f.id}`, { method: 'DELETE' }); loadMemory();
        };
        facts.appendChild(chip);
      }
      el.appendChild(facts); box.appendChild(el);
    }
  } catch { box.innerHTML = '<p class="hint">Failed to load memories.</p>'; }
}

init();
