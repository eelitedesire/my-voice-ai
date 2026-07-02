// AI Session panels: Assistant chat, Session Analysis, Memory.
// Consumes the live transcript via window.__transcript() (set by live.js).
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

async function init() {
  document.querySelectorAll('.ai-tab').forEach((t) => (t.onclick = () => showTab(t.dataset.tab)));
  try {
    const t = await (await fetch('/api/prompt-templates')).json();
    aiEnabled = !!t.ai_enabled;
    const sel = $('aiPersona'); sel.innerHTML = '';
    (t.templates || []).forEach((tp) => {
      const o = document.createElement('option'); o.value = tp.id; o.textContent = tp.name; o.title = tp.description || '';
      sel.appendChild(o);
    });
    if (t.default) sel.value = t.default;
    $('aiStatus').textContent = aiEnabled ? 'ready' : 'disabled — set GROQ_API_KEY';
    if (!aiEnabled) { ['aiSend', 'aiInterject', 'aiAnalyze', 'aiInput', 'aiPersona'].forEach((id) => ($(id).disabled = true)); }
  } catch { $('aiStatus').textContent = 'unavailable'; }

  $('aiSend').onclick = () => { const v = $('aiInput').value.trim(); if (v) { $('aiInput').value = ''; sendMessage(v); } };
  $('aiInput').addEventListener('keydown', (e) => { if (e.key === 'Enter') $('aiSend').click(); });
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
  if (name === 'memory') loadMemory();
}

// ---------- assistant ----------
function appendChat(role, who, text) {
  const hint = $('aiChat').querySelector('.hint'); if (hint) hint.remove();
  const d = document.createElement('div');
  d.className = `ai-msg ${role}`;
  d.innerHTML = `<div class="ai-who">${role === 'therapist' ? '🧑‍⚕️ Therapist' : esc(who || 'You')}</div>` +
                `<div class="ai-text">${esc(text)}</div>`;
  $('aiChat').appendChild(d);
  $('aiChat').scrollTop = $('aiChat').scrollHeight;
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
    const data = await r.json();
    typing.remove();
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
    `<div class="an-block"><h4>Mood</h4><p>${esc(a.mood)}</p></div>` +
    `<div class="an-block"><h4>Key breakthroughs</h4>${list(a.keyBreakthroughs)}</div>` +
    `<div class="an-block"><h4>Homework</h4><p>${esc(a.homework)}</p></div>` +
    `<div class="an-block ${(a.concerns && a.concerns.length) ? 'concern' : ''}"><h4>Concerns</h4>${list(a.concerns)}</div>`;
}

// ---------- memory ----------
async function loadMemory() {
  const box = $('aiMemory'); box.innerHTML = '<p class="hint">Loading…</p>';
  try {
    const db = await (await fetch('/api/memory')).json();
    const speakers = Object.values(db.speakers || {});
    if (!speakers.length) {
      box.innerHTML = '<div class="empty-state"><div class="e-icon">🧠</div><p>No memories yet. Chat with the assistant or analyze a session to build them.</p></div>';
      return;
    }
    box.innerHTML = '';
    for (const s of speakers) {
      const el = document.createElement('div'); el.className = 'mem-spk';
      el.innerHTML = `<div class="mem-name">${esc(s.name)}</div>`;
      const facts = document.createElement('div'); facts.className = 'mem-facts';
      for (const f of s.facts) {
        const chip = document.createElement('span'); chip.className = 'mem-fact';
        chip.innerHTML = `<span class="cat">${esc(f.category)}</span> ${esc(f.content)} <span class="x" title="delete">✕</span>`;
        chip.querySelector('.x').onclick = async () => {
          await fetch(`/api/memory/${encodeURIComponent(s.name)}/${f.id}`, { method: 'DELETE' });
          loadMemory();
        };
        facts.appendChild(chip);
      }
      el.appendChild(facts); box.appendChild(el);
    }
  } catch { box.innerHTML = '<p class="hint">Failed to load memories.</p>'; }
}

init();
