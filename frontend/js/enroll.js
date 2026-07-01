import { MicRecorder } from '/js/recorder.js';

const $ = (id) => document.getElementById(id);
const samples = []; // { name, blob, dur }
let rec = null, recording = false;
let recTimer = null, recStart = 0;
const fmtDur = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;

function toast(msg, kind = '') {
  const t = $('toast');
  t.textContent = msg;
  t.className = `toast show ${kind}`;
  setTimeout(() => (t.className = 'toast'), 2600);
}

function renderChips() {
  const box = $('chips');
  box.innerHTML = '';
  samples.forEach((s, i) => {
    const c = document.createElement('span');
    c.className = 'chip';
    const dur = s.dur ? ` · ${fmtDur(s.dur)}` : '';
    c.innerHTML = `${s.name}${dur} <span class="x" data-i="${i}">✕</span>`;
    box.appendChild(c);
  });
  $('enrollBtn').disabled = samples.length === 0;
  box.querySelectorAll('.x').forEach((x) =>
    x.addEventListener('click', () => { samples.splice(+x.dataset.i, 1); renderChips(); }));
}

const COLORS = ['#059669','#d97706','#e11d48','#0d9488','#db2777','#ca8a04','#be185d','#15803d'];

async function loadSpeakers() {
  const { speakers } = await (await fetch('/api/speakers')).json();
  const ul = $('speakers');
  const badge = $('spkCountBadge');
  if (badge) badge.textContent = `${speakers.length} speaker${speakers.length !== 1 ? 's' : ''}`;
  if (!speakers.length) {
    ul.innerHTML = '<div class="empty-state"><div class="e-icon">👤</div><p>No speakers enrolled yet.</p></div>';
    return;
  }
  ul.innerHTML = '';
  for (const [i, s] of speakers.entries()) {
    const col = COLORS[i % COLORS.length];
    const initials = s.name.split(' ').map(w => w[0]).join('').slice(0,2).toUpperCase();
    const li = document.createElement('li');
    li.innerHTML = `
      <div class="spk-info">
        <div class="spk-avatar" style="background:${col}">${initials}</div>
        <div>
          <div class="spk-name">${s.name}</div>
          <div class="spk-meta">${s.num_samples} sample${s.num_samples !== 1 ? 's' : ''}</div>
        </div>
      </div>`;
    const del = document.createElement('button');
    del.className = 'danger'; del.textContent = 'Delete';
    del.style.cssText = 'padding:6px 12px;font-size:12.5px;';
    del.onclick = async () => {
      await fetch(`/api/speakers/${s.id}`, { method: 'DELETE' });
      toast(`Deleted ${s.name}`); loadSpeakers();
    };
    li.appendChild(del); ul.appendChild(li);
  }
}

function startRecUI() {
  recStart = Date.now();
  $('recBtn').textContent = '■ Stop';
  $('micViz').classList.add('active');
  const status = $('recStatus');
  status.style.color = 'var(--danger)';
  status.style.fontWeight = '600';
  const tick = () => { status.textContent = `● ${fmtDur((Date.now() - recStart) / 1000)}`; };
  tick();
  recTimer = setInterval(tick, 200);
}
function stopRecUI() {
  clearInterval(recTimer); recTimer = null;
  $('recBtn').textContent = '● Record sample';
  $('micViz').classList.remove('active');
  const status = $('recStatus');
  status.textContent = 'idle'; status.style.color = ''; status.style.fontWeight = '';
}

$('recBtn').addEventListener('click', async () => {
  if (!recording) {
    try {
      rec = new MicRecorder({}); // capture mode (no onChunk)
      await rec.start();
      recording = true;
      startRecUI();
    } catch (e) { toast('Mic error: ' + e.message, 'err'); }
  } else {
    const dur = (Date.now() - recStart) / 1000;
    const blob = rec.takeWavBlob();
    rec.stop(); recording = false;
    stopRecUI();
    if (dur < 0.8) { toast('That clip was too short — hold for a few seconds', 'err'); return; }
    samples.push({ name: `recording-${samples.length + 1}.wav`, blob, dur });
    renderChips();
  }
});

$('fileInput').addEventListener('change', (e) => {
  for (const f of e.target.files) samples.push({ name: f.name, blob: f });
  e.target.value = '';
  renderChips();
});

$('clearBtn').addEventListener('click', () => { samples.length = 0; renderChips(); });

$('enrollBtn').addEventListener('click', async () => {
  const name = $('name').value.trim();
  if (!name) return toast('Enter a speaker name', 'err');
  if (!samples.length) return toast('Add at least one sample', 'err');
  const fd = new FormData();
  fd.append('name', name);
  samples.forEach((s) => fd.append('files', s.blob, s.name));
  $('enrollBtn').disabled = true; $('enrollBtn').textContent = 'Enrolling…';
  try {
    const r = await fetch('/api/speakers/enroll', { method: 'POST', body: fd });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || 'enroll failed');
    toast(data.message, 'ok');
    samples.length = 0; $('name').value = '';
    renderChips(); loadSpeakers();
  } catch (e) { toast(e.message, 'err'); }
  finally { $('enrollBtn').textContent = 'Enroll speaker'; $('enrollBtn').disabled = false; }
});

renderChips();
loadSpeakers();
