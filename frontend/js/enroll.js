import { MicRecorder } from '/js/recorder.js';

const $ = (id) => document.getElementById(id);
const samples = []; // { name, blob }
let rec = null, recording = false;

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
    c.innerHTML = `${s.name} <span class="x" data-i="${i}">✕</span>`;
    box.appendChild(c);
  });
  $('enrollBtn').disabled = samples.length === 0;
  box.querySelectorAll('.x').forEach((x) =>
    x.addEventListener('click', () => { samples.splice(+x.dataset.i, 1); renderChips(); }));
}

async function loadSpeakers() {
  const { speakers } = await (await fetch('/api/speakers')).json();
  const ul = $('speakers');
  ul.innerHTML = speakers.length ? '' : '<li class="meta">None yet.</li>';
  for (const s of speakers) {
    const li = document.createElement('li');
    li.innerHTML = `<span><strong>${s.name}</strong> <span class="meta">· ${s.num_samples} sample(s)</span></span>`;
    const del = document.createElement('button');
    del.className = 'danger'; del.textContent = 'Delete';
    del.onclick = async () => {
      await fetch(`/api/speakers/${s.id}`, { method: 'DELETE' });
      toast(`Deleted ${s.name}`); loadSpeakers();
    };
    li.appendChild(del); ul.appendChild(li);
  }
}

$('recBtn').addEventListener('click', async () => {
  if (!recording) {
    try {
      rec = new MicRecorder({}); // capture mode (no onChunk)
      await rec.start();
      recording = true;
      $('recBtn').textContent = '■ Stop';
      $('recStatus').textContent = 'recording…';
    } catch (e) { toast('Mic error: ' + e.message, 'err'); }
  } else {
    const blob = rec.takeWavBlob();
    rec.stop(); recording = false;
    $('recBtn').textContent = '● Record sample';
    $('recStatus').textContent = 'idle';
    samples.push({ name: `recording-${samples.length + 1}.wav`, blob });
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
