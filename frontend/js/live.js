import { MicRecorder, floatToInt16 } from '/js/recorder.js';

const $ = (id) => document.getElementById(id);
let ws = null, rec = null, running = false;
const colors = ['#4f8cff', '#22c55e', '#f59e0b', '#ec4899', '#a855f7', '#14b8a6', '#f43f5e'];
const speakerColor = {};
function colorFor(id) {
  if (!id) return '#64748b';
  if (!(id in speakerColor)) speakerColor[id] = colors[Object.keys(speakerColor).length % colors.length];
  return speakerColor[id];
}
function toast(msg, kind = '') {
  const t = $('toast'); t.textContent = msg; t.className = `toast show ${kind}`;
  setTimeout(() => (t.className = 'toast'), 2600);
}
const fmt = (s) => { s = Math.max(0, s); const m = Math.floor(s / 60); return `${m}:${String(Math.floor(s % 60)).padStart(2, '0')}`; };

// ---- config panel ----
const CFG_FIELDS = [
  ['id_threshold', 'ID threshold', 0, 1, 0.01],
  ['switch_margin', 'Switch margin', 0, 0.5, 0.01],
  ['min_switch_windows', 'Switch windows', 1, 10, 1],
  ['ema_alpha', 'EMA alpha', 0.1, 1, 0.05],
  ['min_segment_sec', 'Min segment (s)', 0.2, 5, 0.1],
  ['vad_threshold', 'VAD threshold', 0.1, 0.9, 0.05],
];
async function loadConfig() {
  const cfg = await (await fetch('/api/config')).json();
  const grid = $('configGrid'); grid.innerHTML = '';
  for (const [key, label, min, max, step] of CFG_FIELDS) {
    const wrap = document.createElement('div');
    wrap.innerHTML = `<small>${label}</small><input type="number" id="cfg_${key}" value="${cfg[key]}" min="${min}" max="${max}" step="${step}">`;
    grid.appendChild(wrap);
  }
  $('spkCount').textContent = `${await countSpeakers()} enrolled`;
}
async function countSpeakers() {
  const { speakers } = await (await fetch('/api/speakers')).json();
  return speakers.length;
}
$('applyCfg').addEventListener('click', async () => {
  const body = {};
  for (const [key] of CFG_FIELDS) body[key] = parseFloat($(`cfg_${key}`).value);
  await fetch('/api/config', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  toast('Config applied', 'ok');
});

// ---- now-speaking indicator ----
function setNow(ev) {
  const col = ev.unknown ? '#64748b' : colorFor(ev.speaker_id);
  $('nowSwatch').style.background = col;
  $('nowName').textContent = ev.speaker;
  $('nowName').className = 'name' + (ev.unknown ? ' unknown' : '');
  const pct = Math.max(0, Math.min(1, ev.confidence)) * 100;
  $('nowMeter').style.width = pct + '%';
  $('nowMeter').style.background = col;
  $('nowConf').textContent = ev.confidence.toFixed(2);
}
function clearNow() { setNow({ speaker: '—', unknown: true, confidence: 0 }); }

// ---- transcript ----
function addTurn(ev) {
  const box = $('transcript');
  const hint = box.querySelector('.hint'); if (hint) hint.remove();
  const div = document.createElement('div');
  div.className = 'turn' + (ev.unknown ? ' unknown' : '');
  const col = ev.unknown ? '#64748b' : colorFor(ev.speaker_id);
  div.innerHTML =
    `<div class="who" style="color:${col}"><span class="swatch" style="background:${col}"></span>${ev.speaker}` +
    `<span class="conf">conf ${ev.confidence.toFixed(2)}</span></div>` +
    `<div class="what">${ev.text ? escapeHtml(ev.text) : '<em class="meta">…</em>'}</div>` +
    `<div class="ts">${fmt(ev.start)}–${fmt(ev.end)}</div>`;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}
function escapeHtml(s) { const d = document.createElement('div'); d.textContent = s; return d.innerHTML; }

// ---- streaming ----
async function start() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws/stream`);
  ws.binaryType = 'arraybuffer';

  ws.onopen = async () => {
    $('connDot').className = 'dot live'; $('connText').textContent = 'connected';
    rec = new MicRecorder({
      onChunk: (float32) => {
        if (ws && ws.readyState === WebSocket.OPEN) ws.send(floatToInt16(float32).buffer);
      },
    });
    try { await rec.start(); }
    catch (e) { toast('Mic error: ' + e.message, 'err'); return stop(); }
    ws.send(JSON.stringify({ type: 'start', sample_rate: rec.sampleRate }));
    running = true;
    $('startBtn').disabled = true; $('stopBtn').disabled = false;
  };

  ws.onmessage = (e) => {
    const ev = JSON.parse(e.data);
    switch (ev.type) {
      case 'ready': $('spkCount').textContent = `${ev.speakers} enrolled`; break;
      case 'vad':
        $('speechDot').className = 'dot' + (ev.active ? ' speech' : '');
        $('speechText').textContent = ev.active ? 'speech' : 'silence';
        if (!ev.active) clearNow();
        break;
      case 'partial': setNow(ev); break;
      case 'segment': addTurn(ev); break;
      case 'error': toast('Server: ' + ev.message, 'err'); break;
    }
  };
  ws.onclose = () => { $('connDot').className = 'dot'; $('connText').textContent = 'disconnected'; };
  ws.onerror = () => toast('WebSocket error', 'err');
}

function stop() {
  running = false;
  try { if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: 'stop' })); } catch {}
  if (rec) rec.stop();
  setTimeout(() => { if (ws) ws.close(); }, 300);
  $('startBtn').disabled = false; $('stopBtn').disabled = true;
  clearNow();
}

$('startBtn').addEventListener('click', start);
$('stopBtn').addEventListener('click', stop);
clearNow();
loadConfig();
