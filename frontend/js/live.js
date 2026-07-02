import { MicRecorder, floatToInt16 } from '/js/recorder.js';
import { TranscriptView, colorFor, fmtClock } from '/js/transcript.js';

const $ = (id) => document.getElementById(id);
let ws = null, rec = null, running = false, busy = false;
let clientChunk = 512;   // audio worklet frame size; overridden from /api/config
let startTime = 0, timerInt = null, exportName = 'sanuvia-transcript';
const view = new TranscriptView($('transcript'), { autoscroll: true });

function toast(msg, kind = '') {
  const t = $('toast'); t.textContent = msg; t.className = `toast show ${kind}`;
  setTimeout(() => (t.className = 'toast'), 2600);
}
function showDropzone(show) { $('dropzone').style.display = show ? 'block' : 'none'; }

// ---------- now-speaking ----------
function setNow(ev) {
  const col = ev.unknown ? '#64748b' : colorFor(ev.speaker_id, false);
  $('nowSwatch').style.background = col;
  $('nowName').textContent = ev.speaker || '—';
  $('nowName').className = 'name' + (ev.unknown ? ' unknown' : '');
  $('nowMeter').style.width = Math.max(0, Math.min(1, ev.confidence || 0)) * 100 + '%';
  $('nowConf').textContent = (ev.confidence || 0).toFixed(2);
}
const clearNow = () => setNow({ speaker: '—', unknown: true, confidence: 0 });

// ---------- waveform + timer ----------
const wave = $('wave'), wctx = wave.getContext('2d');
let levels = new Array(180).fill(0);
function pushLevel(f32) {
  let s = 0; for (let i = 0; i < f32.length; i++) s += f32[i] * f32[i];
  levels.push(Math.min(1, Math.sqrt(s / f32.length) * 3.5)); levels.shift();
}
function drawWave() {
  const w = wave.width = wave.clientWidth, h = wave.height = wave.clientHeight;
  wctx.clearRect(0, 0, w, h);
  const n = levels.length, bw = w / n;
  for (let i = 0; i < n; i++) {
    const bh = Math.max(2, levels[i] * h * 0.9);
    wctx.fillStyle = running ? '#059669' : '#cbd5e1';
    wctx.fillRect(i * bw, (h - bh) / 2, bw * 0.7, bh);
  }
  if (running) requestAnimationFrame(drawWave);
}
function startTimer() { startTime = Date.now(); timerInt = setInterval(() => { $('timer').textContent = fmtClock((Date.now() - startTime) / 1000); }, 250); }
function stopTimer() { clearInterval(timerInt); }

// ---------- config panel ----------
const CFG = [
  ['id_threshold', 'ID threshold', 0, 1, 0.01], ['switch_margin', 'Switch margin', 0, 0.5, 0.01],
  ['min_switch_windows', 'Switch windows', 1, 10, 1], ['ema_alpha', 'EMA alpha', 0.1, 1, 0.05],
  ['change_sim_threshold', 'Change sensitivity', 0.2, 0.9, 0.05],
  ['asr_tick_sec', 'ASR tick (s)', 0.2, 2, 0.05], ['vad_threshold', 'VAD threshold', 0.1, 0.9, 0.05],
];
async function loadConfig() {
  const cfg = await (await fetch('/api/config')).json();
  if (cfg.client_chunk_samples) clientChunk = cfg.client_chunk_samples;
  const grid = $('configGrid'); grid.innerHTML = '';
  for (const [k, label, min, max, step] of CFG) {
    const d = document.createElement('div');
    d.innerHTML = `<small>${label}</small><input type="number" id="cfg_${k}" value="${cfg[k]}" min="${min}" max="${max}" step="${step}">`;
    grid.appendChild(d);
  }
  const { speakers } = await (await fetch('/api/speakers')).json();
  $('spkCount').textContent = `${speakers.length} enrolled`;
}
$('applyCfg').addEventListener('click', async () => {
  const body = {}; for (const [k] of CFG) body[k] = parseFloat($(`cfg_${k}`).value);
  await fetch('/api/config', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  toast('Config applied', 'ok');
});

// ---------- toolbar (search / copy) ----------
$('search').addEventListener('input', (e) => view.search(e.target.value));
$('copyBtn').addEventListener('click', async () => {
  const txt = view.plainText();
  if (!txt) return toast('Nothing to copy yet', 'err');
  await navigator.clipboard.writeText(txt); toast('Transcript copied', 'ok');
});

// ============================================================ LIVE RECORDING
async function start() {
  if (busy) return toast('Finish the current upload first', 'err');
  view.clear(); showDropzone(false); exportName = 'sanuvia-live';
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws/stream`);
  ws.binaryType = 'arraybuffer';

  ws.onopen = async () => {
    $('connDot').className = 'dot live'; $('connText').textContent = 'connected';
    rec = new MicRecorder({ chunkSize: clientChunk, onChunk: (f32) => { pushLevel(f32); if (ws?.readyState === 1) ws.send(floatToInt16(f32).buffer); } });
    try { await rec.start(); } catch (e) { toast('Mic error: ' + e.message, 'err'); return stop(); }
    ws.send(JSON.stringify({ type: 'start', sample_rate: rec.sampleRate }));
    running = true;
    $('recDot').className = 'dot live'; $('recText').textContent = 'recording';
    $('startBtn').disabled = true; $('stopBtn').disabled = false; $('uploadBtn').disabled = true;
    startTimer(); drawWave();
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
      case 'now': setNow(ev); break;
      case 'block_open':
      case 'block_label':
      case 'transcript_partial':
      case 'transcript_final': view.upsert(ev); break;
      case 'error': toast('Server: ' + ev.message, 'err'); break;
    }
  };
  ws.onclose = () => { $('connDot').className = 'dot'; $('connText').textContent = 'disconnected'; };
  ws.onerror = () => toast('WebSocket error', 'err');
}
function stop() {
  running = false;
  try { if (ws?.readyState === 1) ws.send(JSON.stringify({ type: 'stop' })); } catch {}
  if (rec) rec.stop();
  setTimeout(() => { if (ws) ws.close(); }, 800);
  $('recDot').className = 'dot'; $('recText').textContent = 'idle';
  $('startBtn').disabled = false; $('stopBtn').disabled = true; $('uploadBtn').disabled = false;
  stopTimer(); clearNow();
}
$('startBtn').addEventListener('click', start);
$('stopBtn').addEventListener('click', stop);

// ============================================================ FILE UPLOAD
$('uploadBtn').addEventListener('click', () => { if (!running) $('fileInput').click(); });
$('fileInput').addEventListener('change', (e) => { if (e.target.files[0]) processFile(e.target.files[0]); });

const dz = $('dropzone');
['dragover', 'dragenter'].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add('hover'); }));
['dragleave', 'drop'].forEach((ev) => dz.addEventListener(ev, () => dz.classList.remove('hover')));
dz.addEventListener('drop', (e) => { e.preventDefault(); if (e.dataTransfer.files[0]) processFile(e.dataTransfer.files[0]); });

async function processFile(file) {
  if (running) return toast('Stop the live session first', 'err');
  busy = true;
  exportName = file.name.replace(/\.[^.]+$/, '') || 'transcript';
  view.clear(); showDropzone(false);
  $('progWrap').style.display = 'block'; setProgress('Uploading…', 2);
  $('uploadBtn').disabled = true; $('startBtn').disabled = true;
  $('recText').textContent = 'file'; $('recDot').className = 'dot live';

  const fd = new FormData(); fd.append('file', file, file.name);
  let resp;
  try { resp = await fetch('/api/transcribe-file', { method: 'POST', body: fd }); }
  catch (err) { toast('Upload failed: ' + err.message, 'err'); return endUpload(); }

  const reader = resp.body.getReader(), dec = new TextDecoder();
  let buf = '', result = null;
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split('\n'); buf = lines.pop();
    for (const line of lines) {
      if (!line.trim()) continue;
      const m = JSON.parse(line);
      if (m.stage === 'error') { toast('Error: ' + m.message, 'err'); return endUpload(); }
      else if (m.stage === 'done') result = m;
      else setProgress(stageLabel(m), m.pct);
    }
  }
  setProgress('Done', 100);
  if (result) {
    const segs = result.segments || [];
    if (!segs.length) toast('No speech detected in file', 'err');
    segs.forEach((s, i) => view.upsert({ ...s, block_id: i, is_final: true }));
    $('spkCount').textContent = `${new Set(segs.map((s) => s.speaker)).size} speakers · ${fmtClock(result.duration || 0)}`;
    toast('Transcription complete', 'ok');
  }
  endUpload();
}
function endUpload() {
  busy = false; $('uploadBtn').disabled = false; $('startBtn').disabled = false;
  $('recText').textContent = 'idle'; $('recDot').className = 'dot';
  setTimeout(() => { $('progWrap').style.display = 'none'; }, 1200);
}
function stageLabel(m) {
  if (m.stage === 'decoding') return 'Decoding audio…';
  if (m.stage === 'diarizing') return 'Detecting speakers…';
  if (m.stage === 'transcribing') return `Transcribing… ${m.done || 0}/${m.total || '?'}`;
  return m.stage;
}
function setProgress(label, pct) { $('progStage').textContent = label; $('progBar').style.width = pct + '%'; $('progPct').textContent = Math.round(pct) + '%'; }

// ---------- init ----------
clearNow(); drawWave(); loadConfig(); showDropzone(true);
