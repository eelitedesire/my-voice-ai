// Shared microphone capture built on AudioWorklet.
//   - onChunk(Float32Array) called continuously with mono PCM at ctx.sampleRate
//   - encodeWav(chunks, sampleRate) -> Blob (16-bit PCM WAV) for enrollment uploads
export class MicRecorder {
  constructor({ onChunk = null, targetSampleRate = 16000 } = {}) {
    this.onChunk = onChunk;
    this.targetSampleRate = targetSampleRate;
    this.ctx = null;
    this.stream = null;
    this.node = null;
    this.source = null;
    this.sampleRate = null;
    this.recording = false;
    this.captured = []; // used when onChunk not provided (enrollment mode)
  }

  async start() {
    // Ask the browser for 16 kHz directly; falls back to hardware rate if ignored.
    this.ctx = new (window.AudioContext || window.webkitAudioContext)({
      sampleRate: this.targetSampleRate,
    });
    this.sampleRate = this.ctx.sampleRate;
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    await this.ctx.audioWorklet.addModule('/js/pcm-worklet.js');
    this.source = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, 'pcm-processor');
    this.node.port.onmessage = (e) => {
      const chunk = e.data; // Float32Array
      if (!this.recording) return;
      if (this.onChunk) this.onChunk(chunk, this.sampleRate);
      else this.captured.push(chunk);
    };
    this.source.connect(this.node);
    // Do NOT connect node to destination -> avoids echo/feedback.
    this.recording = true;
  }

  stop() {
    this.recording = false;
    if (this.source) this.source.disconnect();
    if (this.node) this.node.disconnect();
    if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
    if (this.ctx) this.ctx.close();
  }

  // Convert accumulated Float32 chunks into a WAV Blob (native sample rate).
  takeWavBlob() {
    const total = this.captured.reduce((n, c) => n + c.length, 0);
    const pcm = new Float32Array(total);
    let off = 0;
    for (const c of this.captured) { pcm.set(c, off); off += c.length; }
    this.captured = [];
    return encodeWav(pcm, this.sampleRate);
  }
}

export function floatToInt16(float32) {
  const out = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    let s = Math.max(-1, Math.min(1, float32[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

export function encodeWav(float32, sampleRate) {
  const pcm16 = floatToInt16(float32);
  const buffer = new ArrayBuffer(44 + pcm16.length * 2);
  const view = new DataView(buffer);
  const wr = (o, s) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };
  wr(0, 'RIFF');
  view.setUint32(4, 36 + pcm16.length * 2, true);
  wr(8, 'WAVE');
  wr(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  wr(36, 'data');
  view.setUint32(40, pcm16.length * 2, true);
  let off = 44;
  for (let i = 0; i < pcm16.length; i++, off += 2) view.setInt16(off, pcm16[i], true);
  return new Blob([view], { type: 'audio/wav' });
}
