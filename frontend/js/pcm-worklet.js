// AudioWorklet that forwards raw mono Float32 PCM frames to the main thread.
// Buffers to ~1024-sample chunks to keep postMessage overhead low.
class PCMProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buf = new Float32Array(0);
    this._chunk = 1024;
  }
  process(inputs) {
    const input = inputs[0];
    if (input && input[0]) {
      const ch = input[0];
      const merged = new Float32Array(this._buf.length + ch.length);
      merged.set(this._buf, 0);
      merged.set(ch, this._buf.length);
      this._buf = merged;
      while (this._buf.length >= this._chunk) {
        const out = this._buf.slice(0, this._chunk);
        this._buf = this._buf.slice(this._chunk);
        this.port.postMessage(out, [out.buffer]);
      }
    }
    return true;
  }
}
registerProcessor('pcm-processor', PCMProcessor);
