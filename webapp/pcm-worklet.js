/**
 * AudioWorklet processor that forwards raw PCM chunks to the main thread.
 * Runs on the audio-rendering thread — no GC pressure on the main thread.
 */
class PCMProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0]?.[0];
    if (ch && ch.length > 0) {
      // Transfer a copy to the main thread
      this.port.postMessage(new Float32Array(ch));
    }
    return true;
  }
}
registerProcessor("pcm-processor", PCMProcessor);
