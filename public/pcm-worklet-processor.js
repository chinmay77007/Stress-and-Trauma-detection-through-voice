// pcm-worklet-processor.js
//
// Runs on the dedicated audio rendering thread (NOT the main thread) --
// this is what replaces the deprecated ScriptProcessorNode. Two jobs:
//
//   1. Adaptive VAD: tracks a running noise-floor estimate and gates on
//      (noiseFloor * multiplier) instead of a fixed RMS constant. This
//      adapts to the actual room/mic instead of chopping quiet speech in
//      a loud room or letting steady background noise through in a quiet
//      one. A short "hangover" keeps sending briefly after speech drops
//      below threshold, so trailing consonants don't get cut off.
//
//   2. Downsampling to 16kHz Int16 PCM (what Vosk expects), done here so
//      it never blocks the main/UI thread.
//
// Speech frames are posted to the main thread as transferable ArrayBuffers
// for the WebSocket to send on.
//
// NOTE: this is an adaptive energy-based VAD, not a learned model. If you
// want to upgrade to true Silero VAD later, a package like
// @ricky0123/vad-web runs the ONNX model on the main thread instead of in
// a worklet (worklets can't easily load WASM/ONNX runtimes) -- it would
// replace this file's gating logic while this file's resampling logic
// could stay as-is.

class PCMWorkletProcessor extends AudioWorkletProcessor {
  constructor() {
    super();

    this.outputSampleRate = 16000;

    // process() is called with tiny ~128-sample chunks by the Web Audio
    // spec, which is far too small a window to get a stable RMS reading --
    // making a VAD decision at that granularity causes the gate to flicker
    // on/off mid-word and chop audio into garbage fragments. Instead we
    // accumulate into a window roughly matching the old ScriptProcessorNode
    // buffer size (~85ms at 48kHz) before deciding anything.
    this.frameSize = 4096;
    this.buffer = new Float32Array(this.frameSize);
    this.bufferIndex = 0;

    // --- Adaptive VAD state (now operating on stable ~85ms windows) ---
    this.noiseFloor = 0.003;      // starting estimate; adapts over time
    this.floorRiseAlpha = 0.98;   // how slowly the floor drifts UP (tracks steady noise)
    this.floorFallAlpha = 0.5;    // how quickly it drifts DOWN (so loud transients aren't "adopted" as noise)
    this.gateMultiplier = 2.2;    // speech must exceed noiseFloor * this to count
    this.minThreshold = 0.004;    // absolute floor, in case the room is dead silent
    this.hangoverWindows = 3;     // ~3 windows (~250ms) kept sending after speech drops, to avoid clipped word endings
    this.hangoverCounter = 0;

    this.port.onmessage = (event) => {
      if (event.data?.type === "reset") {
        this.noiseFloor = 0.003;
        this.hangoverCounter = 0;
        this.bufferIndex = 0;
      }
    };
  }

  process(inputs) {
    const input = inputs[0];

    if (!input || input.length === 0 || !input[0] || input[0].length === 0) {
      return true;
    }

    const channelData = input[0];

    // Accumulate into the larger window; process a full window whenever
    // it fills up (may happen 0 or more times per process() call).
    for (let i = 0; i < channelData.length; i++) {
      this.buffer[this.bufferIndex] = channelData[i];
      this.bufferIndex += 1;

      if (this.bufferIndex >= this.frameSize) {
        this._processWindow(this.buffer);
        this.bufferIndex = 0;
      }
    }

    return true;
  }

  _processWindow(frame) {
    // --- RMS over the full ~85ms window (stable, unlike per-128-sample) ---
    let sum = 0;
    for (let i = 0; i < frame.length; i++) {
      sum += frame[i] * frame[i];
    }
    const rms = Math.sqrt(sum / frame.length);

    // --- Adaptive noise-floor update ---
    if (rms < this.noiseFloor) {
      this.noiseFloor =
        this.floorFallAlpha * rms + (1 - this.floorFallAlpha) * this.noiseFloor;
    } else {
      this.noiseFloor =
        this.floorRiseAlpha * this.noiseFloor + (1 - this.floorRiseAlpha) * rms;
    }

    const threshold = Math.max(this.noiseFloor * this.gateMultiplier, this.minThreshold);
    const isSpeech = rms > threshold;

    if (isSpeech) {
      this.hangoverCounter = this.hangoverWindows;
    } else if (this.hangoverCounter > 0) {
      this.hangoverCounter -= 1;
    }

    const shouldSend = isSpeech || this.hangoverCounter > 0;

    if (!shouldSend) {
      return;
    }

    // --- Downsample to 16kHz, Float32 -> Int16 (linear interpolation) ---
    const ratio = sampleRate / this.outputSampleRate;
    const outputLength = Math.floor(frame.length / ratio);
    const pcm = new Int16Array(outputLength);

    for (let i = 0; i < outputLength; i++) {
      const position = i * ratio;
      const index = Math.floor(position);
      const nextIndex = Math.min(index + 1, frame.length - 1);
      const fraction = position - index;

      const sample = frame[index] * (1 - fraction) + frame[nextIndex] * fraction;
      const clamped = Math.max(-1, Math.min(1, sample));

      pcm[i] = clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff;
    }

    // Transfer ownership of the buffer for zero-copy postMessage
    this.port.postMessage({ type: "audio", buffer: pcm.buffer }, [pcm.buffer]);
  }
}

registerProcessor("pcm-processor", PCMWorkletProcessor);
