import * as ort from './ort.wasm.min.mjs';

// Audio must already be mono Float32 PCM at SAMPLE_RATE Hz.
export const SAMPLE_RATE = 16000;
export const HOP_SIZE = 256;
export const FMIN = 46.875;
export const FMAX = 2093.75;
// Match the float32 pitch centers stored in the ONNX graph.
const PITCH_BINS = Float32Array.from({length: 95}, (_, i) => FMIN * (FMAX / FMIN) ** (i / 94));
const SILENCE_PEAK = 1e-3;

// Frames whose audio peaks below SILENCE_PEAK get confidence 0, as in the Python package.
function gateSilence(audio, confidence) {
  for (let i = 0; i < confidence.length; i++) {
    let peak = 0;
    const end = Math.min(audio.length, (i + 1) * HOP_SIZE);
    for (let j = i * HOP_SIZE; j < end; j++) peak = Math.max(peak, Math.abs(audio[j]));
    if (peak < SILENCE_PEAK) confidence[i] = 0;
  }
  return confidence;
}

export async function createPredictor() {
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = new URL('./', import.meta.url).href;
  const session = await ort.InferenceSession.create(new URL('model.onnx', import.meta.url).href,
    {executionProviders: ['wasm'], logSeverityLevel: 3});
  let closed = false;
  return {
    async predict(audio, fmin = FMIN, fmax = FMAX) {
      if (closed) throw new Error('Predictor has been disposed');
      fmin = Math.fround(Math.max(FMIN, fmin));
      fmax = Math.fround(Math.min(FMAX, fmax));
      if (!PITCH_BINS.some(hz => hz >= fmin && hz <= fmax))
        throw new RangeError(`Band ${fmin}-${fmax} Hz contains no pitch bin`);
      if (!(audio instanceof Float32Array) || !audio.length)
        throw new TypeError('Expected a nonempty Float32Array of mono audio');
      if (!audio.every(Number.isFinite)) throw new TypeError('Audio contains non-finite samples');
      const input = new ort.Tensor('float32', audio, [1, audio.length]);
      let output;
      try {
        output = await session.run({audio: input, fmin: new ort.Tensor('float32', Float32Array.of(fmin), []),
          fmax: new ort.Tensor('float32', Float32Array.of(fmax), [])});
        return {pitchHz: new Float64Array(output.pitch.data),
          confidence: gateSilence(audio, new Float32Array(output.confidence.data)),
          frameStepSeconds: HOP_SIZE / SAMPLE_RATE};
      } finally {
        input.dispose();
        if (output) for (const tensor of Object.values(output)) tensor.dispose();
      }
    },
    async dispose() {
      if (!closed) {closed = true; await session.release();}
    }
  };
}
