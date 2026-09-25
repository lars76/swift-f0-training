import { readFileSync } from 'node:fs';
import { performance } from 'node:perf_hooks';
import { pathToFileURL } from 'node:url';

const usage = 'node web/bench.mjs (web|DIST_DIR) MODEL.onnx [THREADS]';
const [,, runtime, modelPath, threadsArg] = process.argv;
if (!runtime || !modelPath) { console.error(usage); process.exit(2); }
const threads = Number(threadsArg ?? 1);
const SAMPLE_RATE = 16000, SECONDS = 10, RUNS = 5, BATCH_MS = 500;

let ort;
if (runtime === 'web') {
  ort = await import('onnxruntime-web');
  ort.env.wasm.numThreads = threads;
} else {
  ort = await import(pathToFileURL(runtime + '/ort.wasm.min.mjs').href);
  ort.env.wasm.numThreads = threads;
  ort.env.wasm.wasmPaths = pathToFileURL(runtime + '/').href;
  ort.env.wasm.wasmBinary = readFileSync(runtime + '/ort-wasm-simd-threaded.wasm');
}
const n = SAMPLE_RATE * SECONDS;
const audio = new Float32Array(n);
let peak = 0;
for (let i = 0; i < n; i++) {
  const t = i / SAMPLE_RATE;
  audio[i] = [1, 2, 3].reduce((acc, k) => acc + Math.sin(2 * Math.PI * 440 * k * t) / k, 0);
  peak = Math.max(peak, Math.abs(audio[i]));
}
for (let i = 0; i < n; i++) audio[i] /= peak;

const options = { executionProviders: ['wasm'], logSeverityLevel: 3 };
const bytes = new Uint8Array(readFileSync(modelPath));
const loadStart = performance.now();
const session = await ort.InferenceSession.create(bytes, options);
const loadMs = performance.now() - loadStart;
const input = new ort.Tensor('float32', audio, [1, n]);
const feed = { audio: input, fmin: new ort.Tensor('float32', Float32Array.of(46.875), []),
  fmax: new ort.Tensor('float32', Float32Array.of(2093.75), []) };
for (let i = 0; i < 3; i++) await session.run(feed);
const runs = [];
for (let r = 0; r < RUNS; r++) {
  const start = performance.now();
  let calls = 0;
  while (performance.now() - start < BATCH_MS) { await session.run(feed); calls++; }
  runs.push((performance.now() - start) / calls);
}
runs.sort((a, b) => a - b);
const median = runs[Math.floor(RUNS / 2)];
console.log(JSON.stringify({ runtime: runtime.split('/').pop(), model: modelPath.split('/').pop(), bytes: bytes.length, threads,
  load_ms: +loadMs.toFixed(1), infer_ms: +median.toFixed(2), real_time_factor: +(SECONDS * 1e3 / median).toFixed(1) }));
await session.release();
