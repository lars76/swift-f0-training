# SwiftF0 training

Training code and ONNX export for [SwiftF0](https://pypi.org/project/swift-f0/) 0.3.0,
a pitch detector for monophonic audio. `runs/seed1` holds the shipped model: the
PyTorch weights, the training log and `model.onnx`, the file the Python package runs.

## Model

The network has 14 386 trainable parameters. It reads 16 kHz audio and gives a pitch
and a confidence for every 256 samples (16 ms). The
[How it works](https://swift-f0.github.io/how/) article explains the features, the
network and the note segmentation in detail.

## Train

Install dependencies with [uv](https://docs.astral.sh/uv/) and pass the prepared
training split explicitly:

```bash
uv sync --locked
uvx --from huggingface_hub hf download lars1234/pitch-benchmark train.tar SHA256SUMS --repo-type dataset --local-dir dataset
(cd dataset && sha256sum -c --ignore-missing SHA256SUMS)   # macOS: shasum -a 256 -c --ignore-missing SHA256SUMS
tar -xf dataset/train.tar -C dataset
uv run python train.py --dataset dataset/train --out runs/mine
```

`uv sync` resolves for Linux x86-64 and macOS arm64. `--out` names the run directory
and defaults to `runs/local`. `--workers` sets the data loader
processes. `--stop-after` ends training after that many epochs without writing
`weights.pt`, for a quick check of the pipeline.

The download is 4.7 GB and extracts to about the same size again. Training runs on an NVIDIA GPU
when PyTorch finds one and on the CPU otherwise. The locked PyTorch build for Linux uses
CUDA 13.2 and needs a recent driver. On macOS arm64 training runs on the CPU and takes days.

The dataset is on [Hugging Face](https://huggingface.co/datasets/lars1234/pitch-benchmark)
(CC BY-NC-SA 4.0, research use) and can be rebuilt with the
[preparation code](https://github.com/lars76/pitch-benchmark/blob/main/prepare/README.md).
Each corpus holds `audio/NNNNNNN.flac` at 16 kHz and `labels/NNNNNNN.npz` with `pitch`
in Hz (0 where unvoiced) and `trusted`, one value per 256 samples. `augmentation/` holds
the noise, music, speech and room pools, each as `pool.flac` with an `index.json`, listed
in `pools.json`.

### The shipped run

`runs/seed1` was trained with `config.py` as committed and seed 1, on an RTX 4070 Laptop
GPU (8 GB) with PyTorch 2.14.0+cu132. The 270 epochs took 3.9 hours. The last line of its
`train_log.jsonl` reads raw pitch accuracy (RPA) 0.8795 and loss 1.2285. A new run on the same data does not
match it bit for bit, since cuDNN and cuFFT are not deterministic and the MP3 augmentation
depends on the bundled libsndfile.

`config.py` controls corpora, augmentation, training schedule and output paths. The
network is defined in `model.py`.

## Export ONNX

```bash
uv run python export_onnx.py --dataset dataset/train --out runs/mine
```

The exporter reads the run's `weights.pt`, checks the graph against the PyTorch model on
audio from the training split, and writes `model.onnx` and `export.json` alongside it.
`--out runs/seed1` re-exports the shipped run and reproduces its `model.onnx` byte for byte.
The file holds the whole pipeline in standard ONNX operators (opset 18): the three STFTs
as matrix products, the float32 network and the pitch and confidence decoding.

The graph takes `audio`, float32 of shape `[1, samples]` at 16 kHz, and the scalars
`fmin` and `fmax` in Hz, which restrict the pitch search to a band within 46.875 to
2093.75 Hz. It returns `pitch`, float64 in Hz, and `confidence`, float32 in 0 to 1,
both of shape `[1, max(1, samples // 256)]`. A frame whose highest pitch score lies
outside the band gets confidence 0. Every other frame keeps the confidence it has at the full
range. A band that holds no pitch bin gives NaN pitch. The graph does not gate
silence. The package and `web/predict.mjs` set the confidence of frames whose audio
peaks below 0.001 to 0 afterwards.

```python
import numpy as np, onnxruntime as ort

audio = np.zeros(16000, np.float32)  # one second of mono audio at 16 kHz
session = ort.InferenceSession("runs/seed1/model.onnx")
pitch, confidence = session.run(None, {"audio": audio[None],
                                       "fmin": np.asarray(46.875, np.float32),
                                       "fmax": np.asarray(2093.75, np.float32)})
```

## Build the browser runtime

The build runs on Linux or WSL on x86-64. It needs `uv` (after `uv sync`), `cmake`,
`ninja`, `git`, `curl`, `unzip`, `python3`, Node 20 or newer, and npm 10 or newer on
`PATH`. ONNX Runtime (ORT) requires npm alongside the Node executable. Activating
Emscripten below supplies a compatible pair.

```bash
git clone --branch v1.29.0 --depth 1 https://github.com/microsoft/onnxruntime.git onnxruntime
git -C onnxruntime submodule update --init --depth 1 cmake/external/emsdk

onnxruntime/cmake/external/emsdk/emsdk install 4.0.23
onnxruntime/cmake/external/emsdk/emsdk activate 4.0.23
source onnxruntime/cmake/external/emsdk/emsdk_env.sh

npm install onnxruntime-web@1.29.0
```

Install the matching `protoc` locally:

```bash
mkdir -p protoc
curl -L --fail https://github.com/protocolbuffers/protobuf/releases/download/v21.12/protoc-21.12-linux-x86_64.zip -o protoc/protoc.zip
unzip -o protoc/protoc.zip -d protoc
protoc/bin/protoc --version  # must print libprotoc 3.21.12
```

Then build:

```bash
bash web/build.sh \
  --model runs/seed1/model.onnx \
  --source-dir onnxruntime \
  --protoc protoc/bin/protoc \
  --js-package node_modules/onnxruntime-web \
  --output dist/
```

`node web/bench.mjs dist runs/seed1/model.onnx` times the build on ten seconds of
audio. Serve `dist/` over HTTP(S), then run predictions in the browser:

```javascript
import {createPredictor} from './dist/predict.mjs';

const predictor = await createPredictor(); // reuse for subsequent recordings
const {pitchHz, confidence, frameStepSeconds} = await predictor.predict(monoAudio16kHz);
// monoAudio16kHz: nonempty Float32Array of mono PCM at 16 kHz.
await predictor.dispose();
```

## Citation

The paper describes SwiftF0 0.1.x. Version 0.2.0 replaced its network with the learned
harmonic comb that this repository trains. If you use SwiftF0 in your research, please cite:

```bibtex
@misc{nieradzik2025swiftf0,
      title={SwiftF0: Fast and Accurate Monophonic Pitch Detection},
      author={Lars Nieradzik},
      year={2025},
      eprint={2508.18440},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2508.18440},
}
```
