# Changelog

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The version is that of the [swift-f0](https://github.com/lars76/swift-f0) release whose model
this code trains and exports.

## [Unreleased]

## [0.3.0] - 2026-09-25

### Added
- First public release: the training code and ONNX export for the model of swift-f0 0.3.0.
- `runs/seed1` holds the shipped run: `weights.pt`, `train_log.jsonl`, `model.onnx` and `export.json`. It was trained with `config.py` and seed 1 in 270 epochs. `export_onnx.py` re-exports its `model.onnx` byte for byte.
- `web/build.sh` builds ONNX Runtime Web 1.29.0 for the model, and `web/predict.mjs` runs it in the browser.

[Unreleased]: https://github.com/lars76/swift-f0-training/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/lars76/swift-f0-training/releases/tag/v0.3.0
