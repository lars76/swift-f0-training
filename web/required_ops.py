"""Validate and snapshot the complete ONNX model. Derive required kernels and types."""

import argparse
import tempfile
from pathlib import Path

import onnx
import onnxruntime as ort
from onnxruntime.tools.ort_format_model.utils import create_config_from_models


def write_if_changed(path, content):
    if not path.exists() or path.read_text() != content:
        path.write_text(content)


def requirements(model, work):
    config = work / "required_operators.config"
    with tempfile.TemporaryDirectory(dir=work, prefix="operators-") as directory:
        snapshots = []
        for level in (ort.GraphOptimizationLevel.ORT_DISABLE_ALL, ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                      ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED, ort.GraphOptimizationLevel.ORT_ENABLE_ALL):
            path = Path(directory) / f"level{int(level)}.ort"
            options = ort.SessionOptions()
            options.intra_op_num_threads = options.inter_op_num_threads = 1
            options.log_severity_level = 3
            options.graph_optimization_level = level
            options.optimized_model_filepath = str(path)
            options.add_session_config_entry("session.save_model_format", "ORT")
            ort.InferenceSession(str(model), options, providers=["CPUExecutionProvider"])
            snapshots.append(path)
        temporary = Path(directory) / "operators.config"
        create_config_from_models(snapshots, temporary, enable_type_reduction=True)
        content = "\n".join(line for line in temporary.read_text().splitlines() if not line.startswith("#"))
    write_if_changed(config, "# Union of original and optimized kernels/types. Deployment stays ONNX.\n"
                     + content + "\n")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--ort-version", required=True)
    args = parser.parse_args()
    if ort.__version__ != args.ort_version:
        parser.error(f"Expected the project environment with ONNX Runtime {args.ort_version}; run uv sync")
    payload = args.model.read_bytes()
    graph = onnx.load_model_from_string(payload)
    if any(tensor.data_location == onnx.TensorProto.EXTERNAL for tensor in graph.graph.initializer):
        parser.error("The model must embed all weights")
    onnx.checker.check_model(graph, full_check=True)
    metadata = {entry.key: entry.value for entry in graph.metadata_props}
    inputs = {value.name: value.type.tensor_type for value in graph.graph.input}
    outputs = {value.name: value.type.tensor_type for value in graph.graph.output}
    initializers = {value.name for value in graph.graph.initializer}
    if (
        set(inputs) != {"audio", "fmin", "fmax"}
        or set(inputs) & initializers
        or inputs["audio"].elem_type != onnx.TensorProto.FLOAT
        or any(inputs[name].elem_type != onnx.TensorProto.FLOAT or len(inputs[name].shape.dim) != 0
               for name in ("fmin", "fmax"))
        or len(inputs["audio"].shape.dim) != 2
        or inputs["audio"].shape.dim[0].dim_value != 1
        or not inputs["audio"].shape.dim[1].dim_param
        or set(outputs) != {"pitch", "confidence"}
        or outputs["pitch"].elem_type != onnx.TensorProto.DOUBLE
        or outputs["confidence"].elem_type != onnx.TensorProto.FLOAT
        or any(len(value.shape.dim) != 2 or value.shape.dim[0].dim_value != 1 for value in outputs.values())
        or metadata.get("sample_rate") != "16000"
        or metadata.get("hop_size") != "256"
    ):
        parser.error("Expected the complete 16 kHz / 256-hop audio model produced by export_onnx.py")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    snapshot = args.work_dir / "model.onnx"
    if not snapshot.exists() or snapshot.read_bytes() != payload:
        snapshot.write_bytes(payload)
    requirements(snapshot, args.work_dir)


if __name__ == "__main__":
    main()
