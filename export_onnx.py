import argparse
import copy
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import soundfile as sf
import torch
from onnx import helper as h
from onnx import numpy_helper as nh
from onnxruntime.quantization.shape_inference import quant_pre_process
from torch import nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

from config import (
    CORPORA,
    HOP_SIZE,
    ONNX_CHECK_CLIPS_PER_CORPUS,
    ONNX_CHECK_MAX_SECONDS,
    ONNX_CHECK_SEED,
    OUT,
    SAMPLE_RATE,
)
from model import (
    DECODE_WINDOW,
    MAX_HZ,
    MIN_HZ,
    N_BINS,
    SPEC_LOG_FLOOR,
    WINDOWS,
    SwiftF0Net,
    expand_time_symmetry,
    extract_f0,
)

CHECK_FRAMES = int(ONNX_CHECK_MAX_SECONDS * SAMPLE_RATE) // HOP_SIZE
FULL_BAND = {"fmin": np.array(MIN_HZ, np.float32), "fmax": np.array(MAX_HZ, np.float32)}
SPLITS = {512: (32, 16), 1024: (32, 32), 2048: (64, 32)}


def ordinary_conv(weight, *, padding, dilation=1, bias=None):
    weight = expand_time_symmetry(weight).detach().clone()
    conv = nn.Conv1d if weight.dim() == 3 else nn.Conv2d
    layer = conv(weight.shape[1], weight.shape[0], weight.shape[2:], padding=padding, dilation=dilation,
                 bias=bias is not None)
    layer.weight = nn.Parameter(weight)
    if bias is not None:
        layer.bias = nn.Parameter(bias.detach().clone())
    return layer.eval()


class InferenceCNN(nn.Module):
    """Expand tied temporal taps once and fold eval-mode BatchNorm into weights."""

    def __init__(self, source):
        super().__init__()
        self.stem = fuse_conv_bn_eval(ordinary_conv(source.stem[0].weight, padding=1), source.stem[1])
        self.blocks = nn.ModuleList()
        for block in source.blocks:
            layers = []
            for conv, norm in ((block.c1, block.n1), (block.c2, block.n2)):
                plain = ordinary_conv(conv.weight, padding=conv.padding, dilation=conv.dilation)
                layers.append(fuse_conv_bn_eval(plain, norm))
            self.blocks.append(nn.ModuleList(layers))
        self.proj = fuse_conv_bn_eval(ordinary_conv(source.proj[0].weight, padding=(0, 1)), source.proj[1])
        self.pitch = copy.deepcopy(source.pitch)
        self.correctness_proj = copy.deepcopy(source.correctness_proj)
        head = source.correctness_head
        self.conf1 = fuse_conv_bn_eval(ordinary_conv(head[0].weight, padding=1, bias=head[0].bias), head[1])
        self.conf2 = copy.deepcopy(head[3])

    def forward(self, features):
        hidden = self.stem(features).relu()
        for conv1, conv2 in self.blocks:
            hidden = (hidden + conv2(conv1(hidden).relu())).relu()
        hidden = torch.nn.functional.pad(hidden[:, :, :N_BINS + 1], (0, 0, 1, 0))
        hidden = self.proj(hidden).relu()
        conf = torch.cat([self.correctness_proj(hidden).amax(dim=2),
                          features.mean(dim=(1, 2), keepdim=True).squeeze(1)], dim=1)
        return (self.pitch(hidden).squeeze(1).transpose(1, 2), self.conf2(self.conf1(conf).relu()).squeeze(1))


def check_audio(source, root):
    """Sample training corpora, with gain/noise variants and silence, for export checks."""
    paths_by_corpus = {corpus: sorted((root / corpus / "audio").glob("*.flac"))
                       for corpus in sorted(name for name, _ in CORPORA)}
    missing = [corpus for corpus, paths in paths_by_corpus.items() if len(paths) < ONNX_CHECK_CLIPS_PER_CORPUS]
    if missing:
        raise ValueError(f"Checks need at least {ONNX_CHECK_CLIPS_PER_CORPUS} mono {SAMPLE_RATE} Hz FLAC files "
                         f"in each <dataset>/<corpus>/audio directory; missing: {', '.join(missing)}")
    rng = np.random.default_rng(ONNX_CHECK_SEED)
    features, audios = [], []

    def append(audio):
        audios.append(np.ascontiguousarray(audio[None]))
        value = source.features(torch.from_numpy(audio).unsqueeze(0)).contiguous()
        value = value.repeat(1, 1, 1, (CHECK_FRAMES + value.shape[-1] - 1) // value.shape[-1])
        features.append(value[..., :CHECK_FRAMES].contiguous().numpy())

    for paths in paths_by_corpus.values():
        for index in rng.choice(len(paths), ONNX_CHECK_CLIPS_PER_CORPUS, replace=False):
            path = paths[index]
            with sf.SoundFile(path) as audio_file:
                if audio_file.samplerate != SAMPLE_RATE or audio_file.channels != 1:
                    raise ValueError(f"Expected mono {SAMPLE_RATE} Hz audio: {path}")
                audio = audio_file.read(frames=int(ONNX_CHECK_MAX_SECONDS * SAMPLE_RATE), dtype="float32")
            if len(audio) < HOP_SIZE or not np.isfinite(audio).all():
                raise ValueError(f"Audio is too short or non-finite: {path}")
            append(audio)
            noisy = audio.copy()
            rms = max(float(np.sqrt(np.mean(noisy * noisy))), 1e-5)
            noisy += rng.normal(0, rms * 0.2, noisy.shape).astype(np.float32)
            noisy *= np.float32(10 ** rng.uniform(-1.5, 0))
            append(noisy)
    append(np.zeros(SAMPLE_RATE, np.float32))
    return features, audios


def prune(model):
    live, nodes = {value.name for value in [*model.graph.output, *model.graph.input]}, []
    for node in reversed(model.graph.node):
        if live.intersection(node.output):
            nodes.append(node)
            live.update(node.input)
    tensors = [tensor for tensor in model.graph.initializer if tensor.name in live]
    del model.graph.node[:]
    model.graph.node.extend(reversed(nodes))
    del model.graph.initializer[:]
    model.graph.initializer.extend(tensors)
    del model.graph.value_info[:]
    return model


def dynamic_time(model):
    del model.graph.value_info[:]
    model.graph.input[0].type.tensor_type.shape.dim[3].dim_param = "frames"
    for output in model.graph.output:
        output.type.tensor_type.shape.dim[1].dim_param = "frames"
    return model


def compact(model):
    """Deduplicate constants, shorten private names, and discard export metadata."""
    model = prune(model)
    aliases, constants, seen = {}, [], {}
    for tensor in model.graph.initializer:
        array = nh.to_array(tensor)
        key = (array.dtype.str, array.shape, array.tobytes())
        if key in seen:
            aliases[tensor.name] = seen[key]
        else:
            seen[key] = tensor.name
            constants.append(tensor)
    del model.graph.initializer[:]
    model.graph.initializer.extend(constants)
    public = {value.name for value in [*model.graph.input, *model.graph.output]}
    names = {}

    def rename(value):
        value = aliases.get(value, value)
        if not value or value in public:
            return value
        if value not in names:
            names[value] = f"v{len(names)}"
        return names[value]

    for node in model.graph.node:
        node.input[:] = [rename(value) for value in node.input]
        node.output[:] = [rename(value) for value in node.output]
        node.ClearField("name")
    for tensor in model.graph.initializer:
        tensor.name = rename(tensor.name)
    for message in [model, model.graph, *model.graph.node, *model.graph.initializer,
                    *model.graph.input, *model.graph.output]:
        message.ClearField("doc_string")
        message.ClearField("metadata_props")
    model.ClearField("producer_name")
    model.ClearField("producer_version")
    if model.functions or any(node.domain not in ("", "ai.onnx") for node in model.graph.node):
        raise ValueError("Export must contain only standard ONNX operators and no local functions")
    onnx.checker.check_model(model, full_check=True)
    return model


def audio_model(source, core):
    """Wrap the network in a matrix-product STFT and pitch and confidence decoding, all standard ONNX operators."""
    model = onnx.compose.add_prefix(copy.deepcopy(core), "cnn/")
    front, back, tensors = [], [], []

    def constant(name, value, dtype=np.int64):
        tensors.append(nh.from_array(np.asarray(value, dtype=dtype), name))
        return name

    def node(nodes, op, inputs, output, **attrs):
        nodes.append(h.make_node(op, inputs, [output], **attrs))
        return output

    zero = constant("zero", [0])
    one = constant("axis1", [1])
    minus_one = constant("minus_one", -1.0, np.float32)
    last = constant("last", [-1])
    hop = constant("hop", HOP_SIZE)
    sample_axis = constant("sample_axis", 1)
    shape = node(front, "Shape", ["audio"], "audio_shape")
    samples = node(front, "Gather", [shape, sample_axis], "samples", axis=0)
    frames = node(front, "Div", [samples, hop], "frames_floor")
    frames = node(front, "Max", [frames, constant("one", 1)], "frames")
    frames = node(front, "Unsqueeze", [frames, zero], "frame_end")
    # Match the package's minimum of one frame for clips shorter than a hop.
    short = node(front, "Sub", [hop, samples], "short_padding")
    zero_scalar = constant("zero_scalar", 0)
    short = node(front, "Max", [short, zero_scalar], "right_padding")
    short = node(front, "Unsqueeze", [short, zero], "right_padding_vector")
    pads = node(front, "Concat", [constant("pad_prefix", [0, 0, 0]), short], "short_pads", axis=0)
    audio = node(front, "Pad", ["audio", pads], "padded_audio", mode="constant")
    floor = constant("log_floor", float(np.float32(SPEC_LOG_FLOOR)), np.float32)
    length = node(front, "Max", [samples, hop], "length")
    blocks = node(front, "Div", [node(front, "Add", [length, constant("hop_minus_one", HOP_SIZE - 1)], "length_up"),
                                 hop], "blocks")
    tail = node(front, "Sub", [node(front, "Mul", [blocks, hop], "rounded_length"), length], "tail")
    tail = node(front, "Unsqueeze", [tail, zero], "tail_vector")
    block_shape = constant("block_shape", [-1, HOP_SIZE])

    def table(name, rows, cols, period, kind):
        # cos or -sin of 2*pi*(r*c mod period)/period, folded into a constant when the session loads
        product = node(front, "Mul", [constant(name + "_rows", np.arange(rows).reshape(rows, 1)),
                                      constant(name + "_cols", np.arange(cols).reshape(1, cols))], name + "_product")
        modulus = constant(name + "_period", period)
        turns = node(front, "Div", [product, modulus], name + "_turns")
        wrapped = node(front, "Mul", [turns, modulus], name + "_wrapped")
        angle = node(front, "Cast", [node(front, "Sub", [product, wrapped], name + "_remainder")], name + "_index",
                     to=onnx.TensorProto.FLOAT)
        angle = node(front, "Mul", [angle, constant(name + "_scale", 2 * np.pi / period, np.float32)], name + "_angle")
        if kind == "Cos":
            return node(front, "Cos", [angle], name)
        return node(front, "Mul", [node(front, "Sin", [angle], name + "_sin"), minus_one], name)

    warps = []
    for i, size in enumerate(WINDOWS):
        prefix = f"stft{i}/"
        n1, n2 = SPLITS[size]
        head = constant(prefix + "pad_head", [0, size // 2, 0])
        pads = node(front, "Concat", [head, tail], prefix + "pads_head", axis=0)
        pads = node(front, "Add", [pads, constant(prefix + "pad_tail", [0, 0, 0, size // 2])], prefix + "pads")
        padded = node(front, "Pad", [audio, pads], prefix + "padded", mode="constant")
        rows = node(front, "Reshape", [padded, block_shape], prefix + "blocks")
        views = []
        for j in range(size // HOP_SIZE):
            end = node(front, "Add", [frames, constant(prefix + f"offset{j}", [j])], prefix + f"end{j}")
            start = constant(prefix + f"start{j}", [j])
            views.append(node(front, "Slice", [rows, start, end, zero], prefix + f"view{j}"))
        window = constant(prefix + "window", getattr(source, f"window_{i}").numpy(), np.float32)
        framed = node(front, "Concat", views, prefix + "frames", axis=1)
        framed = node(front, "Mul", [framed, window], prefix + "windowed")
        signal = node(front, "Reshape", [framed, constant(prefix + "split", [-1, n1, n2])], prefix + "split_frames")
        # Cooley-Tukey: DFT of length n1 down the columns, twiddles, DFT of length n2 along the rows
        real = node(front, "MatMul", [table(prefix + "cos1", n1, n1, n1, "Cos"), signal], prefix + "stage1_real")
        imag = node(front, "MatMul", [table(prefix + "sin1", n1, n1, n1, "Sin"), signal], prefix + "stage1_imag")
        twiddle_cos = table(prefix + "twiddle_cos", n1, n2, size, "Cos")
        twiddle_sin = table(prefix + "twiddle_sin", n1, n2, size, "Sin")
        mid_real = node(front, "Sub", [node(front, "Mul", [real, twiddle_cos], prefix + "rc"),
                                       node(front, "Mul", [imag, twiddle_sin], prefix + "is")], prefix + "mid_real")
        mid_imag = node(front, "Add", [node(front, "Mul", [real, twiddle_sin], prefix + "rs"),
                                       node(front, "Mul", [imag, twiddle_cos], prefix + "ic")], prefix + "mid_imag")
        cos2, sin2 = table(prefix + "cos2", n2, n2, n2, "Cos"), table(prefix + "sin2", n2, n2, n2, "Sin")
        out_real = node(front, "Sub", [node(front, "MatMul", [mid_real, cos2], prefix + "rr"),
                                       node(front, "MatMul", [mid_imag, sin2], prefix + "ii")], prefix + "real")
        out_imag = node(front, "Add", [node(front, "MatMul", [mid_real, sin2], prefix + "ri"),
                                       node(front, "MatMul", [mid_imag, cos2], prefix + "ir")], prefix + "imag")
        power = node(front, "Add", [node(front, "Mul", [out_real, out_real], prefix + "real2"),
                                    node(front, "Mul", [out_imag, out_imag], prefix + "imag2")], prefix + "power_split")
        power = node(front, "Reshape", [node(front, "Transpose", [power], prefix + "power_t", perm=[0, 2, 1]),
                                        constant(prefix + "flat", [0, -1])], prefix + "power_flat")
        bins = constant(prefix + "bins", [size // 2 + 1])
        power = node(front, "Slice", [power, zero, bins, one], prefix + "power_bins")
        power = node(front, "Unsqueeze", [power, zero], prefix + "power")
        magnitude = node(front, "Sqrt", [power], prefix + "magnitude")
        # Store only nonzero filter coefficients. Runtimes reconstruct the exact FP32 matrix.
        filters = getattr(source, f"lin2log_{i}").numpy().T.copy()
        indices = np.flatnonzero(filters).astype(np.int32)
        empty = node(front, "ConstantOfShape", [constant(prefix + "flat_shape", [filters.size])],
                     prefix + "zeros")
        filled = node(front, "ScatterElements", [empty, constant(prefix + "indices", indices, np.int32),
                      constant(prefix + "values", filters.ravel()[indices], np.float32)], prefix + "filled", axis=0)
        weights = node(front, "Reshape", [filled, constant(prefix + "shape", filters.shape)], prefix + "filters")
        warped = node(front, "MatMul", [magnitude, weights], prefix + "warped")
        logged = node(front, "Log", [node(front, "Add", [warped, floor], prefix + "shifted")], prefix + "logged")
        transposed = node(front, "Transpose", [logged], prefix + "transposed", perm=[0, 2, 1])
        warps.append(node(front, "Unsqueeze", [transposed, one], prefix + "features"))
    node(front, "Concat", warps, "cnn/features", axis=1)

    centers = source.pitch_bin_centers.numpy()
    centers_name = constant("centers", centers, np.float32)
    low = node(back, "GreaterOrEqual", [centers_name, "fmin"], "above_min")
    high = node(back, "LessOrEqual", [centers_name, "fmax"], "below_max")
    band = node(back, "And", [low, high], "band")
    negative_inf = constant("negative_inf", -np.inf, np.float32)
    logits = node(back, "Where", [band, "cnn/pitch", negative_inf], "masked_logits")
    peak = node(back, "ArgMax", [logits], "peak", axis=-1, keepdims=1, select_last_index=0)
    offsets = constant("offsets", np.arange(-DECODE_WINDOW, DECODE_WINDOW + 1))
    positions = node(back, "Add", [peak, offsets], "positions")
    decode_pads = constant("decode_pads", [0, 0, DECODE_WINDOW, 0, 0, DECODE_WINDOW])
    padded = node(back, "Pad", [logits, decode_pads, negative_inf], "padded_logits", mode="constant")
    gather_pos = node(back, "Add", [positions, constant("decode_window", DECODE_WINDOW)], "gather_positions")
    selected = node(back, "GatherElements", [padded, gather_pos], "selected", axis=-1)
    selected = node(back, "Cast", [selected], "selected_double", to=onnx.TensorProto.DOUBLE)
    weights = node(back, "Softmax", [selected], "decode_weights", axis=-1)
    clipped = node(back, "Clip", [positions, zero_scalar, constant("last_bin", N_BINS - 1)], "clipped")
    log_centers = constant("log_centers", torch.log(source.pitch_bin_centers.double()).numpy(), np.float64)
    centers_selected = node(back, "Gather", [log_centers, clipped], "selected_centers", axis=0)
    weighted = node(back, "Mul", [weights, centers_selected], "weighted_centers")
    mean = node(back, "ReduceSum", [weighted, last], "log_pitch", keepdims=0)
    node(back, "Exp", [mean], "pitch")
    full_peak = node(back, "ArgMax", ["cnn/pitch"], "full_peak", axis=-1, keepdims=0, select_last_index=0)
    inside = node(back, "Gather", [band, full_peak], "full_peak_in_band", axis=0)
    raw = node(back, "Sigmoid", ["cnn/confidence"], "raw_confidence")
    node(back, "Where", [inside, raw, constant("zero_confidence", 0.0, np.float32)], "confidence")
    nodes = [*front, *model.graph.node, *back]
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    model.graph.initializer.extend(tensors)
    del model.graph.input[:]
    model.graph.input.extend([
        h.make_tensor_value_info("audio", onnx.TensorProto.FLOAT, [1, "samples"]),
        h.make_tensor_value_info("fmin", onnx.TensorProto.FLOAT, []),
        h.make_tensor_value_info("fmax", onnx.TensorProto.FLOAT, []),
    ])
    del model.graph.output[:]
    model.graph.output.extend([
        h.make_tensor_value_info("pitch", onnx.TensorProto.DOUBLE, [1, "frames"]),
        h.make_tensor_value_info("confidence", onnx.TensorProto.FLOAT, [1, "frames"]),
    ])
    return model


def session(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def validate(baseline, result, features, rows):
    exported = session(result)
    original = session(baseline)
    count = 0

    def check(value):
        nonlocal count
        expected = original.run(None, {"features": value})
        actual = exported.run(["pitch", "confidence"], {"features": value})
        for a, b in zip(actual, expected):
            if not np.isfinite(a).all():
                raise ValueError("Non-finite export output")
            np.testing.assert_array_equal(a, b, err_msg=f"frames={value.shape[-1]}")
        count += 1

    for value in features:
        check(value)
    rng = np.random.default_rng(716)
    for frames in (1, 2, 7, 8, 31, 62, 63, 64, 127, 128, 129, 255, 256, 257, 625):
        shape = (1, 3, rows, frames)
        check(rng.normal(-3, 3, shape).astype(np.float32))
        check(np.zeros(shape, np.float32))
        check(np.full(shape, np.float32(np.log(1e-8)), np.float32))
        check(np.resize(np.array([-1e6, 1e6], np.float32), shape))
    return count


def validate_audio(source, core, complete, result, audios):
    """Check STFT, feature mapping, decoder, and the final serialized graph."""
    debug = copy.deepcopy(complete)
    debug.graph.output.extend([
        h.make_tensor_value_info("cnn/features", onnx.TensorProto.FLOAT, [1, 3, source.n_log, "frames"]),
        *(h.make_tensor_value_info(f"stft{i}/magnitude", onnx.TensorProto.FLOAT,
                                  [1, "frames", size // 2 + 1]) for i, size in enumerate(WINDOWS)),
    ])
    detailed, final, network = session(debug), session(result), session(core)
    rng = np.random.default_rng(717)
    cases = list(audios)
    for length in (1, HOP_SIZE - 1, HOP_SIZE, HOP_SIZE + 1, 2 * HOP_SIZE - 1,
                   1023, 1024, 1025, SAMPLE_RATE + 1):
        cases.extend([np.zeros((1, length), np.float32), rng.normal(0, 0.1, (1, length)).astype(np.float32)])
    for hz in (55, 220, 440, 1000):
        cases.append(np.sin(np.arange(SAMPLE_RATE, dtype=np.float32) * (2 * np.pi * hz / SAMPLE_RATE))[None])
    stats = {"audio_cases": 0, "max_feature_delta_vs_torch": 0.0,
             "max_voiced_pitch_cents_vs_torch_frontend": 0.0,
             "max_confidence_delta_vs_torch_frontend": 0.0, "voicing_flips_vs_torch_frontend": 0}
    for audio in cases:
        feed = {"audio": audio, **FULL_BAND}
        pitch, confidence, features, *magnitudes = detailed.run(None, feed)
        for a, b in zip(final.run(None, feed), (pitch, confidence)):
            np.testing.assert_array_equal(a, b)
            if not np.isfinite(a).all():
                raise ValueError("Non-finite audio-model output")
        padded = np.pad(audio, ((0, 0), (0, max(HOP_SIZE - audio.shape[1], 0))))
        samples = torch.from_numpy(padded)
        n_frames = padded.shape[1] // HOP_SIZE
        np.testing.assert_equal(pitch.shape, (1, n_frames))
        warps = []
        for i, (size, magnitude) in enumerate(zip(WINDOWS, magnitudes)):
            stft = torch.stft(torch.nn.functional.pad(samples, (size // 2, size // 2)),
                              n_fft=size, hop_length=HOP_SIZE, window=getattr(source, f"window_{i}"),
                              center=False, return_complex=True)
            expected = stft.abs()[..., :n_frames].transpose(1, 2).numpy()
            np.testing.assert_allclose(magnitude, expected, rtol=1e-4,
                                       atol=5e-4 * max(1.0, float(np.abs(audio).max())))
            warp = magnitude @ getattr(source, f"lin2log_{i}").numpy().T
            warps.append(np.log(warp + SPEC_LOG_FLOOR).transpose(0, 2, 1))
        np.testing.assert_allclose(features, np.stack(warps, axis=1), atol=1e-5, rtol=1e-5)
        logits, conf_logits = network.run(None, {"features": features})
        expected_pitch = extract_f0(torch.from_numpy(logits), source.pitch_bin_centers).numpy()
        np.testing.assert_allclose(pitch, expected_pitch, rtol=1e-12, atol=1e-10)
        np.testing.assert_allclose(confidence, torch.from_numpy(conf_logits).sigmoid().numpy(), atol=1e-6, rtol=1e-5)
        old_features = source.features(samples).numpy()
        old_logits, old_conf = network.run(None, {"features": old_features})
        old_pitch = extract_f0(torch.from_numpy(old_logits), source.pitch_bin_centers).numpy()
        old_conf = torch.from_numpy(old_conf).sigmoid().numpy()
        voiced = (confidence >= 0.5) & (old_conf >= 0.5)
        cents = np.abs(1200 * np.log2(pitch / old_pitch))
        stats["audio_cases"] += 1
        stats["max_feature_delta_vs_torch"] = max(stats["max_feature_delta_vs_torch"],
                                                   float(np.abs(features - old_features).max()))
        stats["max_voiced_pitch_cents_vs_torch_frontend"] = max(
            stats["max_voiced_pitch_cents_vs_torch_frontend"], float(cents[voiced].max()) if voiced.any() else 0.0)
        stats["max_confidence_delta_vs_torch_frontend"] = max(
            stats["max_confidence_delta_vs_torch_frontend"], float(np.abs(confidence - old_conf).max()))
        stats["voicing_flips_vs_torch_frontend"] += int(np.count_nonzero((confidence >= 0.5) != (old_conf >= 0.5)))
    if stats["voicing_flips_vs_torch_frontend"] or stats["max_voiced_pitch_cents_vs_torch_frontend"] > 2.0 \
            or stats["max_confidence_delta_vs_torch_frontend"] > 0.05:
        raise ValueError(f"Front end deviates from PyTorch beyond the export tolerances: {stats}")
    centers = source.pitch_bin_centers.numpy()
    edges = [(100.0, 500.0), (centers[0], centers[0]), (centers[-1], centers[-1]), (centers[10], centers[10]),
             (centers[-1] - 1.0, 5000.0)]
    for audio, (low, high) in zip(cases[::-1], edges):
        bounds = {"audio": audio, "fmin": np.array(low, np.float32), "fmax": np.array(high, np.float32)}
        pitch, confidence, features, *_ = detailed.run(None, bounds)
        logits, conf_logits = network.run(None, {"features": features})
        band = (source.pitch_bin_centers >= np.float32(low)) & (source.pitch_bin_centers <= np.float32(high))
        masked = torch.from_numpy(logits).masked_fill(~band, -torch.inf)
        expected = extract_f0(masked, source.pitch_bin_centers).numpy()
        np.testing.assert_allclose(pitch, expected, rtol=1e-12, atol=1e-10)
        kept = band.numpy()[logits.argmax(axis=-1)]
        expected_confidence = np.where(kept, torch.from_numpy(conf_logits).sigmoid().numpy(), 0.0)
        np.testing.assert_allclose(confidence, expected_confidence, atol=1e-6, rtol=1e-5)
        if not np.isfinite(pitch).all():
            raise ValueError("Non-finite pitch inside a valid band")
        for a, b in zip(final.run(None, bounds), (pitch, confidence)):
            np.testing.assert_array_equal(a, b)
    stats["search_band_cases"] = len(edges)
    return stats


def main():
    parser = argparse.ArgumentParser(description="Export one audio-to-Hz/confidence ONNX model.")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to the prepared train/ split")
    parser.add_argument("--out", type=Path, default=Path(OUT) / "local")
    args = parser.parse_args()
    out_dir = args.out
    checkpoint = out_dir / "weights.pt"
    output = out_dir / "model.onnx"
    if not CORPORA or ONNX_CHECK_CLIPS_PER_CORPUS < 1 or CHECK_FRAMES < 1:
        parser.error("Export checks require corpora, a positive clip count, and at least one frame")
    if not checkpoint.is_file():
        parser.error(f"Checkpoint not found: {checkpoint}")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    ort.disable_telemetry_events()
    source = SwiftF0Net().eval()
    source.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
    cnn = InferenceCNN(source).eval()
    with torch.inference_mode():
        audio = torch.sin(torch.arange(SAMPLE_RATE, dtype=torch.float32) * (2 * np.pi * 440 / SAMPLE_RATE))[None]
        torch.testing.assert_close(source(audio), cnn(source.features(audio)), atol=5e-4, rtol=5e-4)
    print(f"Sampling audio for the export checks (seed {ONNX_CHECK_SEED})...", flush=True)
    with torch.inference_mode():
        features, audios = check_audio(source, args.dataset)
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Build and validate in isolation. Publish the requested output only on success.
    with tempfile.TemporaryDirectory(prefix="swiftf0-export-", dir=output.parent) as temp:
        temp = Path(temp)
        fp32, prepared = temp / "fp32.onnx", temp / "prepared.onnx"
        sample = torch.from_numpy(features[0])
        print("Exporting BatchNorm-folded network...", flush=True)
        with torch.no_grad():
            torch.onnx.export(cnn, (sample,), str(fp32), input_names=["features"],
                              output_names=["pitch", "confidence"], opset_version=18,
                              dynamo=True, external_data=False)
        quant_pre_process(fp32, prepared, skip_symbolic_shape=True)
        with torch.inference_mode():
            exported = session(onnx.load(prepared))
            for value in features:
                for a, b in zip(exported.run(None, {"features": value}), cnn(torch.from_numpy(value))):
                    np.testing.assert_allclose(a, b.numpy(), atol=5e-4, rtol=5e-4)
        baseline = dynamic_time(onnx.load(prepared))
        with torch.inference_mode():
            exported = session(baseline)
            for value in features:
                for frames in (1, 7, 64, 129):
                    part = np.ascontiguousarray(value[..., :frames])
                    for a, b in zip(exported.run(None, {"features": part}), cnn(torch.from_numpy(part))):
                        np.testing.assert_allclose(a, b.numpy(), atol=5e-4, rtol=5e-4)
        core = compact(copy.deepcopy(baseline))
        print("Checking dynamic lengths, sampled inputs, and exact graph-rewrite outputs...", flush=True)
        cases = validate(baseline, core, features, source.n_log)
        print("Adding the matrix-product STFT and Hz/confidence decoding...", flush=True)
        complete = audio_model(source, core)
        result = compact(copy.deepcopy(complete))
        print("Checking the complete audio pipeline...", flush=True)
        with torch.inference_mode():
            audio_checks = validate_audio(source, core, complete, result, audios)
        h.set_model_props(result, {
            "sample_rate": str(SAMPLE_RATE), "hop_size": str(HOP_SIZE),
            "minimum_samples": "1", "inputs": "audio, fmin, fmax", "frontend": "matrix-product STFT (float32)",
            "network": "float32",
            "checkpoint_sha256": checkpoint_sha256,
        })
        candidate = temp / "model.onnx"
        onnx.save(result, candidate, save_as_external_data=False)
        report = {"checkpoint_sha256": checkpoint_sha256,
                  "onnx_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                  "onnx_bytes": candidate.stat().st_size,
                  "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__},
                  "operators": dict(sorted(Counter(node.op_type for node in result.graph.node).items())),
                  "validation": {"exact_rewrite_cases": cases, "sampled_clips": len(audios), **audio_checks}}
        saved = session(onnx.load(candidate))
        for a, b in zip(saved.run(None, {"audio": audios[0], **FULL_BAND}),
                        session(result).run(None, {"audio": audios[0], **FULL_BAND})):
            np.testing.assert_array_equal(a, b)
        (out_dir / "export.json").write_text(json.dumps(report, indent=2) + "\n")
        os.replace(candidate, output)
    print(f"Saved {output}: {report['onnx_bytes']:,} bytes; {cases} exact core checks passed.")
    print(json.dumps(report["validation"], indent=2))


if __name__ == "__main__":
    main()
