import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import HOP_SIZE, SAMPLE_RATE

HZ_GAP_FLOOR = 1e-9
SPEC_LOG_FLOOR = 1e-8

WINDOWS = (1024, 2048, 512)
N_BINS = 95
DECODE_WINDOW = 1
MIN_HZ, MAX_HZ = 46.875, 2093.75
CHANNELS = 12
BLOCKS = (("local", 1, 7, 3), ("harmonic", 2, 7, 1), ("harmonic", 3, 5, 1), ("harmonic", 5, 3, 1),
          ("harmonic", 7, 3, 1), ("local", 2, 7, 3))
PROJ_CHANNELS = 4
PROJ_FREQ_WIDTH = 5
CORRECTNESS_CHANNELS = 16


def build_logfreq_filters(n_fft, sample_rate, centers):
    n_lin = n_fft // 2 + 1
    lin_hz = np.linspace(0.0, sample_rate / 2.0, n_lin)
    centers = np.asarray(centers, dtype=np.float64)
    n_log = len(centers)
    fb = np.zeros((n_log, n_lin), dtype=np.float32)
    for i, fc in enumerate(centers):
        lo = centers[i - 1] if i > 0 else fc * fc / centers[1]
        hi = centers[i + 1] if i < n_log - 1 else fc * fc / centers[-2]
        tri = np.clip(np.minimum((lin_hz - lo) / (fc - lo + HZ_GAP_FLOOR),
                                 (hi - lin_hz) / (hi - fc + HZ_GAP_FLOOR)), 0.0, None)
        s = float(tri.sum())
        if s < 1e-6:
            tri = np.zeros(n_lin, dtype=np.float32)
            tri[int(np.argmin(np.abs(lin_hz - fc)))] = 1.0
            s = 1.0
        fb[i] = tri / s
    return torch.from_numpy(fb)


def expand_time_symmetry(p):
    if p.shape[-1] == 1:
        return p
    tail = p[..., 1:]
    return torch.cat([tail.flip(-1), p[..., :1], tail], dim=-1)


def symmetric_planes(shape):
    w = torch.empty(*shape[:-1], shape[-1] // 2 + 1)
    nn.init.kaiming_uniform_(w, a=math.sqrt(5))
    return nn.Parameter(w)


class TimeSym:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight = symmetric_planes(self.weight.shape)

    def forward(self, x):
        return self._conv_forward(x, expand_time_symmetry(self.weight), self.bias)


class TimeSymConv2d(TimeSym, nn.Conv2d):
    pass


class TimeSymConv1d(TimeSym, nn.Conv1d):
    pass


class HarmonicBlock(nn.Module):
    def __init__(self, ch, d, time_depth, freq_width):
        super().__init__()
        shape = {"kernel_size": (freq_width, time_depth), "padding": (d * (freq_width // 2), time_depth // 2),
                 "dilation": (d, 1), "bias": False}
        self.c1 = TimeSymConv2d(ch, ch, **shape)
        self.n1 = nn.BatchNorm2d(ch)
        self.c2 = TimeSymConv2d(ch, ch, **shape)
        self.n2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        return self.n2(self.c2(F.relu(self.n1(self.c1(x)))))


class SwiftF0Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.eff_bpo = (N_BINS - 1) / math.log2(MAX_HZ / MIN_HZ)
        nyquist = SAMPLE_RATE / 2.0
        self.n_log = math.ceil(self.eff_bpo * math.log2(nyquist / MIN_HZ))
        centers = MIN_HZ * 2.0 ** (np.arange(self.n_log) / self.eff_bpo)

        self.register_buffer("pitch_bin_centers",
                             torch.tensor(centers[:N_BINS], dtype=torch.float32), persistent=False)
        for i, w in enumerate(WINDOWS):
            win = torch.hann_window(w, dtype=torch.float64)
            self.register_buffer(f"window_{i}", win.float(), persistent=False)
            fb = build_logfreq_filters(w, SAMPLE_RATE, centers)
            self.register_buffer(f"lin2log_{i}", fb, persistent=False)

        self.stem = nn.Sequential(TimeSymConv2d(len(WINDOWS), CHANNELS, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(CHANNELS), nn.ReLU(inplace=True))
        self.blocks = nn.ModuleList(
            [HarmonicBlock(CHANNELS, round(self.eff_bpo * math.log2(v)) if kind == "harmonic" else v, depth, width)
             for kind, v, width, depth in BLOCKS])
        self.proj = nn.Sequential(TimeSymConv2d(CHANNELS, PROJ_CHANNELS, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(PROJ_CHANNELS), nn.ReLU(inplace=True))
        self.pitch = nn.Conv2d(PROJ_CHANNELS, 1, 1)
        self.correctness_proj = nn.Conv2d(PROJ_CHANNELS, PROJ_CHANNELS, (PROJ_FREQ_WIDTH, 1),
                                          padding=(PROJ_FREQ_WIDTH // 2, 0))
        self.correctness_head = nn.Sequential(
            TimeSymConv1d(PROJ_CHANNELS + 1, CORRECTNESS_CHANNELS, 3, padding=1),
            nn.BatchNorm1d(CORRECTNESS_CHANNELS), nn.ReLU(inplace=True),
            nn.Conv1d(CORRECTNESS_CHANNELS, 1, 1))

    def features(self, x):
        n_frames = x.shape[-1] // HOP_SIZE
        warps = []
        for i, w in enumerate(WINDOWS):
            win = getattr(self, f"window_{i}")
            fb = getattr(self, f"lin2log_{i}")
            st = torch.stft(F.pad(x, (w // 2, w // 2)), n_fft=w, hop_length=HOP_SIZE,
                            win_length=w, window=win, center=False,
                            normalized=False, onesided=True, return_complex=True)
            warps.append(torch.einsum("fl,blt->bft", fb, torch.abs(st)))
        return torch.stack([torch.log(s[..., :n_frames] + SPEC_LOG_FLOOR) for s in warps], dim=1)

    def forward(self, x):
        f = self.features(x)
        h = self.stem(f)
        for b in self.blocks:
            h = F.relu(h + b(h))
        h = self.proj(h)[:, :, :N_BINS]
        c = torch.cat([self.correctness_proj(h).amax(dim=2),
                       f.mean(dim=(1, 2), keepdim=True).squeeze(1)], dim=1)
        return (self.pitch(h).squeeze(1).transpose(1, 2), self.correctness_head(c).squeeze(1))


def extract_f0(logits, bin_centers):
    off = torch.arange(-DECODE_WINDOW, DECODE_WINDOW + 1, device=logits.device)
    pos = logits.argmax(dim=-1).unsqueeze(-1) + off
    w = F.pad(logits, (DECODE_WINDOW, DECODE_WINDOW), value=-torch.inf).gather(-1, pos + DECODE_WINDOW)
    log_centers = torch.log(bin_centers.double())[pos.clamp(0, logits.shape[-1] - 1)]
    return torch.exp((w.double().softmax(dim=-1) * log_centers).sum(dim=-1))
