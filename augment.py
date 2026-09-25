import io
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
import torch

from config import (
    BRICKWALL_EDGE_HZ,
    BRICKWALL_PROB,
    F_HI_HZ_LOG_RANGE,
    F_LO_CHOICES,
    HOP_SIZE,
    LEVEL_PEAK_DBFS_RANGE,
    MIC_HALF_ORDER_CHOICES,
    MP3_PROB,
    NOISE_ALPHA_RANGE,
    NOISE_FMIN_HZ,
    PATTERNS,
    POWER_FLOOR,
    RMS_FLOOR,
    SAMPLE_RATE,
    SIR_DB_RANGE,
    SOURCE_SLOTS,
    VOICED_THRESHOLD,
    VOICELESS_RMS_DBFS_RANGE,
)

STAGE_ORDER = ("scene", "room", "mic", "level")
(SALT_PATTERN, SALT_SCENE, SALT_ROOM, SALT_MIC, SALT_LEVEL, SALT_CODEC,
 SALT_CHUNK, SALT_NOISE, SALT_PERM, SALT_SHUFFLE, SALT_BN_RECAL, SALT_CALIB) = range(12)
PATTERN_NAMES = tuple(p["name"] for p in PATTERNS)
PATTERN_STAGES = {p["name"]: frozenset(p["stages"]) for p in PATTERNS}
SRC_EMPTY, SRC_COLORED, SRC_POOL = 0, 1, 2

ACTIVE_WIN_S, ACTIVE_HOP_S, ACTIVE_CUT_DB = 0.100, 0.050, 25.0
NORM_FREQ_MIN, NORM_FREQ_MAX = 1e-4, 0.999
CORNER_GAP = 1e-4
TAN_ARG_MIN, TAN_ARG_MARGIN = 1e-7, 1e-4
PEAK_DIVIDE_FLOOR = 1e-30
REFLECT_PAD = 4096
CODEC_THREADS = 4
CODEC_LEAD_S = 0.5

_codec_pool = None


def item_seed(*keys):
    return int(np.random.SeedSequence([int(k) for k in keys])
               .generate_state(1, dtype=np.uint64)[0])


def stage_rng(seed, key, salt):
    return np.random.default_rng(item_seed(seed, salt, *key))


def draw_augmentation(pools, rooms, seed, key, n, has_voice):
    rng = stage_rng(seed, key, SALT_PATTERN)
    pattern = PATTERN_NAMES[int(rng.integers(len(PATTERNS)))]
    stages = PATTERN_STAGES[pattern]
    seg = np.zeros((SOURCE_SLOTS, n), dtype=np.float32)
    src_kind = np.zeros(SOURCE_SLOTS, dtype=np.int64)
    src_point = np.zeros(SOURCE_SLOTS, dtype=bool)
    alpha = np.zeros(SOURCE_SLOTS, dtype=np.float32)
    sir_db = np.zeros(SOURCE_SLOTS, dtype=np.float32)
    voiceless_dbfs = np.zeros(SOURCE_SLOTS, dtype=np.float32)
    n_sources = 0
    if "scene" in stages:
        rng = stage_rng(seed, key, SALT_SCENE)
        n_sources = 1 + int(rng.integers(SOURCE_SLOTS))
        usable = [p for p in pools if has_voice or p.label_safe]
        for j in range(n_sources):
            pool = usable[int(rng.integers(len(usable)))]
            src_point[j] = pool.point
            if pool.kind == "colored":
                src_kind[j] = SRC_COLORED
                alpha[j] = float(rng.uniform(*NOISE_ALPHA_RANGE))
                seg[j] = torch.randn(
                    n, generator=torch.Generator().manual_seed(
                        item_seed(seed, SALT_NOISE, *key, j))).numpy()
            else:
                src_kind[j] = SRC_POOL
                seg[j] = pool.draw_window(n, rng)
            if has_voice:
                sir_db[j] = float(rng.uniform(*SIR_DB_RANGE))
            else:
                voiceless_dbfs[j] = float(rng.uniform(*VOICELESS_RMS_DBFS_RANGE))

    rir = [np.zeros(1, dtype=np.float32)] * (SOURCE_SLOTS + 1)
    rir_p = np.zeros(SOURCE_SLOTS + 1, dtype=np.int64)
    rir_on = np.zeros(SOURCE_SLOTS + 1, dtype=np.int64)
    if "room" in stages:
        rng = stage_rng(seed, key, SALT_ROOM)
        for slot in (0, *(j + 1 for j in range(n_sources) if src_point[j])):
            rir[slot], rir_p[slot] = rooms[int(rng.integers(len(rooms)))].draw(rng)
            rir_on[slot] = 1

    f_lo = f_hi = 0.0
    order = brickwall = codec = 0
    if "mic" in stages:
        rng = stage_rng(seed, key, SALT_MIC)
        f_lo = float(F_LO_CHOICES[int(rng.integers(len(F_LO_CHOICES)))])
        f_hi = float(np.exp(rng.uniform(*np.log(F_HI_HZ_LOG_RANGE))))
        order = int(MIC_HALF_ORDER_CHOICES[int(rng.integers(len(MIC_HALF_ORDER_CHOICES)))])
        brickwall = int(rng.random() < BRICKWALL_PROB)
        codec = int(stage_rng(seed, key, SALT_CODEC).random() < MP3_PROB)

    level_dbfs = 0.0
    if "level" in stages:
        level_dbfs = float(stage_rng(seed, key, SALT_LEVEL).uniform(*LEVEL_PEAK_DBFS_RANGE))

    return {"stage_on": np.array([int(s in stages) for s in STAGE_ORDER], np.int64),
            "has_voice": np.int64(int(has_voice)),
            "seg": seg, "src_kind": src_kind, "alpha": alpha,
            "sir_db": sir_db, "voiceless_dbfs": voiceless_dbfs,
            "rir": rir, "rir_p": rir_p, "rir_on": rir_on,
            "f_lo": np.float32(f_lo), "f_hi": np.float32(f_hi),
            "order": np.int64(order),
            "brickwall": np.int64(brickwall),
            "codec": np.int64(codec),
            "level_dbfs": np.float32(level_dbfs)}


def _rms(x):
    return torch.sqrt(x.pow(2).mean(dim=-1))


def shape_colored(white, alpha, sr):
    L = white.shape[-1]
    f = torch.fft.rfftfreq(L, 1.0 / sr, device=white.device)
    fmin = max(NOISE_FMIN_HZ, sr / L)
    scale = torch.clamp(f, min=fmin).unsqueeze(0) ** (-alpha.unsqueeze(-1) / 2.0)
    spec = torch.fft.rfft(white) * scale
    spec[..., 0] = 0.0
    n = torch.fft.irfft(spec, n=L)
    return n / (_rms(n) + RMS_FLOOR).unsqueeze(-1)


def voiced_power(chunk, per, hop):
    N = chunk.shape[-1]
    F = per.shape[-1]
    frame = torch.clamp((torch.arange(N, device=chunk.device) + hop // 2) // hop,
                        max=F - 1)
    mask = (torch.nan_to_num(per, nan=0.0)[:, frame] >= VOICED_THRESHOLD).float()
    cnt = mask.sum(dim=-1)
    pw = (chunk.pow(2) * mask).sum(dim=-1) / torch.clamp(cnt, min=1.0) + POWER_FLOOR
    fallback = chunk.pow(2).mean(dim=-1)
    return torch.where(cnt > 0, pw, fallback)


def active_power(seg, sr):
    L = seg.shape[-1]
    w = int(ACTIVE_WIN_S * sr)
    h = int(ACTIVE_HOP_S * sr)
    if w > L:
        return seg.pow(2).mean(dim=-1)
    pw = seg.unfold(-1, w, h).pow(2).mean(dim=-1)
    db = 10.0 * torch.log10(pw + POWER_FLOOR)
    keep = (db >= db.amax(dim=-1, keepdim=True) - ACTIVE_CUT_DB).float()
    return (pw * keep).sum(dim=-1) / torch.clamp(keep.sum(dim=-1), min=1.0)


def apply_codec(y, idx, sr, lo):
    global _codec_pool
    sel = torch.nonzero(idx != 0, as_tuple=False).squeeze(-1)
    if not sel.numel():
        return y
    start = max(0, lo - int(CODEC_LEAD_S * sr))
    host = y[sel, start:].detach().to("cpu", torch.float32).numpy()
    n = host.shape[-1]

    def code_one(j):
        src = host[j]
        peak = float(np.abs(src).max())
        g = 1.0 / peak if peak > RMS_FLOOR else 1.0
        buf = io.BytesIO()
        with sf.SoundFile(buf, "w", samplerate=sr, channels=1, format="MP3",
                          subtype="MPEG_LAYER_III") as fh:
            fh.write((src * g).astype(np.float32))
        buf.seek(0)
        z, _ = sf.read(buf, dtype="float32", always_2d=False)
        if len(z) != n:
            raise SystemExit(f"augment: mp3 changed length {n} -> {len(z)}; "
                             f"labels would drift")
        return z / g

    if _codec_pool is None:
        _codec_pool = ThreadPoolExecutor(max_workers=CODEC_THREADS)
    out = np.stack(list(_codec_pool.map(code_one, range(host.shape[0]))))
    z = torch.from_numpy(out).to(y.device, y.dtype)
    y[sel, start:] = match_level(z, y[sel, start:], lo - start)
    return y


def butter_logmag2(lt, fc, order, sr, highpass):
    nyq = sr / 2.0
    wn = torch.clamp(fc / nyq, min=NORM_FREQ_MIN, max=NORM_FREQ_MAX)
    ltc = torch.log(torch.tan(math.pi * wn / 2.0))
    z = 2.0 * order.unsqueeze(-1) * (lt.unsqueeze(0) - ltc.unsqueeze(-1))
    return torch.nn.functional.logsigmoid(z if highpass else -z)


def min_phase(logmag, n):
    c = torch.fft.irfft(logmag, n=n)
    w = torch.zeros(n, device=logmag.device, dtype=c.dtype)
    w[0] = 1.0
    w[1:(n + 1) // 2] = 2.0
    if n % 2 == 0:
        w[n // 2] = 1.0
    return torch.exp(torch.fft.rfft(c * w, n=n))


def match_level(y, ref, lo):
    r = _rms(ref[..., lo:])
    ry = _rms(y[..., lo:])
    scale = torch.where((r > RMS_FLOOR) & (ry > RMS_FLOOR), r / ry,
                        torch.ones_like(r))
    return y * scale.unsqueeze(-1)


def convolve_aligned(sig, h, p, L):
    nfft = 1 << (sig.shape[-1] + h.shape[-1] - 1).bit_length()
    y = torch.fft.irfft(torch.fft.rfft(sig, nfft) * torch.fft.rfft(h, nfft), nfft)
    idx = p.unsqueeze(-1) + torch.arange(L, device=sig.device)
    return torch.gather(y, -1, idx)


def render(batch, ctx):
    sr, hop = SAMPLE_RATE, HOP_SIZE
    audio = batch["audio"]
    B, T = audio.shape
    C = ctx
    stage_on = batch["stage_on"]

    def rows_of(stage):
        return torch.nonzero(stage_on[:, STAGE_ORDER.index(stage)].bool(),
                             as_tuple=False).squeeze(-1)

    scene_rows, mic_rows, room_rows = rows_of("scene"), rows_of("mic"), rows_of("room")
    target = audio
    parts = torch.zeros(B, SOURCE_SLOTS, T, device=audio.device)
    if scene_rows.numel():
        rows = scene_rows
        seg = batch["seg"][rows]
        kind = batch["src_kind"][rows]
        col = torch.nonzero(kind.reshape(-1) == SRC_COLORED, as_tuple=False).squeeze(-1)
        flat = seg.reshape(-1, T)
        if col.numel():
            flat[col] = shape_colored(flat[col],
                                      batch["alpha"][rows].reshape(-1)[col], sr)
        hv = batch["has_voice"][rows].bool()
        sig_pow = voiced_power(audio[rows][:, C:], batch["periodicity"][rows], hop)
        act = active_power(seg[..., C:], sr)
        mean_pow = seg[..., C:].pow(2).mean(dim=-1)
        sir = batch["sir_db"][rows]
        dbfs = batch["voiceless_dbfs"][rows]
        s_voiced = torch.sqrt(sig_pow.unsqueeze(-1)
                              / (10.0 ** (sir / 10.0) * act + POWER_FLOOR))
        s_voiced = torch.where(act > POWER_FLOOR, s_voiced,
                               torch.zeros_like(s_voiced))
        s_quiet = 10.0 ** (dbfs / 20.0) / (torch.sqrt(mean_pow) + RMS_FLOOR)
        s_quiet = torch.where(mean_pow > POWER_FLOOR, s_quiet,
                              torch.zeros_like(s_quiet))
        scale = torch.where(hv.unsqueeze(-1), s_voiced, s_quiet)
        active = (kind != SRC_EMPTY).float()
        parts[rows] = seg * scale.unsqueeze(-1) * active.unsqueeze(-1)

    if room_rows.numel():
        rows = room_rows
        sig = torch.cat([target[rows].unsqueeze(1), parts[rows]], dim=1)
        h = batch["rir"][rows]
        on = batch["rir_on"][rows].bool()
        wet = convolve_aligned(sig.reshape(-1, T), h.reshape(-1, h.shape[-1]),
                               batch["rir_p"][rows].reshape(-1), T)
        wet = match_level(wet.reshape(len(rows), SOURCE_SLOTS + 1, T), sig, C)
        mixed = torch.where(on.unsqueeze(-1), wet, sig)
        t2 = target.clone()
        t2[rows] = mixed[:, 0]
        target = t2
        p2 = parts.clone()
        p2[rows] = mixed[:, 1:]
        parts = p2

    y = target + parts.sum(dim=1)

    if mic_rows.numel():
        rows = mic_rows
        x = y[rows]
        P = min(REFLECT_PAD, T - 1)
        left = 2.0 * x[:, :1] - x[:, 1:P + 1].flip(-1)
        right = 2.0 * x[:, -1:] - x[:, -P - 1:-1].flip(-1)
        xe = torch.cat([left, x, right], dim=-1)
        f = torch.fft.rfftfreq(T + 2 * P, 1.0 / sr, device=x.device)
        ordr = batch["order"][rows].float()
        lt = torch.log(torch.tan(torch.clamp(math.pi * f / sr, min=TAN_ARG_MIN,
                                             max=math.pi / 2 - TAN_ARG_MARGIN)))
        la_hp = butter_logmag2(lt, batch["f_lo"][rows], ordr, sr, True)
        nyq = sr / 2.0
        w_lo = torch.clamp(batch["f_lo"][rows] / nyq, min=NORM_FREQ_MIN, max=NORM_FREQ_MAX)
        fhi = torch.clamp(batch["f_hi"][rows] / nyq, min=NORM_FREQ_MIN, max=NORM_FREQ_MAX)
        fhi = torch.maximum(fhi, w_lo + CORNER_GAP) * nyq
        la_lp = butter_logmag2(lt, fhi, ordr, sr, False)
        wall = batch["brickwall"][rows].bool().unsqueeze(-1)
        edge = (f.unsqueeze(0) - fhi.unsqueeze(-1)) / BRICKWALL_EDGE_HZ
        la_lp = torch.where(wall, torch.nn.functional.logsigmoid(-edge), la_lp)
        H = min_phase(la_hp + la_lp, T + 2 * P)
        z = torch.fft.irfft(torch.fft.rfft(xe) * H, n=T + 2 * P)[:, P:P + T]
        z = match_level(z, x, C)
        y2 = y.clone()
        y2[rows] = z
        y = y2

    y = apply_codec(y, batch["codec"], sr, C)

    y = y[:, C:]
    level_rows = rows_of("level")
    if level_rows.numel():
        rows = level_rows
        peak = y[rows].abs().amax(dim=-1)
        gain = torch.where(peak > 0,
                           10.0 ** (batch["level_dbfs"][rows] / 20.0)
                           / torch.clamp(peak, min=PEAK_DIVIDE_FLOOR),
                           torch.ones_like(peak))
        y2 = y.clone()
        y2[rows] = y[rows] * gain.unsqueeze(-1)
        y = y2
    peak = y.abs().amax(dim=-1)
    guard = torch.where(peak > 1.0, 1.0 / peak, torch.ones_like(peak))
    y = y * guard.unsqueeze(-1)
    return y
