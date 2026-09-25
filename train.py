import argparse
import json
import math
import os
import time
from queue import Queue
from threading import Thread

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize
from scipy.special import expit

import augment
from augment import SALT_BN_RECAL, SALT_CALIB, item_seed
from config import (
    BN_RECAL_BATCHES,
    CALIB_CLIPS_PER_CORPUS,
    CALIB_MAX_SECONDS,
    CORRECTNESS_LOSS_WEIGHT,
    CORRECTNESS_TOL_CENTS,
    CTX_SECONDS,
    DEVICE,
    EPOCHS,
    HOP_SIZE,
    LR,
    LR_DECAY_FRAC,
    MARGIN_SECONDS,
    MIN_LR_FACTOR,
    OUT,
    POWER_FLOOR,
    SAMPLE_RATE,
    SEED,
    SWA_TAIL,
    WARMUP_EPOCHS,
    WEIGHT_DECAY,
    WORKERS,
)
from data import (
    any_voiced,
    attach_augmentation,
    build_loader,
    collate,
    load_corpora,
)
from model import SwiftF0Net, extract_f0
from pools import load_pools

LOG_FLOOR_HZ = 1e-6
CALIB_GRID_POINTS = 401
CALIB_MIN_FRAMES = 10000
CALIB_RATE_BOUNDS = (0.05, 0.95)


class WarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_epochs, train_epochs, steps_per_epoch,
                 min_lr_factor, decay_frac):
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.t_total = train_epochs * steps_per_epoch
        self.min_lr_factor = min_lr_factor
        if not 0.0 < decay_frac <= 1.0:
            raise SystemExit(f"lr_decay_frac must be in (0, 1], got {decay_frac!r}")
        self.decay_frac = decay_frac
        super().__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            p = float(step) / float(max(1, self.warmup_steps))
            return self.min_lr_factor + (1.0 - self.min_lr_factor) * p
        span = float(max(1.0, self.t_total - self.warmup_steps))
        hold = span * (1.0 - self.decay_frac)
        t = float(step - self.warmup_steps)
        if t < hold:
            return 1.0
        p = min(1.0, max(0.0, (t - hold) / max(1.0, span - hold)))
        return max(self.min_lr_factor, 0.5 * (1.0 + math.cos(math.pi * p)))


def augment_batch(batch, ctx, device):
    moved = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    return {"audio": augment.render(moved, ctx), "pitch": moved["pitch"]}


def rendered_batches(loader, ctx, device):
    queue = Queue(maxsize=2)

    def work():
        try:
            for batch in loader:
                queue.put(augment_batch(batch, ctx, device))
            queue.put(None)
        except BaseException as e:
            queue.put(e)

    Thread(target=work, daemon=True).start()
    while (item := queue.get()) is not None:
        if isinstance(item, BaseException):
            raise item
        yield item


def frame_masks(pitch):
    known = ~torch.isnan(pitch)
    return known, known & (pitch > 0)


def cents_error(f0, pitch):
    return 1200.0 * (torch.log2(f0.double().clamp_min(LOG_FLOOR_HZ))
                     - torch.log2(pitch.double().clamp_min(LOG_FLOOR_HZ))).abs()


def make_step(model, device):
    centers = model.pitch_bin_centers
    log_centers = torch.log(centers)
    nbins = log_centers.numel()

    def step(batch):
        audio = batch["audio"].float()
        pitch = batch["pitch"].float()
        logits, clogit = model(audio)
        T = pitch.shape[1]
        if logits.shape[1] != T:
            raise SystemExit(f"network produced {logits.shape[1]} frames for {T} label frames")

        known, valid = frame_masks(pitch)
        with torch.no_grad():
            err = cents_error(extract_f0(logits, centers), pitch)
            ctarget = (valid & (err <= CORRECTNESS_TOL_CENTS)).float()
        metrics = {}
        if valid.any():
            vl = logits[valid]
            lp = torch.log(pitch[valid])
            with torch.no_grad():
                k = (torch.searchsorted(log_centers, lp.contiguous(),
                                        right=True) - 1).clamp(0, nbins - 2)
                lo = log_centers[k]
                frac = ((lp - lo) / (log_centers[k + 1] - lo)).clamp(0.0, 1.0)
            ls = F.log_softmax(vl, dim=-1)
            pitch_loss = -((1.0 - frac) * ls.gather(1, k[:, None]).squeeze(1)
                           + frac * ls.gather(1, (k + 1)[:, None]).squeeze(1)).mean()
            metrics["rpa"] = float((err[valid] <= CORRECTNESS_TOL_CENTS).float().mean())
        else:
            pitch_loss = torch.tensor(0.0, device=device, requires_grad=True)
            metrics["rpa"] = float("nan")
        loss = pitch_loss
        metrics["pitch_loss"] = float(pitch_loss.detach())

        if known.any():
            correct_loss = F.binary_cross_entropy_with_logits(clogit[known],
                                                              ctarget[known])
            loss = loss + CORRECTNESS_LOSS_WEIGHT * correct_loss
            with torch.no_grad():
                metrics["correct_loss"] = float(correct_loss)
                metrics["correct_rate"] = float(ctarget[known].mean())
                metrics["correct_acc"] = float(((clogit[known] > 0)
                                                == (ctarget[known] >= 0.5)).float().mean())
        metrics["loss"] = float(loss.detach())
        return loss, metrics

    return step


def swa_average(paths):
    out, n = {}, 0
    for path in paths:
        sd = torch.load(path, map_location="cpu", weights_only=True)["state_dict"]
        for k, v in sd.items():
            if "num_batches_tracked" in k:
                out[k] = v.clone()
            elif k in out:
                out[k] += v.float()
            else:
                out[k] = v.float().clone()
        n += 1
    for k in out:
        if "num_batches_tracked" not in k:
            out[k] /= n
    return out


def recalibrate_bn(state_dict, corpora, pools, rooms, workers, ctx, device):
    net = SwiftF0Net().to(device)
    net.load_state_dict(state_dict)
    for m in net.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            m.reset_running_stats()
            m.momentum = None
    net.train()
    loader, sampler = build_loader(item_seed(SEED, SALT_BN_RECAL), workers, corpora, pools,
                                   rooms)
    sampler.set_epoch(1)
    n = 0
    with torch.no_grad():
        for b in loader:
            net(augment_batch(b, ctx, device)["audio"].float())
            n += 1
            if n >= BN_RECAL_BATCHES:
                break
    net.eval()
    return net.state_dict()


def calib_stream(corpora, pools, rooms, net, device):
    centers = net.pitch_bin_centers
    max_frames = int(CALIB_MAX_SECONDS * SAMPLE_RATE / HOP_SIZE)
    calib_seed = item_seed(SEED, SALT_CALIB)
    rng = np.random.default_rng(calib_seed)
    zs, ys, vs = [], [], []
    for d, (ds, _w) in enumerate(corpora):
        idxs = rng.choice(len(ds), size=min(CALIB_CLIPS_PER_CORPUS, len(ds)),
                          replace=False)
        for i in idxs:
            s = ds[int(i)]
            nf = min(max_frames, s["pitch"].size(-1))
            audio = ds.audio(int(i), 0, nf * HOP_SIZE)
            if float(np.mean(audio ** 2)) < POWER_FLOOR:
                continue
            truth = s["pitch"].reshape(-1)[:nf]
            per = s["voiced"].reshape(-1)[:nf]
            sample = attach_augmentation(pools, rooms, calib_seed, (d, int(i)),
                                         {"audio": audio, "pitch": truth.numpy(),
                                          "periodicity": per.numpy()},
                                         any_voiced(per.numpy()))
            x = augment.render(collate([sample]), 0).to(device)
            with torch.inference_mode():
                logits, clogit = net(x)
                z = clogit[0].cpu()
                f0 = extract_f0(logits[0], centers).cpu()
            m = min(len(z), len(truth))
            z, f0, truth = z[:m], f0[:m], truth[:m]
            keep, is_voiced = frame_masks(truth)
            y = is_voiced & (cents_error(f0, truth) <= CORRECTNESS_TOL_CENTS)
            zs.append(z[keep].double())
            ys.append(y[keep].double())
            vs.append(is_voiced[keep].double())
    z, y, v = (torch.cat(zs).numpy(), torch.cat(ys).numpy(), torch.cat(vs).numpy())
    rate = float((y > 0).mean())
    lo, hi = CALIB_RATE_BOUNDS
    if len(z) < CALIB_MIN_FRAMES or not lo < rate < hi:
        raise SystemExit(f"calibration stream degenerate: {len(z)} frames, "
                         f"event rate {rate:.3f}")
    return z, y, v


def fit_platt(z, y):
    X = np.stack([z, np.ones_like(z)], axis=1)

    def bce(w):
        u = X @ w
        return float(np.mean(np.logaddexp(0.0, u) - y * u))

    def grad(w):
        return X.T @ (expit(X @ w) - y) / len(y)

    res = minimize(bce, np.array([1.0, 0.0]), jac=grad, method="BFGS")
    if not res.success:
        raise SystemExit(f"Platt fit did not converge: {res.message}")
    return float(res.x[0]), float(res.x[1])


def calibrate_confidence(soup, corpora, pools, rooms, device):
    net = SwiftF0Net()
    net.load_state_dict(soup)
    net = net.eval().to(device)
    z, y, v = calib_stream(corpora, pools, rooms, net, device)
    a, b = fit_platt(z, y)
    if a <= 0:
        raise SystemExit(f"calibration fit inverted the confidence (a={a:.4f})")
    cal = expit(a * z + b)
    ths = np.linspace(0.0, 1.0, CALIB_GRID_POINTS)
    order, pos = np.sort(cal), np.sort(cal[y > 0])
    n_pred = len(order) - np.searchsorted(order, ths, side="left")
    hits = len(pos) - np.searchsorted(pos, ths, side="left")
    f = 2.0 * hits / (n_pred + (v > 0).sum())
    star = int(np.argmax(f))
    if star in (0, CALIB_GRID_POINTS - 1):
        raise SystemExit(f"calibration selected a degenerate operating point "
                         f"theta={ths[star]:g}")
    shift = math.log(ths[star] / (1.0 - ths[star]))
    last = len(net.correctness_head) - 1
    soup[f"correctness_head.{last}.weight"] = soup[f"correctness_head.{last}.weight"] * a
    soup[f"correctness_head.{last}.bias"] = (soup[f"correctness_head.{last}.bias"] * a
                                             + (b - shift))
    return soup


def resolve_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_loop(model, loader, sampler, step, opt, sched, epochs, out_dir, ctx, device,
               swa_start):
    with open(os.path.join(out_dir, "train_log.jsonl"), "w") as log:
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            sampler.set_epoch(epoch)
            model.train()
            agg, nb = {}, 0
            for aug in rendered_batches(loader, ctx, device):
                opt.zero_grad()
                loss, m = step(aug)
                if not math.isfinite(m["loss"]):
                    raise SystemExit(f"loss went non-finite at epoch {epoch}, "
                                     f"step {nb + 1}")
                loss.backward()
                opt.step()
                sched.step()
                nb += 1
                for k, v in m.items():
                    if not math.isnan(v):
                        agg.setdefault(k, []).append(v)
            if epoch >= swa_start:
                torch.save({"state_dict": {k: v.detach().cpu().clone()
                                           for k, v in model.state_dict().items()}},
                           os.path.join(out_dir, "epochs", f"epoch_{epoch:04d}.pt"))
            row = {k: round(sum(v) / len(v), 4) for k, v in agg.items()}
            row.update(epoch=epoch, total=EPOCHS, seconds=round(time.time() - t0, 1))
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"[epoch {epoch}/{epochs}] {row}", flush=True)


def finalize(corpora, pools, rooms, paths, workers, ctx, device):
    soup = swa_average(paths)
    soup = recalibrate_bn(soup, corpora, pools, rooms, workers, ctx, device)
    soup = calibrate_confidence(soup, corpora, pools, rooms, device)
    return {k: v.detach().cpu().clone() for k, v in soup.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Path to the prepared train/ split")
    parser.add_argument("--out", default=os.path.join(OUT, "local"))
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--stop-after", type=int, default=EPOCHS,
                        help="train this many epochs of the full schedule and skip finalization")
    args = parser.parse_args()
    if not os.path.isfile(os.path.join(args.dataset, "augmentation", "pools.json")):
        parser.error(f"{args.dataset} is not a prepared train/ split (no augmentation/pools.json)")
    if not 1 <= args.stop_after <= EPOCHS:
        parser.error(f"--stop-after must be between 1 and {EPOCHS}")
    device = resolve_device(DEVICE)
    ctx = int(CTX_SECONDS * SAMPLE_RATE) - int(MARGIN_SECONDS * SAMPLE_RATE)
    out_dir = args.out

    corpora = load_corpora(args.dataset)
    pools, rooms = load_pools(os.path.join(args.dataset, "augmentation"))
    os.makedirs(os.path.join(out_dir, "epochs"), exist_ok=True)

    torch.manual_seed(SEED)
    model = SwiftF0Net().to(device)
    centers = model.pitch_bin_centers
    print(f"model     {sum(p.numel() for p in model.parameters())} params, "
          f"{model.n_log} rows,\n          {model.eff_bpo:.3f} bins/octave, "
          f"{float(centers[0]):.1f}-{float(centers[-1]):.1f} Hz", flush=True)
    print(f"device    {device}, seed {SEED}, {EPOCHS} epochs", flush=True)
    print(f"out       {out_dir}", flush=True)

    loader, sampler = build_loader(SEED, args.workers, corpora, pools, rooms)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = WarmupScheduler(opt, WARMUP_EPOCHS, EPOCHS, len(loader), MIN_LR_FACTOR, LR_DECAY_FRAC)
    post = EPOCHS - WARMUP_EPOCHS
    decay_start = WARMUP_EPOCHS + post * (1.0 - LR_DECAY_FRAC)
    swa_start = EPOCHS - max(1, round(EPOCHS * SWA_TAIL)) + 1
    if post > 0 and swa_start < decay_start:
        need = 1.0 - (swa_start - WARMUP_EPOCHS) / post
        raise SystemExit(
            f"the LR decay starts at epoch {decay_start:.1f} but SWA_TAIL "
            f"{SWA_TAIL} averages from epoch {swa_start}; set SWA_TAIL to "
            f"{LR_DECAY_FRAC} or raise LR_DECAY_FRAC to at least {need:.4f}")

    train_loop(model, loader, sampler, make_step(model, device), opt, sched, args.stop_after,
               out_dir, ctx, device, swa_start)
    if args.stop_after < EPOCHS:
        return

    paths = [os.path.join(out_dir, "epochs", f"epoch_{e:04d}.pt")
             for e in range(swa_start, EPOCHS + 1)]
    soup = finalize(corpora, pools, rooms, paths, args.workers, ctx, device)
    path = os.path.join(out_dir, "weights.pt")
    torch.save(soup, path)
    print(f"wrote {path} from epochs {swa_start}-{EPOCHS}", flush=True)


if __name__ == "__main__":
    main()
