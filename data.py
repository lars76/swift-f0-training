import bisect
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

from augment import SALT_CHUNK, SALT_PERM, SALT_SHUFFLE, draw_augmentation, item_seed
from config import (
    BATCH_SIZE,
    CHUNK_SECONDS,
    CORPORA,
    CTX_SECONDS,
    HOP_SIZE,
    MARGIN_SECONDS,
    SAMPLE_RATE,
    SOURCE_SLOTS,
)
from model import MAX_HZ, MIN_HZ


class AudioDataset:
    def __init__(self, path):
        self.dir = Path(path)
        self.indices = sorted(int(p.stem) for p in (self.dir / "audio").glob("*.flac"))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        with np.load(self.dir / "labels" / f"{index:07d}.npz", allow_pickle=False) as labels:
            sample = {k: torch.from_numpy(labels[k].copy()) for k in labels.files}
        pitch = sample["pitch"]
        voiced = pitch > 0
        unknown = voiced & (~sample.pop("trusted") | (pitch < MIN_HZ) | (pitch > MAX_HZ))
        sample["pitch"] = torch.where(unknown, torch.nan, pitch)
        sample["voiced"] = voiced.to(torch.float32)
        return sample

    def audio(self, idx, start, frames):
        path = self.dir / "audio" / f"{self.indices[idx]:07d}.flac"
        audio, _ = sf.read(str(path), dtype="float32", start=start, frames=frames)
        return audio


def any_voiced(pitch):
    return bool(np.any(pitch > 0))


def attach_augmentation(pools, rooms, seed, key, sample, has_voice):
    sample.update(draw_augmentation(pools, rooms, seed, key, len(sample["audio"]), has_voice))
    return sample


class ChunkDataset(Dataset):
    def __init__(self, datasets, pools, rooms, seed):
        self.datasets = datasets
        self.pools, self.rooms = pools, rooms
        self.seed = int(seed)
        self.hop = HOP_SIZE
        self.chunk_samples = int(CHUNK_SECONDS * SAMPLE_RATE)
        self.ctx_samples = int(CTX_SECONDS * SAMPLE_RATE)
        self.margin_samples = int(MARGIN_SECONDS * SAMPLE_RATE)
        self.expected_frames = self.chunk_samples // self.hop
        self.margin_frames = self.margin_samples // self.hop
        self.offsets, acc = [], 0
        for ds in datasets:
            self.offsets.append(acc)
            acc += len(ds)
        self.total_files = acc

    def __len__(self):
        return self.total_files

    def __getitem__(self, key):
        idx, draw = int(key[0]), int(key[1])
        rng = random.Random(item_seed(self.seed, SALT_CHUNK, draw))
        ds_idx = bisect.bisect_right(self.offsets, idx) - 1
        sample_idx = idx - self.offsets[ds_idx]
        sample = self.datasets[ds_idx][sample_idx]
        total_frames = sample["pitch"].size(-1)
        start_frame = (0 if total_frames <= self.expected_frames
                       else rng.randint(0, total_frames - self.expected_frames))
        start_sample = start_frame * self.hop
        end_frame = start_frame + self.expected_frames
        audio = np.zeros(self.ctx_samples + self.chunk_samples + self.margin_samples,
                         dtype=np.float32)
        lo = max(0, start_sample - self.ctx_samples)
        left = start_sample - lo
        got = self.datasets[ds_idx].audio(sample_idx, lo,
                                          left + self.chunk_samples + self.margin_samples)
        audio[self.ctx_samples - left:self.ctx_samples - left + len(got)] = got
        m = self.margin_frames
        frames = self.expected_frames + 2 * m

        def crop_pad(t, pad):
            t = t.reshape(-1).numpy().astype(np.float32)
            out = np.full(frames, pad, dtype=np.float32)
            a, b = max(start_frame - m, 0), min(end_frame + m, len(t))
            out[a - (start_frame - m):a - (start_frame - m) + (b - a)] = t[a:b]
            return out

        pitch = np.full(frames, np.nan, dtype=np.float32)
        known = sample["pitch"].reshape(-1).numpy().astype(np.float32)[start_frame:end_frame]
        pitch[m:m + len(known)] = known
        voiced = crop_pad(sample["voiced"], 0.0)
        out = {"audio": audio, "pitch": pitch, "periodicity": voiced}
        has_voice = any_voiced(voiced)
        return attach_augmentation(self.pools, self.rooms, self.seed, (draw,), out,
                                   has_voice)


class CorpusMixSampler(torch.utils.data.Sampler):
    def __init__(self, lengths, weights, seed):
        self.lengths = [int(n) for n in lengths]
        self.offsets, acc = [], 0
        for n in self.lengths:
            self.offsets.append(acc)
            acc += n
        self.seed = int(seed)
        self.epoch = 0
        s = float(sum(weights))
        self.quota = [round(w / s * acc) for w in weights]

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return int(sum(self.quota))

    def _rng(self, *parts):
        return random.Random(item_seed(self.seed, *parts))

    def _perm(self, d, p):
        order = list(range(self.lengths[d]))
        self._rng(SALT_PERM, d, p).shuffle(order)
        return order

    def _epoch_indices(self, d):
        n, q = self.lengths[d], self.quota[d]
        if n == 0 or q == 0:
            return []
        out = []
        start = self.epoch * q
        p, pos = start // n, start % n
        perm = self._perm(d, p)
        for _ in range(q):
            if pos == n:
                p += 1
                pos = 0
                perm = self._perm(d, p)
            out.append(self.offsets[d] + perm[pos])
            pos += 1
        return out

    def __iter__(self):
        idx = []
        for d in range(len(self.lengths)):
            idx.extend(self._epoch_indices(d))
        self._rng(SALT_SHUFFLE, self.epoch).shuffle(idx)
        base = self.epoch * len(self)
        return iter([(g, base + p) for p, g in enumerate(idx)])


def collate(batch):
    out = {}
    lmax = max(max(len(h) for h in s["rir"]) for s in batch)
    rir = np.zeros((len(batch), SOURCE_SLOTS + 1, lmax), dtype=np.float32)
    for b, s in enumerate(batch):
        for j, h in enumerate(s["rir"]):
            rir[b, j, :len(h)] = h
    out["rir"] = torch.from_numpy(rir)
    for k in batch[0]:
        if k == "rir":
            continue
        out[k] = torch.stack([torch.as_tensor(s[k]) for s in batch])
    return out


def load_corpora(root):
    corpora = [(AudioDataset(Path(root) / name), float(weight)) for name, weight in CORPORA]
    for dataset, _weight in corpora:
        if not len(dataset):
            raise SystemExit(f"no FLAC files under {dataset.dir / 'audio'}")
    return corpora


def build_loader(seed, workers, corpora, pools, rooms):
    datasets = [d for d, _w in corpora]
    aug = ChunkDataset(datasets, pools, rooms, seed)
    sampler = CorpusMixSampler([len(d) for d in datasets], [w for _d, w in corpora], seed=seed)
    kw = {"num_workers": workers}
    if workers > 0:
        kw.update(persistent_workers=True, prefetch_factor=6)
    loader = DataLoader(aug, batch_size=BATCH_SIZE, sampler=sampler,
                        collate_fn=collate, pin_memory=torch.cuda.is_available(),
                        generator=torch.Generator().manual_seed(seed), **kw)
    return loader, sampler
