import json
import os

import numpy as np
import soundfile as sf


class Pool:
    def __init__(self, entry, pools_root):
        self.name = entry["name"]
        self.kind = entry.get("kind", "")
        self.point = bool(entry.get("point", False))
        self.label_safe = bool(entry.get("label_safe", False))
        self.dir = os.path.join(pools_root, entry["name"])
        self.path = os.path.join(self.dir, "pool.flac")
        self.index = {}
        self.offsets = np.zeros(1, dtype=np.int64)
        if self.kind == "colored":
            return
        with open(os.path.join(self.dir, "index.json")) as fh:
            self.index = json.load(fh)
        self.offsets = np.asarray(self.index["offsets"], dtype=np.int64)
        frames = sf.info(self.path).frames
        if int(self.offsets[-1]) != frames:
            raise SystemExit(
                f"{self.name}: index.json ends at sample {int(self.offsets[-1])} but "
                f"pool.flac holds {frames}; the offsets do not describe this file and "
                f"every window would be cut from the wrong clip")

    def __len__(self):
        return len(self.offsets) - 1

    def _read(self, i, start, frames):
        lo, hi = int(self.offsets[i]), int(self.offsets[i + 1])
        if start < 0 or frames < 0 or lo + start + frames > hi:
            raise SystemExit(
                f"{self.name}: window [{start}, {start + frames}) runs past clip {i} "
                f"({hi - lo} samples); a window crossing a clip boundary would splice two "
                f"independently normalized sources into one segment")
        audio, _ = sf.read(self.path, start=lo + int(start), frames=int(frames),
                           dtype="float32")
        return audio

    def draw_window(self, n, rng):
        i = int(rng.integers(len(self)))
        m = int(self.offsets[i + 1] - self.offsets[i])
        if m >= n:
            s = int(rng.integers(0, m - n + 1))
            return self._read(i, s, n)
        x = self._read(i, 0, m)
        reps = (n + m - 1) // m
        return np.tile(x, reps)[:n]


class RoomPool(Pool):
    def __init__(self, entry, pools_root):
        super().__init__(entry, pools_root)
        self.p = self.index["p"]

    def draw(self, rng):
        i = int(rng.integers(len(self)))
        return self._read(i, 0, int(self.offsets[i + 1] - self.offsets[i])), int(self.p[i])


def load_pools(pools_root):
    with open(os.path.join(pools_root, "pools.json")) as fh:
        declared = json.load(fh)
    return ([Pool(e, pools_root) for e in declared["pools"]],
            [RoomPool(e, pools_root) for e in declared["rooms"]])
