"""Content-addressed pre-head features for deterministic, fully frozen backbones."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np


def _digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


class FeatureCache:
    def __init__(self, root):
        self.root = Path(root)

    @staticmethod
    def key(weights_sha256, preprocessing, sample_manifest):
        payload = {"version": 1, "weights": weights_sha256, "preprocessing": preprocessing, "samples": sample_manifest}
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _open(self, directory, key, sizes, width):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["key"] != key or set(manifest["splits"]) != set(sizes):
            raise ValueError("feature cache manifest differs from the request")
        arrays = {}
        for name, size in sizes.items():
            path = directory / f"{name}.npy"
            entry = manifest["splits"][name]
            if _digest(path) != entry["sha256"]:
                raise ValueError(f"{name} feature cache checksum mismatch")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.shape != (size, width) or array.dtype != np.float32:
                raise ValueError(f"{name} feature cache shape or dtype mismatch")
            arrays[name] = array
        return arrays, manifest

    def materialize(self, splits, *, sample_manifest, weights_sha256, preprocessing, batch, width, encode):
        """encode takes one list of real samples, pads if necessary, and returns exactly those N rows."""
        if batch < 1 or width < 1 or not splits or set(splits) - {"train", "val", "test"}:
            raise ValueError("expected train/val/test splits, positive batch and feature width")
        sizes = {name: len(values) for name, values in splits.items()}
        if sizes != {name: len(values) for name, values in sample_manifest.items()}:
            raise ValueError("sample manifest does not cover the requested splits")
        key = self.key(weights_sha256, preprocessing, sample_manifest)
        final = self.root / key
        if final.is_dir():
            arrays, manifest = self._open(final, key, sizes, width)
            return arrays, {**manifest, "hit": True}
        self.root.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{key[:12]}-", dir=self.root))
        started = time.perf_counter()
        try:
            entries = {}
            for name, samples in splits.items():
                path = staging / f"{name}.npy"
                features = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(len(samples), width))
                try:
                    for first in range(0, len(samples), batch):
                        selected = samples[first:first+batch]
                        values = np.asarray(encode(selected))
                        if values.shape != (len(selected), width) or values.dtype != np.float32 or not np.isfinite(values).all():
                            raise ValueError("encoder returned invalid FP32 feature rows")
                        features[first:first+len(selected)] = values
                    features.flush()
                finally:
                    del features
                entries[name] = {"samples": len(samples), "width": width, "sha256": _digest(path)}
            manifest = {"schema_version": 1, "key": key, "weights_sha256": weights_sha256, "preprocessing": preprocessing,
                        "splits": entries, "extraction_s": time.perf_counter() - started}
            (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            try:
                os.rename(staging, final)
            except OSError:
                if not final.is_dir():
                    raise
            arrays, published = self._open(final, key, sizes, width)
            return arrays, {**published, "hit": False}
        finally:
            if staging.is_dir():
                if staging.resolve().parent != self.root.resolve():
                    raise RuntimeError("temporary cache directory escaped its root")
                shutil.rmtree(staging)
