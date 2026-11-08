"""Empirical, per-layout register GEMM plans. Tuning never changes the process-wide CUDA math mode."""

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import time

import numpy as np

from ...core.device import _lazy_cupy
from . import build, native

VERSION = 1


@lru_cache(maxsize=16)
def device_key(device):
    cp = _lazy_cupy()
    props = cp.cuda.runtime.getDeviceProperties(device)
    uuid = props.get("uuid")
    identity = {
        "uuid": bytes(uuid).hex() if uuid else None,
        "pci": cp.cuda.Device(device).pci_bus_id,
        "name": props["name"].decode(),
        "cc": [props["major"], props["minor"]],
        "sms": props["multiProcessorCount"],
        "driver": cp.cuda.runtime.driverGetVersion(),
        "runtime": cp.cuda.runtime.runtimeGetVersion(),
        "native_sha256": build.source_digest(),
        "protocol": VERSION,
    }
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return digest, identity


def _path(directory, device, layout, m, n, k):
    key, _ = device_key(device)
    root = Path(directory) if directory else build.OUTPUT.parent / "gemm-plans"
    return root.resolve() / key / f"{layout}-{m}x{n}x{k}.json"


def cache_file(layout, m, n, k):
    return _path(
        os.environ.get("MYFLOWS_AUTOTUNE_CACHE"), _lazy_cupy().cuda.runtime.getDevice(), layout, m, n, k
    )


def _read(path, layout, shape, identity):
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["identity"] != identity or row["layout"] != layout or row["shape"] != list(shape):
            return None
        chosen = row["selected"]
        tile, requested = chosen["tile"], chosen["requested_slices"]
        native._plan_code(tile, requested)
        if not any(
            c["tile"] == tile and c["requested_slices"] == requested and c.get("times_ms")
            for c in row["candidates"]
        ):
            return None
        return tile, requested
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _calibrate(layout, shape, identity):
    cp = _lazy_cupy()
    m, n, k = shape
    ta, tb = layout == "TN", layout == "NT"
    rng = cp.random.RandomState(197)
    a = rng.standard_normal((k, m) if ta else (m, k), dtype=cp.float32)
    b = rng.standard_normal((n, k) if tb else (k, n), dtype=cp.float32)
    out = cp.empty((m, n), dtype=cp.float32)
    ia = cp.asarray(np.linspace(0, m - 1, min(m, 8), dtype=int))
    ib = cp.asarray(np.linspace(0, n - 1, min(n, 8), dtype=int))
    exact = cp.asnumpy((a.T if ta else a)[ia]).astype(np.float64) @ cp.asnumpy(
        (b.T if tb else b)[:, ib]
    ).astype(np.float64)
    manual = native.gemm_plan(m, n, k)
    sample = lambda: cp.asnumpy(out[ia[:, None], ib[None, :]])

    def launch(tile, requested):
        return native.gemm(a, b, trans_a=ta, trans_b=tb, out=out, plan=(tile, requested))

    launch(native.GEMM_TILES[0], 1)
    unsplit = sample()
    free, total = cp.cuda.runtime.memGetInfo()
    workspace_budget = max(0, int(free) - max(512 * 2**20, int(total) // 8))
    seen, rows = set(), []
    for requested in range(1, 33):
        slices = native._slices(k, requested)
        if slices in seen:
            continue
        seen.add(slices)
        bytes_needed = 4 * m * n * slices if slices > 1 else 0
        for tile in native.GEMM_TILES:
            rows.append(
                {
                    "tile": tile,
                    "requested_slices": requested,
                    "slices": slices,
                    "workspace_bytes": bytes_needed,
                    "times_ms": [],
                    "skip": "workspace exceeds available dedicated budget"
                    if bytes_needed > workspace_budget
                    else None,
                }
            )
    order = [row for row in rows if not row["skip"]]
    events = cp.cuda.Event(), cp.cuda.Event()
    shuffler = random.Random(2201)
    started = time.monotonic()
    repetitions = 3
    for round_index in range(3):
        shuffler.shuffle(order)
        for row in order:
            if row["skip"]:
                continue
            tile, requested = row["tile"], row["requested_slices"]
            try:
                launch(tile, requested)
                if round_index == 0:
                    got = sample()
                    if row["slices"] == 1 and got.tobytes() != unsplit.tobytes():
                        raise ArithmeticError(
                            f"unsplit tile changed the accumulation order: {tile} {layout} {shape}"
                        )
                    row["sample_max_abs_vs_unsplit"] = float(np.max(np.abs(got - unsplit)))
                    row["sample_relative_vs_fp64"] = float(
                        np.max(np.abs(got - exact)) / max(np.max(np.abs(exact)), 1e-30)
                    )
                events[0].record()
                for _ in range(repetitions):
                    launch(tile, requested)
                events[1].record()
                events[1].synchronize()
                row["times_ms"].append(float(cp.cuda.get_elapsed_time(*events)) / repetitions)
            except cp.cuda.memory.OutOfMemoryError:
                row["skip"] = "allocation failed"
    valid = [r for r in rows if not r["skip"] and len(r["times_ms"]) == 3]
    if not valid:
        raise RuntimeError("no register GEMM plan could be measured")
    for row in valid:
        row["median_ms"] = statistics.median(row["times_ms"])
    winner = min(valid, key=lambda r: r["median_ms"])
    return {
        "identity": identity,
        "layout": layout,
        "shape": list(shape),
        "candidates": rows,
        "manual": {"tile": manual[0], "slices": manual[1]},
        "selected": winner,
        "seconds": time.monotonic() - started,
        "workspace_budget_bytes": workspace_budget,
        "repeats": repetitions,
    }


@lru_cache(maxsize=4096)
def _selected(directory, device, layout, m, n, k):
    path = _path(directory, device, layout, m, n, k)
    _, identity = device_key(device)
    shape = (m, n, k)
    existing = _read(path, layout, shape, identity)
    if existing:
        return existing
    with build._exclusive(path.parent / "measure.lock", timeout_s=1800):
        existing = _read(path, layout, shape, identity)
        if existing:
            return existing
        cp = _lazy_cupy()
        pool = cp.cuda.MemoryPool()
        with cp.cuda.using_allocator(pool.malloc):
            result = _calibrate(layout, shape, identity)
        pool.free_all_blocks()
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(result, allow_nan=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        win = result["selected"]
        return win["tile"], win["requested_slices"]


def selected_plan(layout, m, n, k):
    if layout not in ("NN", "NT", "TN") or min(m, n, k) < 1:
        raise ValueError("expected NN, NT or TN and positive GEMM dimensions")
    device = _lazy_cupy().cuda.runtime.getDevice()
    return _selected(os.environ.get("MYFLOWS_AUTOTUNE_CACHE"), device, layout, m, n, k)
