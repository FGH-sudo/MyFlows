"""Event helper shared by PS, Ring and launcher monitor."""

from __future__ import annotations

import time


def emit(channel, config, role, worker_id, step, phase, **fields):
    if channel is None:
        return
    payload = {
        "run_id": config.get("run_id"),
        "role": role,
        "worker_id": worker_id,
        "step_id": step,
        "phase": phase,
        "monotonic_s": time.perf_counter(),
        "mode": config.get("mode"),
        "transport": config.get("transport"),
    }
    payload.update(fields)
    try:
        channel.put(payload, timeout=float(config.get("timeout", 30)))
    except Exception:
        pass
