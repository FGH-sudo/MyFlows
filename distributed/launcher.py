"""Bounded spawn launcher, event monitor and cleanup of this run's processes."""

import multiprocessing as mp
import queue
import time
import uuid

import numpy as np

from .ps import server_main
from .worker import worker_main


def run_ps(*, workers=2, global_batch=32, steps=20, seed=0, shard_sizes=None,
           timeout=10.0, fault=None, fault_step=0, run_id=None):
    if type(workers) is not int or workers <= 0 or type(steps) is not int or steps <= 0:
        raise ValueError("workers and steps must be positive integers")
    if type(global_batch) is not int or global_batch < workers or not np.isfinite(timeout) or timeout <= 0:
        raise ValueError("batch must give every worker a sample; timeout must be positive")
    if fault not in (None, "crash", "timeout", "duplicate", "schema"):
        raise ValueError("unknown fault injection")
    if not 0 <= fault_step < steps:
        raise ValueError("fault_step must be within the training run")
    if shard_sizes is None:
        shard_sizes = [len(a) for a in np.array_split(np.arange(global_batch), workers)]
    if len(shard_sizes) != workers or any(type(n) is not int or n <= 0 for n in shard_sizes) or sum(shard_sizes) != global_batch:
        raise ValueError("shard_sizes must be positive and sum to global_batch")
    config = dict(workers=workers, global_batch=global_batch, steps=steps, seed=seed,
                  shard_sizes=list(shard_sizes), timeout=timeout, fault=fault, fault_step=fault_step,
                  run_id=run_id or uuid.uuid4().hex)
    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    incoming, events, results = ctx.Queue(maxsize=workers * 2), ctx.Queue(maxsize=256), ctx.Queue(maxsize=1)
    replies = [ctx.Queue(maxsize=2) for _ in range(workers)]
    server = ctx.Process(target=server_main, name="myflows-ps", args=(config, incoming, replies, events, results, stop))
    processes = [server]
    offset = 0
    for worker, size in enumerate(shard_sizes):
        shard = np.arange(offset, offset + size)
        processes.append(ctx.Process(target=worker_main, name=f"myflows-worker-{worker}",
                                     args=(worker, config, shard, replies[worker], incoming, events, stop)))
        offset += size
    collected, result = [], None
    started = time.perf_counter()
    try:
        for process in processes:
            process.start()
        deadline = started + timeout * (steps + 3) + 15
        while True:
            try:
                collected.append(events.get(timeout=0.02))
            except queue.Empty:
                pass
            if result is None:
                try:
                    result = results.get_nowait()
                except queue.Empty:
                    pass
            failed = [p for p in processes if p.exitcode not in (None, 0)]
            if failed:
                if result is None or result["status"] == "passed":
                    result = {"status": "failed", "error": "; ".join(f"{p.name} exited {p.exitcode}" for p in failed),
                              "failed_step": max((e["step_id"] for e in collected), default=-1)}
                break
            if result is not None and result["status"] == "failed":
                break
            if all(p.exitcode is not None for p in processes):
                if result is None:
                    try:
                        result = results.get(timeout=0.2)
                    except queue.Empty:
                        result = {"status": "failed", "error": "server exited without a result"}
                break
            if time.perf_counter() > deadline:
                result = {"status": "failed", "error": "launcher deadline exceeded"}
                break
    finally:
        cleanup_start = time.perf_counter()
        stop.set()
        for process in processes:
            if process.pid is not None:
                process.join(timeout=max(0, min(0.2, cleanup_start + 1 - time.perf_counter())))
        forced = False
        for process in processes:
            if process.pid is not None and process.is_alive():
                forced = True
                process.terminate()
        for process in processes:
            if process.pid is not None:
                process.join(timeout=max(0, cleanup_start + 4 - time.perf_counter()))
                if process.is_alive():
                    process.kill()
                    process.join(timeout=max(0, cleanup_start + 5 - time.perf_counter()))
        # A forcibly terminated producer can leave a Queue pipe mid-message.
        # Do not read it after termination; the monitor already retained events.
        while not forced:
            try:
                collected.append(events.get_nowait())
            except queue.Empty:
                break
        alive = [p.pid for p in processes if p.pid is not None and p.is_alive()]
        exitcodes = {p.name: p.exitcode for p in processes}
        for channel in [incoming, events, results, *replies]:
            channel.cancel_join_thread()
            channel.close()
        cleanup_s = time.perf_counter() - cleanup_start
    result = result or {"status": "failed", "error": "run aborted"}
    if alive or cleanup_s > 5:
        result.update(status="failed", error="process cleanup did not meet deadline")
    result.update(config=config, events=collected, exitcodes=exitcodes, alive_pids=alive,
                  cleanup_s=cleanup_s, elapsed_s=time.perf_counter() - started)
    return result
