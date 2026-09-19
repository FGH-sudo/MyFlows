"""Spawn PS or Ring ranks, monitor events and clean up this run's processes."""

from __future__ import annotations

import multiprocessing as mp
import queue
import socket
import time
import uuid

import numpy as np

from .constants import BIND_HOST, DEFAULT_CLEANUP_S, KNOWN_TASKS, LEGAL_MODES
from .data_meta import merge_data_meta
from .metrics_reduce import reduce_step_metrics
from .ps import server_main
from .schedule import resolve_budget, run_local_training
from .shards import split_global_batch
from .worker import shard_plan, worker_main


def allocate_port(host=BIND_HOST):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, 0))
    port = sock.getsockname()[1]
    sock.close()
    return int(port)


def normalize_config(**kwargs):
    config = dict(kwargs)
    config.setdefault("mode", "ps")
    if config["mode"] == "single":
        config["transport"] = "none"
        config["train_workers"] = 1
        config["workers"] = 1
    else:
        config.setdefault("transport", "socket_json")
    config.setdefault("task", "synthetic")
    config.setdefault("device", "cpu")
    config.setdefault("optimizer", "mbgd")
    config.setdefault("learning_rate", 0.01)
    config.setdefault("seed", 0)
    config.setdefault("steps", 20)
    config.setdefault("epochs", 1)
    config.setdefault("global_batch", 32)
    config.setdefault("timeout", 10.0)
    config.setdefault("fault", None)
    config.setdefault("fault_step", 0)
    config.setdefault("collect_history", True)
    config.setdefault("bind_host", BIND_HOST)
    config["train_workers"] = int(config.get("train_workers") or config.get("workers") or 2)
    config["workers"] = config["train_workers"]
    if str(config["task"]) not in KNOWN_TASKS:
        raise ValueError(f"unknown task {config['task']}")
    if (config["mode"], config["transport"]) not in LEGAL_MODES:
        raise ValueError(f"illegal mode/transport {config['mode']}/{config['transport']}")
    workers = config["train_workers"]
    steps = int(config["steps"])
    epochs = int(config["epochs"])
    if workers <= 0 or steps <= 0 or epochs <= 0:
        raise ValueError("workers, steps and epochs must be positive integers")
    if int(config["global_batch"]) < workers or not np.isfinite(config["timeout"]) or config["timeout"] <= 0:
        raise ValueError("batch must give every worker a sample; timeout must be positive")
    if config["fault"] not in (None, "crash", "timeout", "duplicate", "retry", "schema", "heartbeat_loss", "gather_crash"):
        raise ValueError("unknown fault injection")
    budget = resolve_budget(config)
    config["steps_semantics"] = budget["steps_semantics"]
    config["planned_steps"] = int(budget["total_steps"])
    if not 0 <= int(config["fault_step"]) < max(int(config["planned_steps"]), 1):
        raise ValueError("fault_step must be within the training run")
    config["shard_sizes"] = split_global_batch(
        int(config["global_batch"]), workers, config.get("shard_sizes"))
    config["run_id"] = config.get("run_id") or uuid.uuid4().hex
    return config


def run_ps(**kwargs):
    kwargs.setdefault("mode", "ps")
    kwargs.setdefault("transport", kwargs.get("transport", "socket_json"))
    return run_training(**kwargs)


def run_training(**kwargs):
    config = normalize_config(**kwargs)
    if config["mode"] == "single":
        result = run_local_training(config)
        result.update(config=config, events=result.get("events") or [])
        return result
    if config["mode"] == "ring":
        from .ring import run_ring
        return run_ring(config)
    return _run_ps(config)


def _run_ps(config):
    config = dict(config)
    config["ps_port"] = config.get("ps_port") or allocate_port(config["bind_host"])
    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    events, results = ctx.Queue(maxsize=512), ctx.Queue(maxsize=1)
    server = ctx.Process(target=server_main, name="myflows-ps", args=(config, events, results, stop))
    processes = [server]
    _sizes, shards = shard_plan(config)
    for worker, shard in enumerate(shards):
        processes.append(ctx.Process(
            target=worker_main, name=f"myflows-worker-{worker}",
            args=(worker, config, shard, events, stop)))
    collected, result = [], None
    started = time.perf_counter()
    workers = processes[1:]
    planned = int(config.get("planned_steps") or config["steps"])
    try:
        for process in processes:
            process.start()
        deadline = started + float(config["timeout"]) * (planned + 8) + 15
        while True:
            for process in processes:
                process.join(timeout=0)
            try:
                collected.append(events.get(timeout=0.02))
            except queue.Empty:
                pass
            failed = [p for p in processes if p.exitcode not in (None, 0)]
            if failed:
                result = {"status": "failed",
                          "error": "; ".join(f"{p.name} exited {p.exitcode}" for p in failed),
                          "failed_step": max((e.get("step_id", -1) for e in collected), default=-1)}
                break
            if workers and all(p.exitcode == 0 for p in workers):
                result = {"status": "passed"}
                break
            if all(p.exitcode is not None for p in processes):
                result = result or {"status": "failed", "error": "processes exited before workers finished"}
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
        while not forced:
            try:
                collected.append(events.get_nowait())
            except queue.Empty:
                break
        alive = [p.pid for p in processes if p.pid is not None and p.is_alive()]
        exitcodes = {p.name: p.exitcode for p in processes}
        for channel in (events, results):
            channel.cancel_join_thread()
            channel.close()
        cleanup_s = time.perf_counter() - cleanup_start
    result = result or {"status": "failed", "error": "run aborted"}
    if alive or cleanup_s > DEFAULT_CLEANUP_S:
        result.update(status="failed", error="process cleanup did not meet deadline")
    result.update(config=config, events=collected, exitcodes=exitcodes, alive_pids=alive,
                  cleanup_s=cleanup_s, elapsed_s=time.perf_counter() - started)
    result.update(_outputs_from_events(collected, config["train_workers"]))
    if result.get("metrics_issues") and result.get("status") == "passed":
        result.update(status="failed", error="; ".join(result["metrics_issues"]))
    result["communication"] = True
    return result


def _outputs_from_events(events, n_workers):
    histories = {}
    losses = {}
    metrics = {}
    cpu_mat = 0
    builds = 0
    communication = False
    metric_events = {}
    data_metas = {}
    cache_stats = None
    for event in events:
        phase = event.get("phase")
        worker = event.get("worker_id")
        if phase == "history" and worker is not None:
            worker = int(worker)
            histories[worker] = event.get("history") or []
            losses[worker] = event.get("losses") or []
            metrics[worker] = event.get("metrics") or []
        elif phase == "metrics" and worker is not None:
            metric_events.setdefault(int(worker), []).append(event)
        elif phase == "data_meta" and worker is not None:
            data_metas[int(worker)] = event
        elif phase == "done":
            cpu_mat = max(cpu_mat, int(event.get("cpu_grad_materializations") or 0))
            builds = max(builds, int(event.get("epoch_index_builds") or 0))
            if event.get("communication"):
                communication = True
        elif phase == "session_ready" and event.get("communication"):
            communication = True
        elif phase == "cache_stats":
            cache_stats = {
                "cache_sizes": event.get("cache_sizes"),
                "cache_footprint_bytes": event.get("cache_footprint_bytes"),
                "retained_payload_bytes": event.get("retained_payload_bytes"),
            }
    if not losses:
        losses = {worker: [float(item.get("loss")) for item in rows]
                  for worker, rows in metric_events.items()}
    if not metrics:
        metrics = {
            worker: [_metric_from_event(item) for item in rows]
            for worker, rows in metric_events.items()
        }
    history = histories.get(0) or (next(iter(histories.values())) if histories else [])
    if len(histories) == n_workers and histories:
        reference = histories[0]
        for worker, actual in histories.items():
            if len(actual) != len(reference):
                raise AssertionError(f"worker {worker} history length differs")
    global_rows = []
    issues = []
    by_step = {}
    for worker, rows in metrics.items():
        for row in rows:
            item = dict(row)
            item.setdefault("worker_id", worker)
            by_step.setdefault(int(item.get("global_step", 0)), []).append(item)
    for step in sorted(by_step):
        merged, step_issues = reduce_step_metrics(by_step[step], n_workers)
        if step_issues:
            issues.extend(f"step {step}: {item}" for item in step_issues)
        elif merged is not None:
            global_rows.append(merged)
    data_meta = None
    try:
        ordered = [data_metas[index] for index in range(int(n_workers))]
        data_meta = merge_data_meta(ordered, n_workers)
    except Exception as exc:  # noqa: BLE001
        issues.append(f"data_meta: {exc}")
        data_meta = None
    if issues:
        global_rows = []
    return {
        "history": history,
        "losses": [float(row["loss"]) for row in global_rows if row.get("loss") is not None],
        "metrics": global_rows,
        "worker_metrics": metrics,
        "metrics_scope": "train_batch",
        "metrics_valid": not issues,
        "metrics_issues": issues,
        "data_meta": data_meta,
        "cache_stats": cache_stats,
        "cpu_grad_materializations": cpu_mat,
        "epoch_index_builds": builds,
        "communication": communication,
    }


def _history_from_events(events, n_workers):
    outputs = _outputs_from_events(events, n_workers)
    return outputs["history"], outputs["losses"]


def _metric_from_event(event):
    skip = {"run_id", "role", "phase", "monotonic_s", "mode", "transport", "step_id"}
    return {key: value for key, value in event.items() if key not in skip}
