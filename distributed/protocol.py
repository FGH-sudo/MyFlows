"""Validated versioned messages and sample-weighted mean-gradient aggregation."""

import hashlib
import json
import queue
import time

import numpy as np


def schema(params):
    return [{"name": name, "shape": list(value.shape), "dtype": value.dtype.str}
            for name, value in sorted(params.items())]


def schema_hash(params):
    return hashlib.sha256(json.dumps(schema(params), sort_keys=True).encode()).hexdigest()


def payload_hash(params):
    sha = hashlib.sha256(schema_hash(params).encode())
    for name, value in sorted(params.items()):
        sha.update(name.encode())
        sha.update(np.ascontiguousarray(value).tobytes())
    return sha.hexdigest()


def payload_bytes(params):
    return sum(v.nbytes for v in params.values())


def clone(params):
    return {name: value.copy() for name, value in params.items()}


def validate_arrays(arrays, reference):
    if not isinstance(arrays, dict) or set(arrays) != set(reference):
        raise ValueError("parameter/gradient keys differ from schema")
    for name, ref in reference.items():
        value = arrays[name]
        if not isinstance(value, np.ndarray) or value.shape != ref.shape or value.dtype != ref.dtype:
            raise ValueError(f"invalid shape/dtype for {name}")
        if not np.isfinite(value).all():
            raise ValueError(f"non-finite array {name}")


def validate_gradient(message, *, run_id, step, parameter_hash, reference, shard_sizes, received):
    if not isinstance(message, dict) or message.get("kind") != "gradient":
        raise ValueError("expected gradient message")
    worker = message.get("worker_id")
    if type(worker) is not int or not 0 <= worker < len(shard_sizes):
        raise ValueError("unknown worker")
    if worker in received:
        raise ValueError("duplicate worker gradient")
    if any(type(message.get(key)) is not int for key in ("step_id", "parameter_version", "n_samples")):
        raise ValueError("step, version and sample count must be integers")
    for key, expected in (("run_id", run_id), ("step_id", step), ("parameter_version", step),
                          ("parameter_hash", parameter_hash), ("schema_hash", schema_hash(reference)),
                          ("n_samples", shard_sizes[worker])):
        if message.get(key) != expected:
            raise ValueError(f"invalid/stale {key}")
    gradients = message.get("gradients")
    validate_arrays(gradients, reference)
    if message.get("gradient_hash") != payload_hash(gradients):
        raise ValueError("gradient payload hash mismatch")
    if not np.isfinite(message.get("loss", np.nan)):
        raise ValueError("non-finite loss")
    return worker


def aggregate(messages):
    total = sum(m["n_samples"] for m in messages)
    if total <= 0 or any(m["n_samples"] <= 0 for m in messages):
        raise ValueError("shards must be nonempty")
    first = messages[0]["gradients"]
    gradients = {name: np.zeros_like(value) for name, value in first.items()}
    for message in messages:
        for name in gradients:
            gradients[name] += message["gradients"][name] * (message["n_samples"] / total)
    loss = sum(m["loss"] * m["n_samples"] for m in messages) / total
    return gradients, loss, total


def receive(channel, stop, timeout):
    deadline = time.perf_counter() + timeout
    while not stop.is_set():
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("message receive timed out")
        try:
            return channel.get(timeout=min(remaining, 0.05))
        except queue.Empty:
            pass
    raise RuntimeError("run stopped")


def send(channel, value, stop, timeout):
    deadline = time.perf_counter() + timeout
    while not stop.is_set():
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("message submit timed out")
        try:
            channel.put(value, timeout=min(remaining, 0.05))
            return
        except queue.Full:
            pass
    raise RuntimeError("run stopped")


def event(channel, stop, config, role, worker_id, step, phase, **fields):
    send(channel, {"run_id": config["run_id"], "role": role, "worker_id": worker_id,
                   "step_id": step, "phase": phase, "monotonic_s": time.perf_counter(), **fields},
         stop, config["timeout"])
