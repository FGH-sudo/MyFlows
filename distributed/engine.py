"""Transport-agnostic synchronous PS state machine.

PS only verifies and aggregates CPU gradients. Optimizer updates happen on workers.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time

import numpy as np

from .constants import (
    PROTOCOL_VERSION,
    STATUS_ABORTED,
    STATUS_DUP,
    STATUS_INVALID,
    STATUS_OK,
    STATUS_SCHEMA,
    STATUS_STALE,
    STATUS_WAITING,
    DEFAULT_PS_DIGEST_ROUNDS,
    DEFAULT_PS_RETAIN_ROUNDS,
)
from .protocol import (
    aggregate_weighted,
    clone_arrays,
    decode_tensors,
    encode_tensors,
    make_update_id,
    payload_hash,
    schema_from_arrays,
    schema_hash,
)


class ProtocolError(ValueError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class _Round:
    def __init__(self):
        self.contributions = {}
        self.avg = None
        self.applied = {}
        self.committed = False
        self.ready = None


class PSEngine:
    def __init__(
        self,
        *,
        run_id,
        n_workers,
        init_parameters,
        optimizer_meta,
        shard_sizes,
        protocol_version=PROTOCOL_VERSION,
        heartbeat_timeout_s=5.0,
        reconnect_wait_s=10.0,
        retain_rounds=DEFAULT_PS_RETAIN_ROUNDS,
        digest_rounds=DEFAULT_PS_DIGEST_ROUNDS,
        wait_strategy="poll",
    ):
        self.run_id = run_id
        self.n_workers = int(n_workers)
        self.protocol_version = protocol_version
        self.shard_sizes = [int(size) for size in shard_sizes]
        if len(self.shard_sizes) != self.n_workers:
            raise ValueError("shard_sizes must match worker count")
        self.init_parameters = clone_arrays(init_parameters)
        self.optimizer_meta = dict(optimizer_meta)
        self.schema = schema_from_arrays(self.init_parameters)
        self.schema_hash = schema_hash(self.schema)
        self.init_parameter_hash = payload_hash(self.init_parameters)
        self.heartbeat_timeout_s = float(heartbeat_timeout_s)
        self.reconnect_wait_s = float(reconnect_wait_s)
        self.retain_rounds = int(retain_rounds)
        self.digest_rounds = int(digest_rounds)
        if wait_strategy not in ("poll", "notify"):
            raise ValueError("wait_strategy must be poll or notify")
        self.wait_strategy = wait_strategy
        if self.retain_rounds < 0:
            raise ValueError("retain_rounds must be >= 0")
        if self.digest_rounds < 1:
            raise ValueError("digest_rounds must be >= 1")
        if self.digest_rounds < self.retain_rounds:
            raise ValueError("digest_rounds must be >= retain_rounds")
        self.lock = threading.RLock()
        self.cv = threading.Condition(self.lock)
        self.heartbeats = {i: time.monotonic() for i in range(self.n_workers)}
        self.initialized = set()
        self.committed_version = 0
        self.rounds = {}
        self.reply_cache = {}
        self.request_index = {}
        self.aborted = None
        self.stopped = False

    def handle(self, message, *, wait_s=0.0):
        if not isinstance(message, dict):
            return {"status": STATUS_INVALID, "error": "message must be an object", "request_id": None}
        try:
            if self.aborted:
                return self._reply(message, STATUS_ABORTED, self.aborted)
            self._check_common(message)
            kind = message.get("type")
            dispatch = {
                "heartbeat": self._heartbeat,
                "stop": self._stop,
                "init": self._init,
                "initial_state_applied": self._initial_applied,
                "push": self._push,
                "pull": lambda msg: self._poll(self._pull_once, msg, wait_s),
                "update_applied": lambda msg: self._poll(self._update_once, msg, wait_s),
            }
            handler = dispatch.get(kind)
            if handler is None:
                raise ProtocolError(STATUS_INVALID, f"unknown type {kind}")
            return handler(message)
        except ProtocolError as exc:
            return self._reply(message, exc.status, str(exc))
        except Exception as exc:  # noqa: BLE001
            return self._reply(message, STATUS_INVALID, f"{type(exc).__name__}: {exc}")

    def abort(self, reason):
        with self.lock:
            self.aborted = str(reason)
            self.cv.notify_all()
            return self.aborted

    def init_ready(self):
        with self.lock:
            return len(self.initialized) == self.n_workers

    def check_liveness(self, now=None):
        """Fail the synchronous run rather than silently dropping a lost worker."""
        with self.lock:
            if not self.init_ready() or self.stopped or self.aborted:
                return []
            now = time.monotonic() if now is None else now
            missing = [rank for rank, last in self.heartbeats.items()
                       if now-last > self.heartbeat_timeout_s]
            if missing:
                self.abort(f'heartbeat timeout: workers {missing}')
            return missing

    def _reply(self, message, status, error="", **fields):
        payload = {
            "protocol_version": self.protocol_version,
            "run_id": self.run_id,
            "request_id": message.get("request_id") if isinstance(message, dict) else None,
            "status": status,
            "error": error,
        }
        payload.update(fields)
        return payload

    def _cached(self, message, handler):
        existing = self._existing_ok(message)
        if existing is not None:
            return existing
        reply = handler()
        if reply.get("status") == STATUS_OK:
            self._store_ok(message, reply)
        return reply

    def _check_common(self, message):
        if message.get("protocol_version") != self.protocol_version:
            raise ProtocolError(STATUS_INVALID, "protocol_version mismatch")
        if message.get("run_id") != self.run_id:
            raise ProtocolError(STATUS_INVALID, "run_id mismatch")
        worker = message.get("worker_id")
        if type(worker) is not int or not 0 <= worker < self.n_workers:
            raise ProtocolError(STATUS_INVALID, "unknown worker")
        if not message.get("request_id"):
            raise ProtocolError(STATUS_INVALID, "request_id required")

    def _heartbeat(self, message):
        with self.lock:
            self.heartbeats[message["worker_id"]] = time.monotonic()
            return self._reply(message, STATUS_OK)

    def _stop(self, message):
        with self.lock:
            self.stopped = True
            self.aborted = self.aborted or "stopped"
            self.cv.notify_all()
            return self._reply(message, STATUS_OK)

    def _init(self, message):
        with self.lock:
            return self._cached(message, lambda: self._init_body(message))

    def _init_body(self, message):
        if message.get("schema_hash") != self.schema_hash:
            raise ProtocolError(STATUS_SCHEMA, "schema_hash mismatch")
        fields = {
            "parameter_hash": self.init_parameter_hash,
            "schema": self.schema,
            "schema_hash": self.schema_hash,
            "optimizer": self.optimizer_meta,
            "parameter_version": getattr(self, 'initial_version', 0),
            "fingerprint": self.init_parameter_hash,
        }
        if message.get("_grpc_wire_arrays"):
            fields["parameter_arrays"] = self.init_parameters
        else:
            fields["parameters"] = encode_tensors(self.init_parameters)
        return self._reply(message, STATUS_OK, **fields)

    def _initial_applied(self, message):
        with self.lock:
            if message.get("parameter_hash") != self.init_parameter_hash:
                raise ProtocolError(STATUS_SCHEMA, "initial parameter hash mismatch")
            self.initialized.add(message["worker_id"])
            self.heartbeats[message["worker_id"]] = time.monotonic()
            return self._reply(
                message, STATUS_OK, ready=len(self.initialized) == self.n_workers,
                initialized=sorted(self.initialized),
            )

    def _round(self, step):
        step = int(step)
        if step < int(self.committed_version):
            raise ProtocolError(STATUS_STALE, "cannot recreate gc'd round")
        return self.rounds.setdefault(step, _Round())

    def _round_or_stale(self, step):
        step = int(step)
        rnd = self.rounds.get(step)
        if rnd is not None:
            return rnd
        if step < int(self.committed_version):
            raise ProtocolError(STATUS_STALE, "request outside retention window")
        return None

    def _ensure_init(self):
        if len(self.initialized) != self.n_workers:
            raise ProtocolError(STATUS_INVALID, "workers have not finished init")

    def _push(self, message):
        with self.lock:
            return self._cached(message, lambda: self._push_body(message))

    def _push_body(self, message):
        self._ensure_init()
        worker = message["worker_id"]
        step = int(message["global_step"])
        version = int(message["parameter_version"])
        if version != self.committed_version or step < 0 or step < int(self.committed_version):
            raise ProtocolError(STATUS_STALE, "stale push version/step")
        rnd = self._round(step)
        if rnd.committed:
            raise ProtocolError(STATUS_STALE, "push after round committed")
        if worker in rnd.contributions:
            raise ProtocolError(STATUS_DUP, "duplicate worker gradient")
        if int(message.get("n_samples")) != self.shard_sizes[worker]:
            raise ProtocolError(STATUS_INVALID, "n_samples does not match shard plan")
        if message.get("schema_hash") not in (None, self.schema_hash):
            raise ProtocolError(STATUS_SCHEMA, "schema_hash mismatch")
        tensors = (message.get("tensor_arrays")
                   if message.get("tensor_arrays") is not None
                   else message.get("tensors") or message.get("gradients"))
        gradients = decode_tensors(tensors, self.schema)
        expected_hash = payload_hash(gradients)
        if message.get("payload_hash") not in (None, expected_hash):
            raise ProtocolError(STATUS_INVALID, "gradient payload hash mismatch")
        if not _finite_loss(message.get("loss")):
            raise ProtocolError(STATUS_INVALID, "non-finite loss")
        rnd.contributions[worker] = {
            "worker_id": worker,
            "n_samples": int(message["n_samples"]),
            "loss": float(message["loss"]),
            "gradients": gradients,
        }
        if len(rnd.contributions) == self.n_workers:
            ordered = [rnd.contributions[i] for i in range(self.n_workers)]
            avg, loss, total = aggregate_weighted(ordered)
            rnd.avg = {
                "gradients": avg,
                "loss": loss,
                "n_samples": total,
                "update_id": make_update_id(self.run_id, step, version),
                "base_parameter_version": version,
                "target_parameter_version": version + 1,
                "payload_hash": payload_hash(avg),
            }
            self.cv.notify_all()
        return self._reply(message, STATUS_OK, accepted=True, global_step=step)

    def _poll(self, once, message, wait_s):
        if self.wait_strategy == "notify":
            return self._wait_for_state(once, message, wait_s)
        # Keep the established scheduling behavior as the default: removing
        # these waits helped tiny MLPs but regressed shared-GPU ResNet runs.
        deadline = time.monotonic() + max(0.0, float(wait_s))
        while True:
            reply = once(message)
            if reply.get("status") != STATUS_WAITING or time.monotonic() >= deadline:
                return reply
            time.sleep(min(0.02, max(0.0, deadline - time.monotonic())))

    def _wait_for_state(self, once, message, wait_s):
        deadline = time.monotonic() + max(0.0, float(wait_s))
        # Check and wait under the same lock as state transitions. Condition
        # wait releases the lock, allowing peer RPCs and heartbeats to progress,
        # and avoids both lost notifications and 20 ms polling delays.
        with self.cv:
            while True:
                if self.aborted:
                    return self._reply(message, STATUS_ABORTED, self.aborted)
                reply = once(message)
                remaining = deadline - time.monotonic()
                if reply.get("status") != STATUS_WAITING or remaining <= 0:
                    return reply
                self.cv.wait(remaining)

    def _pull_once(self, message):
        with self.lock:
            cached = self._existing_ok(message)
            if cached is not None:
                return cached
            self._ensure_init()
            step = int(message.get("global_step", 0))
            version = int(message.get("parameter_version", self.committed_version))
            rnd = self._round_or_stale(step)
            if rnd is None or rnd.avg is None:
                return self._reply(message, STATUS_WAITING, "average gradient not ready")
            if rnd.avg.get("gradients") is None:
                raise ProtocolError(STATUS_STALE, "pull payload expired")
            if version != rnd.avg["base_parameter_version"]:
                raise ProtocolError(STATUS_STALE, "stale pull version")
            payload = rnd.avg
            fields = {
                "payload_hash": payload["payload_hash"],
                "update_id": payload["update_id"],
                "base_parameter_version": payload["base_parameter_version"],
                "target_parameter_version": payload["target_parameter_version"],
                "n_samples": payload["n_samples"],
                "loss": payload["loss"],
            }
            if message.get("_grpc_wire_arrays"):
                fields["tensor_arrays"] = payload["gradients"]
            else:
                fields["tensors"] = encode_tensors(payload["gradients"])
            reply = self._reply(message, STATUS_OK, **fields)
            self._store_ok(message, reply)
            return reply

    def _update_once(self, message):
        with self.lock:
            cached = self._existing_ok(message)
            if cached is not None:
                return cached
            self._ensure_init()
            step = int(message.get("global_step", 0))
            rnd = self._round_or_stale(step)
            if rnd is None or rnd.avg is None:
                return self._reply(message, STATUS_WAITING, "average gradient not ready")
            summary = self._checked_update_summary(message, rnd)
            if rnd.committed and rnd.ready is not None:
                reply = self._reply(message, STATUS_OK, **rnd.ready)
                self._store_ok(message, reply)
                return reply
            previous = rnd.applied.get(message["worker_id"])
            if previous is None:
                rnd.applied[message["worker_id"]] = summary
            if len(rnd.applied) < self.n_workers:
                reply = self._reply(message, STATUS_WAITING, "waiting for remaining update confirmations")
                self._index_request(message, reply)
                return reply
            digests = {item["digest"] for item in rnd.applied.values()}
            steps = {item["optimizer_step"] for item in rnd.applied.values()}
            versions = {item["parameter_version"] for item in rnd.applied.values()}
            if len(digests) != 1 or len(steps) != 1 or len(versions) != 1:
                raise ProtocolError(STATUS_DUP, "worker update summaries differ")
            self.committed_version = rnd.avg["target_parameter_version"]
            rnd.committed = True
            rnd.ready = {
                "round_ready": True,
                "parameter_version": self.committed_version,
                "update_id": rnd.avg["update_id"],
                "global_step": step,
            }
            reply = self._reply(message, STATUS_OK, **rnd.ready)
            self._store_ok(message, reply)
            self._gc_locked()
            self.cv.notify_all()
            return reply

    def _checked_update_summary(self, message, rnd):
        if message.get("update_id") != rnd.avg["update_id"]:
            raise ProtocolError(STATUS_STALE, "update_id mismatch")
        if int(message.get("parameter_version")) != rnd.avg["target_parameter_version"]:
            raise ProtocolError(STATUS_STALE, "updated parameter version mismatch")
        summary = {
            "worker_id": message["worker_id"],
            "parameter_version": int(message["parameter_version"]),
            "optimizer_step": int(message.get("optimizer_step", 0)),
            "digest": message.get("digest"),
        }
        previous = rnd.applied.get(message["worker_id"])
        if previous is not None and previous != summary:
            raise ProtocolError(STATUS_DUP, "conflicting update summary")
        return summary

    def _existing_ok(self, message):
        key = (message.get("type"), message.get("worker_id"), message.get("request_id"))
        digest = _content_hash(message)
        cached = self.reply_cache.get(key)
        if cached is not None:
            if cached["digest"] != digest:
                raise ProtocolError(STATUS_DUP, f"conflicting duplicate {message.get('type')}")
            return cached["reply"]
        indexed = self.request_index.get(key)
        if indexed is not None:
            if indexed["digest"] != digest:
                raise ProtocolError(STATUS_DUP, f"conflicting duplicate {message.get('type')}")
            if indexed.get("light") is not None:
                return indexed["light"]
        return None

    def _index_request(self, message, reply):
        key = (message.get("type"), message.get("worker_id"), message.get("request_id"))
        digest = _content_hash(message)
        kind = message.get("type")
        step = int(message.get("global_step", -1))
        existing = self.request_index.get(key)
        if existing is not None and existing["digest"] != digest:
            raise ProtocolError(STATUS_DUP, f"conflicting duplicate {kind}")
        light = _light_reply(reply) if reply.get("status") == STATUS_OK else None
        self.request_index[key] = {
            "digest": digest, "light": light, "step": step, "type": kind,
        }
        if light is not None:
            self.reply_cache[key] = {
                "digest": digest, "reply": reply, "step": step, "type": kind,
            }

    def _store_ok(self, message, reply):
        self._index_request(message, reply)

    def _gc_locked(self):
        committed = int(self.committed_version)
        tensor_from = max(0, committed - int(self.retain_rounds))
        digest_from = max(0, committed - int(self.digest_rounds))
        for step, rnd in list(self.rounds.items()):
            if not rnd.committed:
                continue
            _strip_contributions(rnd)
            if int(step) < tensor_from:
                _strip_round(rnd)
            if int(step) < digest_from:
                del self.rounds[step]
        for key, cached in list(self.reply_cache.items()):
            kind = cached.get("type") or key[0]
            if kind in ("init", "initial_state_applied"):
                continue
            step = int(cached.get("step", -1))
            if kind in ("push", "pull", "update_applied") and step >= 0 and step < tensor_from:
                del self.reply_cache[key]
        for key, indexed in list(self.request_index.items()):
            kind = indexed.get("type") or key[0]
            if kind in ("init", "initial_state_applied"):
                continue
            step = int(indexed.get("step", -1))
            if kind in ("push", "pull", "update_applied") and step >= 0 and step < digest_from:
                del self.request_index[key]

    def cache_sizes(self):
        with self.lock:
            return {
                "rounds": len(self.rounds),
                "reply_cache": len(self.reply_cache),
                "request_index": len(self.request_index),
            }

    def cache_footprint_bytes(self):
        with self.lock:
            total = self.retained_payload_bytes()
            total += _index_bytes(self.request_index)
            total += _index_bytes(self.reply_cache)
            total += _index_bytes({step: True for step in self.rounds})
            return int(total)

    def retained_payload_bytes(self):
        total = 0
        for rnd in self.rounds.values():
            for contrib in rnd.contributions.values():
                total += _arrays_nbytes(contrib.get("gradients"))
            if rnd.avg:
                total += _arrays_nbytes(rnd.avg.get("gradients"))
        for cached in self.reply_cache.values():
            reply = cached.get("reply") or {}
            for field in ("tensors", "parameters"):
                for item in reply.get(field) or []:
                    data = item.get("data")
                    if data is None:
                        continue
                    total += np.asarray(data, dtype=np.float32).nbytes
            # The gRPC fast path retains decoded NumPy mappings instead of
            # protobuf-style lists.  Include them in diagnostics as well so
            # cache-footprint checks describe the actual retained payload.
            for field in ("tensor_arrays", "parameter_arrays"):
                total += _arrays_nbytes(reply.get(field))
        return int(total)


def _light_reply(reply):
    skip = {"tensors", "tensor_arrays", "parameters", "parameter_arrays", "schema"}
    return {key: value for key, value in reply.items() if key not in skip}


def _strip_contributions(rnd):
    for contrib in rnd.contributions.values():
        contrib["gradients"] = None


def _strip_round(rnd):
    _strip_contributions(rnd)
    if rnd.avg:
        rnd.avg["gradients"] = None


def _index_bytes(container):
    encoded = json.dumps(list(container.items()), default=str, separators=(",", ":")).encode()
    return len(encoded)


def _arrays_nbytes(arrays):
    if not arrays:
        return 0
    return int(sum(np.asarray(value).nbytes for value in arrays.values()))


def _content_hash(message):
    skip = {"monotonic_s"}
    # The payload hash is already checked by the engine.  Do not stringify a
    # full gradient (or the NumPy mapping used by the fast gRPC path) again
    # merely to index an idempotent request.
    if message.get("payload_hash"):
        skip.update({"tensors", "tensor_arrays"})
    elif message.get("parameter_hash"):
        skip.update({"parameters", "parameter_arrays"})
    body = {key: message[key] for key in sorted(message) if key not in skip}
    encoded = json.dumps(body, sort_keys=True, default=str, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _finite_loss(loss):
    try:
        value = float(loss)
    except (TypeError, ValueError):
        return False
    return value == value and abs(value) != float("inf")
