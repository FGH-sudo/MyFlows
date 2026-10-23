"""Gradient synchronization for a training loop that runs in several processes.

The experiment harness (worker.py, ring.py) drives its own measured loop with fault injection.
This module gives any training loop the same synchronous PS (Socket JSON / gRPC) or Ring
AllReduce (gRPC) protocol. The parent process starts a run with ``SyncRun``; each rank process
receives a ``RankSync`` and calls, in the same order on every rank::

    initial = sync.initialize(arrays)             # every rank starts from the run's initial state
    means = sync.allreduce(tensors, n_samples, loss)   # per step: sample-weighted mean of each tensor
    stop = sync.commit(fingerprint, stop=...)     # every rank applied the step; True if one asked to stop
    totals = sync.reduce(key, values)             # plain sums outside the gradient path (validation)

The tensors of a step are fixed by the initial state given to ``SyncRun``. The training entry
synchronizes the gradient of every parameter and the value of every BN running statistic, so
every rank ends each step with the same parameters and BN statistics. ``commit`` compares a
fingerprint of that state across ranks and fails the run when they diverge. The mean shard loss
travels as one more tensor, ``LOSS``.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import queue
import threading
import time
import traceback
import uuid

import numpy as np

from .constants import (
    BIND_HOST,
    DEFAULT_MAX_FRAME,
    LEGAL_MODES,
    PROTOCOL_VERSION,
    STATUS_OK,
    STATUS_WAITING,
)
from .protocol import encode_tensors, payload_hash, schema_hash, tensor_mapping

LOSS = "__loss__"
# JSON spells an FP32 value with up to ~16 characters plus a separator.
_JSON_BYTES_PER_VALUE = 18


def load_state(path):
    with np.load(path, allow_pickle=False) as saved:
        return {name: np.array(saved[name], dtype=np.float32) for name in saved.files}


def save_state(path, arrays):
    """Write the run's initial state; LOSS is added so the PS schema covers every tensor of a step."""
    arrays = {name: np.asarray(value, dtype=np.float32) for name, value in arrays.items()}
    arrays.setdefault(LOSS, np.zeros((1,), np.float32))
    with open(path, "wb") as handle:
        np.savez(handle, **arrays)


def _max_message_bytes(transport, arrays):
    """Message size limit for this run's tensors.

    gRPC messages grow with the model (protobuf carries raw FP32 bytes). Socket JSON keeps the
    framework's 64 MB frame: a full ResNet-18 would need ~200 MB of JSON text per message and
    several times that in Python floats on every rank, so it is refused up front.
    """
    values = sum(int(np.asarray(v).size) for v in arrays.values())
    # Requests and replies carry every tensor once; leave room for names and headers.
    if transport == "socket_json":
        needed = int(values * _JSON_BYTES_PER_VALUE * 1.25) + (1 << 20)
        if needed > DEFAULT_MAX_FRAME:
            raise ValueError(f"Socket JSON needs about {needed / 2**20:.0f} MB per message for {values} values, "
                             f"over its {DEFAULT_MAX_FRAME // 2**20} MB frame limit; use gRPC or a smaller model")
        return DEFAULT_MAX_FRAME
    return max(DEFAULT_MAX_FRAME, int(values * 4 * 1.25) + (1 << 20))


class SyncRun:
    """Parent side of one run: the PS process or the Ring metadata barrier, plus the rank processes."""

    def __init__(self, *, mode, transport, world_size, shard_sizes, initial_state, timeout=300.0,
                 heartbeat_timeout_s=60.0, heartbeat_interval_s=1.0, host=BIND_HOST, run_id=None):
        if (mode, transport) not in LEGAL_MODES or mode == "single":
            raise ValueError(f"illegal mode/transport {mode}/{transport}")
        shard_sizes = [int(size) for size in shard_sizes]
        if len(shard_sizes) != int(world_size) or min(shard_sizes) <= 0:
            raise ValueError("shard_sizes must give every rank a positive batch")
        arrays = load_state(initial_state)
        if LOSS not in arrays:
            raise ValueError("initial state was not written by save_state")
        max_bytes = _max_message_bytes(transport, arrays)
        if max_bytes >= 2**31:
            raise ValueError(f"gRPC cannot carry {max_bytes} bytes per message (2 GB limit)")
        self.spec = {
            "mode": mode, "transport": transport, "world_size": int(world_size), "shard_sizes": shard_sizes,
            "initial_state": str(initial_state), "run_id": run_id or uuid.uuid4().hex, "host": host,
            "timeout": float(timeout), "heartbeat_timeout_s": float(heartbeat_timeout_s),
            "heartbeat_interval_s": float(heartbeat_interval_s), "poll_interval_s": 0.001,
            "max_bytes": max_bytes, "schema_hash": schema_hash(arrays),
        }
        self.ctx = mp.get_context("spawn")
        self.stop = self.ctx.Event()
        self.errors = self.ctx.Queue()
        self.processes = []
        self.server = None
        self._barrier_stop = threading.Event()
        self._barrier_thread = None

    def start(self, target, args=()):
        """Start the PS (or Ring barrier) and one process per rank running ``target(rank, sync, *args)``."""
        from .launcher import allocate_port

        spec, n = self.spec, self.spec["world_size"]
        if spec["mode"] == "ps":
            spec["ps_port"] = allocate_port(spec["host"])
            self.server = self.ctx.Process(target=_serve_ps, name="myflows-ps",
                                           args=(spec, self.stop, self.errors), daemon=True)
            self.server.start()
        elif n > 1:
            from .ring import MetadataBarrier
            from .transport_grpc import serve_ps_grpc

            spec["ring_ports"] = [allocate_port(spec["host"]) for _ in range(n)]
            spec["barrier_port"] = allocate_port(spec["host"])
            self._barrier_thread = threading.Thread(
                target=serve_ps_grpc, args=(MetadataBarrier(n), spec["barrier_port"], self._barrier_stop),
                kwargs={"host": spec["host"], "timeout": spec["timeout"], "max_bytes": spec["max_bytes"]},
                daemon=True)
            self._barrier_thread.start()
        for rank in range(n):
            process = self.ctx.Process(target=_rank_main, name=f"myflows-rank-{rank}",
                                       args=(target, rank, spec, self.stop, self.errors, tuple(args)))
            process.start()
            self.processes.append(process)
        return self

    def first_error(self):
        try:
            return self.errors.get_nowait()
        except queue.Empty:
            return None

    def wait(self, poll_s=0.2):
        """Block until every rank exits; return {"status", "error", "exitcodes"}."""
        error = None
        while True:
            for process in self.processes:
                process.join(timeout=0)
            failed = [p for p in self.processes if p.exitcode not in (None, 0)]
            if self.server is not None and self.server.exitcode not in (None, 0):
                failed.append(self.server)
            reported = self.first_error()
            if reported is not None and error is None:
                error = reported
            if failed or (reported is not None):
                self.stop.set()
                break
            if all(p.exitcode == 0 for p in self.processes):
                break
            time.sleep(poll_s)
        status = "passed" if error is None and all(p.exitcode == 0 for p in self.processes) else "failed"
        if status == "failed" and error is None:
            error = ("?", "; ".join(f"{p.name} exited {p.exitcode}" for p in self.processes + [self.server]
                                    if p is not None and p.exitcode not in (None, 0)) or "run stopped", "")
        return {"status": status, "error": None if error is None else f"rank {error[0]}: {error[1]}",
                "traceback": None if error is None else error[2],
                "exitcodes": {p.name: p.exitcode for p in self.processes}}

    def close(self, timeout_s=10.0):
        self.stop.set()
        self._barrier_stop.set()
        everything = self.processes + ([self.server] if self.server is not None else [])
        deadline = time.monotonic() + timeout_s
        for process in everything:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        for process in everything:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
        if self._barrier_thread is not None:
            self._barrier_thread.join(timeout=2)
        self.errors.cancel_join_thread()
        self.errors.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _rank_main(target, rank, spec, stop, errors, args):
    sync = None
    try:
        sync = connect(spec, rank, stop)
        target(rank, sync, *args)
    except BaseException as exc:  # noqa: BLE001 - every failure must reach the parent and stop the other ranks
        if not stop.is_set():
            errors.put((rank, f"{type(exc).__name__}: {exc}", traceback.format_exc()))
        stop.set()
        raise SystemExit(1) from None
    finally:
        if sync is not None:
            sync.close()


def _serve_ps(spec, stop, errors):
    from .engine import PSEngine

    try:
        engine = PSEngine(run_id=spec["run_id"], n_workers=spec["world_size"],
                          init_parameters=load_state(spec["initial_state"]), optimizer_meta={},
                          shard_sizes=spec["shard_sizes"], heartbeat_timeout_s=spec["heartbeat_timeout_s"])

        def watchdog():
            while not stop.wait(0.5):
                missing = engine.check_liveness()
                if missing:
                    errors.put(("ps", f"no heartbeat from ranks {missing} for {spec['heartbeat_timeout_s']} s", ""))
                    stop.set()
                    return

        threading.Thread(target=watchdog, daemon=True).start()
        if spec["transport"] == "socket_json":
            from .transport_socket import serve_socket
            serve_socket(engine, spec["ps_port"], stop, host=spec["host"], max_bytes=spec["max_bytes"],
                         timeout=spec["timeout"])
        else:
            from .transport_grpc import serve_ps_grpc
            serve_ps_grpc(engine, spec["ps_port"], stop, host=spec["host"], timeout=spec["timeout"],
                          max_bytes=spec["max_bytes"])
    except BaseException as exc:  # noqa: BLE001 - reported to the parent, which stops the ranks
        if not stop.is_set():
            errors.put(("ps", f"{type(exc).__name__}: {exc}", traceback.format_exc()))
        stop.set()
        raise SystemExit(1) from None


def connect(spec, rank, stop):
    if spec["mode"] == "ps":
        return PsSync(spec, rank, stop)
    return RingSync(spec, rank, stop)


class RankSync:
    """Common rank-side state: the run spec, this rank's shard and the protocol round counter."""

    def __init__(self, spec, rank, stop):
        self.spec, self.rank, self.stop = spec, int(rank), stop
        self.world_size = int(spec["world_size"])
        self.n_samples = int(spec["shard_sizes"][self.rank])
        self.total_samples = int(sum(spec["shard_sizes"]))
        self.round = 0
        self.local_version = 0  # read by the Ring chunk exchange
        self._schema = None

    def schema_hash(self):
        return self.spec["schema_hash"]

    def _message(self, kind, **fields):
        return {"protocol_version": PROTOCOL_VERSION, "run_id": self.spec["run_id"], "worker_id": self.rank,
                "type": kind, **fields}

    def _call(self, client, message, wait=False):
        """Send ``message`` until it is answered OK; WAITING replies are retried until the timeout."""
        deadline = time.perf_counter() + self.spec["timeout"]
        while True:
            if self.stop.is_set():
                raise RuntimeError("run stopped by another rank")
            reply = client.call(message)
            if reply.get("status") == STATUS_OK:
                return reply
            if wait and reply.get("status") == STATUS_WAITING:
                if time.perf_counter() >= deadline:
                    raise TimeoutError(f"{message['type']}: other ranks did not arrive within "
                                       f"{self.spec['timeout']} s ({reply.get('error')})")
                time.sleep(self.spec["poll_interval_s"])
                continue
            raise RuntimeError(f"{message['type']} rejected: {reply.get('status')}: {reply.get('error')}")

    def _check_schema(self, arrays):
        names = dict(arrays, **{LOSS: np.zeros((1,), np.float32)})
        if schema_hash(names) != self.spec["schema_hash"]:
            raise ValueError("this rank's tensors differ from the run's initial state (model or batch mismatch)")

    def close(self):
        pass


class PsSync(RankSync):
    """Push the shard's tensors to the PS, pull the weighted mean, confirm the applied update."""

    def __init__(self, spec, rank, stop):
        super().__init__(spec, rank, stop)
        self.grpc = spec["transport"] == "grpc_proto"
        if self.grpc:
            from .transport_grpc import GrpcPsClient
            self.client = GrpcPsClient(spec["host"], spec["ps_port"], timeout=spec["timeout"],
                                       max_bytes=spec["max_bytes"])
        else:
            from .transport_socket import SocketClient
            deadline, last = time.perf_counter() + spec["timeout"], None
            while True:
                try:
                    self.client = SocketClient(spec["host"], spec["ps_port"], timeout=spec["timeout"],
                                               max_bytes=spec["max_bytes"])
                    break
                except OSError as exc:
                    last = exc
                    if time.perf_counter() > deadline:
                        raise ConnectionError(f"PS connect failed: {last}") from None
                    time.sleep(0.05)
        # The heartbeat has its own connection so a long push or pull never delays it.
        self._heartbeat_client = None
        self._heartbeat_stop = threading.Event()
        self._heartbeat = threading.Thread(target=self._beat, daemon=True)
        self._heartbeat.start()
        self._update_id = None

    def _beat(self):
        try:
            if self.grpc:
                from .transport_grpc import GrpcPsClient
                self._heartbeat_client = GrpcPsClient(self.spec["host"], self.spec["ps_port"],
                                                      timeout=self.spec["timeout"])
            else:
                from .transport_socket import SocketClient
                self._heartbeat_client = SocketClient(self.spec["host"], self.spec["ps_port"],
                                                      timeout=self.spec["timeout"])
        except Exception:  # noqa: BLE001 - the main connection reports an unreachable PS
            return
        while not self._heartbeat_stop.is_set():
            try:
                self._heartbeat_client.call(self._message("heartbeat", request_id=f"hb-{uuid.uuid4().hex}"))
            except Exception:  # noqa: BLE001
                if self._heartbeat_stop.is_set():
                    return
            self._heartbeat_stop.wait(self.spec["heartbeat_interval_s"])

    def _tensors(self, arrays):
        return {"tensor_arrays": arrays} if self.grpc else {"tensors": encode_tensors(arrays)}

    def initialize(self, arrays):
        self._check_schema(arrays)
        reply = self._call(self.client, self._message("init", request_id=f"init-{self.rank}",
                                                      schema_hash=self.spec["schema_hash"]))
        initial = tensor_mapping(reply["parameters"])
        self.round = self.local_version = int(reply.get("parameter_version") or 0)
        deadline = time.perf_counter() + self.spec["timeout"]
        while True:  # barrier: the PS accepts pushes only after every rank loaded the initial state
            reply = self._call(self.client, self._message(
                "initial_state_applied", request_id=f"init-applied-{self.rank}-{uuid.uuid4().hex}",
                parameter_hash=payload_hash(initial)))
            if reply.get("ready"):
                break
            if time.perf_counter() > deadline:
                raise TimeoutError("other ranks did not load the initial state in time")
            time.sleep(0.02)
        initial.pop(LOSS, None)
        return initial

    def allreduce(self, tensors, n_samples, loss):
        if int(n_samples) != self.n_samples:
            raise ValueError(f"rank {self.rank} computed {n_samples} samples, the plan gives it {self.n_samples}")
        arrays = {name: np.asarray(value, dtype=np.float32) for name, value in tensors.items()}
        arrays[LOSS] = np.asarray([loss], np.float32)
        step = self.round
        self._call(self.client, self._message(
            "push", request_id=f"push-{self.rank}-{step}", global_step=step, epoch=0, batch_id=f"b{step}",
            parameter_version=step, schema_hash=self.spec["schema_hash"], n_samples=int(n_samples),
            loss=float(loss), payload_hash=payload_hash(arrays), **self._tensors(arrays)))
        pulled = self._call(self.client, self._message(
            "pull", request_id=f"pull-{self.rank}-{step}", global_step=step, parameter_version=step,
            schema_hash=self.spec["schema_hash"]), wait=True)
        self._update_id = pulled["update_id"]
        means = tensor_mapping(pulled["tensors"])
        return means, float(means.pop(LOSS)[0])

    def commit(self, fingerprint, stop=False):
        step = self.round
        reply = self._call(self.client, self._message(
            "update_applied", request_id=f"upd-{self.rank}-{step}", global_step=step,
            parameter_version=step + 1, update_id=self._update_id, optimizer_step=step + 1,
            digest=fingerprint, stop_requested=bool(stop)), wait=True)
        if not reply.get("round_ready"):
            raise RuntimeError("PS confirmed the update without round_ready")
        self.round = self.local_version = step + 1
        return bool(reply.get("stop"))

    def reduce(self, key, values):
        arrays = {name: np.atleast_1d(np.asarray(value, dtype=np.float32)) for name, value in values.items()}
        reply = self._call(self.client, self._message(
            "reduce", request_id=f"reduce-{self.rank}-{key}", reduce_key=key, **self._tensors(arrays)), wait=True)
        return {name: value.astype(np.float64) for name, value in tensor_mapping(reply["tensors"]).items()}

    def close(self):
        self._heartbeat_stop.set()
        self._heartbeat.join(timeout=2)
        for client in (self.client, self._heartbeat_client):
            if client is not None:
                client.close()


class RingSync(RankSync):
    """ScatterReduce + AllGather between neighbours; the parent's barrier confirms each step."""

    def __init__(self, spec, rank, stop):
        super().__init__(spec, rank, stop)
        self.right = self.barrier = self.inbox = None
        self._server = None
        self._server_stop = threading.Event()
        self.layout = None
        self._config = {"run_id": spec["run_id"], "timeout": spec["timeout"]}
        if self.world_size == 1:
            return
        from .transport_grpc import (
            GrpcPsClient,
            GrpcRingClient,
            RingInbox,
            serve_ring_grpc,
        )

        self.inbox = RingInbox()
        port = spec["ring_ports"][self.rank]
        self._server = threading.Thread(
            target=serve_ring_grpc, args=(self.inbox, lambda message: {"status": STATUS_OK, "error": ""},
                                          port, self._server_stop),
            kwargs={"host": spec["host"]}, daemon=True)
        self._server.start()
        right = spec["ring_ports"][(self.rank + 1) % self.world_size]
        self.right = GrpcRingClient(spec["host"], right, timeout=spec["timeout"])
        self.barrier = GrpcPsClient(spec["host"], spec["barrier_port"], timeout=spec["timeout"],
                                    max_bytes=spec["max_bytes"])

    def initialize(self, arrays):
        self._check_schema(arrays)
        initial = load_state(self.spec["initial_state"])
        self.layout = _layout(initial)
        if self.inbox is not None:
            chunk = math.ceil(self.layout["numel"] / self.world_size)
            self.inbox.configure(self.spec["run_id"], self.rank, self.world_size, self.spec["schema_hash"], chunk)
            self._call(self.barrier, self._message("join_init", request_id=f"join-{self.rank}",
                                                   digest=payload_hash(initial)), wait=True)
        initial.pop(LOSS, None)
        return initial

    def allreduce(self, tensors, n_samples, loss):
        if int(n_samples) != self.n_samples:
            raise ValueError(f"rank {self.rank} computed {n_samples} samples, the plan gives it {self.n_samples}")
        # Scale by the FP32 share n_i / N before summing, exactly as the PS aggregates: one rank keeps its
        # tensors bit for bit, and two ranks give the PS result bit for bit (a + b == b + a).
        weight = np.float32(int(n_samples) / self.total_samples)
        weighted = {name: np.asarray(value, dtype=np.float32) * weight for name, value in tensors.items()}
        weighted[LOSS] = np.asarray([loss], np.float32) * weight
        if self.inbox is None:
            summed = weighted
        else:
            from .ring import _exchange
            summed = _exchange(self.rank, self.world_size, self, self.layout, weighted, self.round, self.right,
                               self.inbox, self._config, self.stop)
        means = {name: np.array(value, dtype=np.float32) for name, value in summed.items()}
        return means, float(means.pop(LOSS)[0])

    def commit(self, fingerprint, stop=False):
        step = self.round
        decision = bool(stop)
        if self.barrier is not None:
            reply = self._call(self.barrier, self._message(
                "confirm_update", request_id=f"upd-{self.rank}-{step}", global_step=step,
                parameter_version=step + 1, optimizer_step=step + 1, digest=fingerprint,
                update_id=f"{self.spec['run_id']}:{step}", stop_requested=bool(stop)), wait=True)
            decision = bool(reply.get("stop"))
        self.round = self.local_version = step + 1
        return decision

    def reduce(self, key, values):
        arrays = {name: np.atleast_1d(np.asarray(value, dtype=np.float32)) for name, value in values.items()}
        if self.barrier is None:
            return {name: value.astype(np.float64) for name, value in arrays.items()}
        reply = self._call(self.barrier, self._message(
            "reduce", request_id=f"reduce-{self.rank}-{key}", reduce_key=key, tensor_arrays=arrays), wait=True)
        return {name: value.astype(np.float64) for name, value in tensor_mapping(reply["tensors"]).items()}

    def close(self):
        self._server_stop.set()
        for client in (self.right, self.barrier):
            if client is not None:
                client.close()
        if self._server is not None:
            self._server.join(timeout=2)


def _layout(arrays):
    from .gradient_layout import layout_from_arrays
    return layout_from_arrays(arrays)
