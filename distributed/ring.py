"""Ring AllReduce ranks: neighbor gRPC exchange plus launcher metadata barrier."""

from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import queue
import threading
import time
import traceback
import uuid

import numpy as np

from .constants import BIND_HOST, DEFAULT_CLEANUP_S, PROTOCOL_VERSION, STATUS_OK, STATUS_WAITING
from .gradient_layout import (
    allgather_apply,
    allgather_send_chunk_id,
    layout_from_arrays,
    padded_chunks,
    flatten_gradients,
    restore_gradients,
    scatter_reduce_apply,
    scatter_send_chunk_id,
)
from .launcher import allocate_port, _outputs_from_events
from .monitor import emit
from .protocol import encode_tensors
from .schedule import BatchCursor, iter_steps, metric_row, resolve_budget
from .session import TrainSession
from .transport_grpc import GrpcPsClient, GrpcRingClient, RingInbox, serve_ps_grpc, serve_ring_grpc
from .worker import _data_meta_payload, shard_plan


class MetadataBarrier:
    def __init__(self, n_workers):
        self.n_workers = int(n_workers)
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.init = {}
        self.rounds = {}
        self.stopped = False
        self.aborted = None

    def handle(self, message, wait_s=0.0):
        kind = message.get("type")
        if kind == "join_init":
            return self._join_init(message, wait_s)
        if kind == "confirm_update":
            return self._confirm(message, wait_s)
        if kind == "stop":
            self.stopped = True
            return {"status": STATUS_OK, "request_id": message.get("request_id")}
        return {"status": "INVALID_MESSAGE", "error": f"unknown {kind}",
                "request_id": message.get("request_id")}

    def _join_init(self, message, wait_s=0.0):
        with self.cv:
            rank = int(message["worker_id"])
            fingerprint = message.get("digest")
            previous = self.init.get(rank)
            if previous not in (None, fingerprint):
                return {"status": "DUPLICATE_CONFLICT", "error": "init fingerprint conflict",
                        "request_id": message.get("request_id")}
            self.init[rank] = fingerprint
            self.cv.notify_all()
            deadline = time.monotonic() + max(0.0, float(wait_s))
            while len(self.init) < self.n_workers and time.monotonic() < deadline:
                self.cv.wait(max(0.0, deadline - time.monotonic()))
            ready = len(self.init) == self.n_workers and len(set(self.init.values())) == 1
            status = STATUS_OK if ready else STATUS_WAITING
            return {"status": status, "ready": ready, "request_id": message.get("request_id")}

    def _confirm(self, message, wait_s=0.0):
        with self.cv:
            step = int(message["global_step"])
            rnd = self.rounds.setdefault(step, {})
            rank = int(message["worker_id"])
            summary = (message.get("digest"), int(message.get("parameter_version", 0)),
                       int(message.get("optimizer_step", 0)), message.get("update_id"))
            previous = rnd.get(rank)
            if previous not in (None, summary):
                return {"status": "DUPLICATE_CONFLICT", "error": "update summary conflict",
                        "request_id": message.get("request_id")}
            rnd[rank] = summary
            self.cv.notify_all()
            deadline = time.monotonic() + max(0.0, float(wait_s))
            while len(rnd) < self.n_workers and time.monotonic() < deadline:
                self.cv.wait(max(0.0, deadline - time.monotonic()))
            if len(rnd) < self.n_workers:
                return {"status": STATUS_WAITING, "round_ready": False,
                        "request_id": message.get("request_id")}
            if len(set(rnd.values())) != 1:
                return {"status": "DUPLICATE_CONFLICT", "error": "ring update summaries differ",
                        "request_id": message.get("request_id")}
            for old in list(self.rounds):
                if old < step - 2:
                    self.rounds.pop(old, None)
            self.cv.notify_all()
            return {"status": STATUS_OK, "round_ready": True, "request_id": message.get("request_id")}


def run_ring(config):
    config = dict(config)
    n = int(config["train_workers"])
    config["ring_ports"] = [allocate_port(config.get("bind_host", BIND_HOST)) for _ in range(n)] if n > 1 else []
    config["barrier_port"] = allocate_port(config.get("bind_host", BIND_HOST)) if n > 1 else 0
    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    events = ctx.Queue(maxsize=512)
    barrier_thread = None
    if n > 1:
        barrier = MetadataBarrier(n)
        barrier_thread = threading.Thread(
            target=serve_ps_grpc, args=(barrier, config["barrier_port"], stop),
            kwargs={"host": config.get("bind_host", BIND_HOST), "timeout": float(config.get("timeout", 30))},
            daemon=True,
        )
        barrier_thread.start()
    processes = []
    _sizes, shards = shard_plan(config)
    for rank, shard in enumerate(shards):
        processes.append(ctx.Process(
            target=rank_main, name=f"myflows-ring-{rank}",
            args=(rank, config, shard, events, stop)))
    collected, result = [], None
    started = time.perf_counter()
    planned = int(config.get("planned_steps") or config["steps"])
    try:
        for process in processes:
            process.start()
        deadline = started + float(config["timeout"]) * (planned + 8) + 20
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
                          "error": "; ".join(f"{p.name} exited {p.exitcode}" for p in failed)}
                break
            if processes and all(p.exitcode == 0 for p in processes):
                result = {"status": "passed"}
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
        events.cancel_join_thread()
        events.close()
        cleanup_s = time.perf_counter() - cleanup_start
        if barrier_thread is not None:
            barrier_thread.join(timeout=1)
    result = result or {"status": "failed", "error": "run aborted"}
    if alive or cleanup_s > DEFAULT_CLEANUP_S:
        result.update(status="failed", error="process cleanup did not meet deadline")
    result.update(config=config, events=collected, exitcodes=exitcodes, alive_pids=alive,
                  cleanup_s=cleanup_s, elapsed_s=time.perf_counter() - started)
    result.update(_outputs_from_events(collected, n))
    if result.get("metrics_issues") and result.get("status") == "passed":
        result.update(status="failed", error="; ".join(result["metrics_issues"]))
    if n == 1:
        result["communication"] = False
    else:
        result["communication"] = True
    return result


def rank_main(rank, config, shard, events, stop):
    server_thread = None
    right = None
    barrier = None
    try:
        session = TrainSession(config, for_compute=True)
        n = int(config["train_workers"])
        inbox = None
        init_box = {"payload": None}
        communicate = n > 1
        if communicate:
            inbox = RingInbox()
            dim = sum(node.value.size for node in session.params.values())
            inbox.configure(config['run_id'], rank, n, session.schema_hash(), (dim+n-1)//n)
            inbox.current_step = session.local_version

            def init_handler(message):
                init_box["payload"] = message
                return {"status": STATUS_OK, "error": ""}

            port = config["ring_ports"][rank]
            server_thread = threading.Thread(
                target=serve_ring_grpc, args=(inbox, init_handler, port, stop),
                kwargs={"host": config.get("bind_host", BIND_HOST)}, daemon=True)
            server_thread.start()
            right_rank = (rank + 1) % n
            right = GrpcRingClient(config.get("bind_host", BIND_HOST),
                                   config["ring_ports"][right_rank],
                                   timeout=float(config.get("timeout", 30)))
            barrier = GrpcPsClient(config.get("bind_host", BIND_HOST), config["barrier_port"],
                                   timeout=float(config.get("timeout", 30)))
            _ring_init(rank, n, session, right, init_box, config, stop)
            _barrier_wait(barrier, {
                "protocol_version": PROTOCOL_VERSION,
                "run_id": config["run_id"],
                "worker_id": rank,
                "request_id": f"join-{rank}",
                "type": "join_init",
                "digest": session.digest(),
            }, stop, config)
        emit(events, config, "ring", rank, -1, "session_ready", communication=communicate)
        data = session.task.make_data(config)
        emit(events, config, "ring", rank, -1, "data_meta",
             **_data_meta_payload(data, config, rank))
        budget = resolve_budget(config, data)
        cursor = BatchCursor(session.task, data, config)
        from .measurement import Recorder
        recorder = Recorder(session, data, config, rank, stop)
        recorder.clients = [right, barrier] if communicate else []
        recorder.prepare(cursor, shard)
        collect_history = bool(config.get("collect_history", True))
        history = [session.named_parameters_cpu()] if collect_history else []
        losses = []
        metrics = []
        layout = layout_from_arrays(session.named_parameters_cpu()) if communicate else None
        last_step = -1
        step_offset = session.local_version
        for epoch, step_in_epoch, step in iter_steps(cursor, budget):
            step += step_offset
            last_step = step
            if stop.is_set():
                break
            if step_in_epoch == 0:
                recorder.start_epoch(epoch)
            step_start = time.perf_counter()
            traffic_before = recorder.traffic()
            x, y = cursor.get(epoch, step_in_epoch, shard)
            data_s = time.perf_counter() - step_start
            fault = config.get("fault") if rank == 0 and step == int(config.get("fault_step", 0)) else None
            if fault == "crash":
                os._exit(17)
            if fault == "timeout":
                stop.wait(float(config.get("timeout", 1)) * 3)
                raise TimeoutError("injected rank timeout")
            computed = session.forward_backward(
                x, y, materialize_cpu_gradients=communicate)
            update_id = f"{config['run_id']}:{step}:{session.local_version}"
            sync_start = time.perf_counter()
            sync_s = 0.
            if communicate:
                weighted = {name: value * np.float32(computed["n_samples"])
                            for name, value in computed["gradients"].items()}
                summed = _exchange(rank, n, session, layout, weighted, step, right, inbox, config, stop)
                total = int(sum(config["shard_sizes"]))
                avg = {name: value / np.float32(total) for name, value in summed.items()}
                sync_s = time.perf_counter() - sync_start
                session.apply_global_gradients(avg, update_id)
            else:
                avg = computed['device_gradients']
                session.apply_global_gradients(computed["device_gradients"], update_id)
            confirm_start = time.perf_counter()
            digest_s = 0.
            if communicate:
                digest = session.digest()
                digest_s = time.perf_counter() - confirm_start
                confirm = _barrier_wait(barrier, {
                    "protocol_version": PROTOCOL_VERSION,
                    "run_id": config["run_id"],
                    "worker_id": rank,
                    "request_id": f"upd-{rank}-{step}",
                    "type": "confirm_update",
                    "global_step": step,
                    "parameter_version": session.local_version,
                    "optimizer_step": int(session.optimizer_meta().get("t", step + 1)),
                    "digest": digest,
                    "update_id": update_id,
                }, stop, config)
                if not confirm.get("round_ready"):
                    raise RuntimeError("ring barrier missing round_ready")
            recorder.record(epoch, step, step_start, data_s, computed, sync_s,
                            time.perf_counter()-confirm_start if communicate else 0.,
                            digest_s, traffic_before, avg)
            row = metric_row(
                rank, epoch, step, computed["n_samples"], computed["loss"],
                computed.get("metrics"), computed.get("metric_stats"))
            losses.append(float(row["loss"]))
            metrics.append(row)
            if collect_history:
                history.append(session.named_parameters_cpu())
            emit(events, config, "ring", rank, step, "metrics",
                 **{k: v for k, v in row.items() if k != "worker_id"})
            emit(events, config, "ring", rank, step, "gradient_acknowledged",
                 n_samples=computed["n_samples"], update_id=update_id,
                 backends=computed.get("backends"), communication=communicate)
            if step_in_epoch + 1 == cursor.n_batches(epoch):
                recorder.end_epoch(epoch, step)
        recorder.finish()
        emit(events, config, "ring", rank, last_step + 1, "done", digest=session.digest(),
             communication=communicate, cpu_grad_materializations=session.cpu_grad_materializations,
             epoch_index_builds=cursor.builds)
        emit(events, config, "ring", rank, last_step + 1, "history",
             history=[{k: np.asarray(v) for k, v in snap.items()} for snap in history],
             losses=losses, metrics=metrics)
    except BaseException as exc:
        if not stop.is_set():
            emit(events, config, "ring", rank, -1, "error",
                 error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        stop.set()
        raise SystemExit(1) from None
    finally:
        if right is not None:
            right.close()
        if barrier is not None:
            barrier.close()


def _ring_init(rank, n, session, right, init_box, config, stop):
    if n == 1 or right is None:
        return
    if rank == 0:
        params = encode_tensors(session.named_parameters_cpu())
        right.init({
            "protocol_version": PROTOCOL_VERSION,
            "run_id": config["run_id"],
            "sender_rank": rank,
            "request_id": f"init-{rank}",
            "schema_hash": session.schema_hash(),
            "parameter_hash": session.parameter_hash(),
            "parameter_version": 0,
            "parameters": params,
        })
        return
    deadline = time.perf_counter() + float(config.get("timeout", 30))
    while init_box["payload"] is None:
        if stop.is_set() or time.perf_counter() > deadline:
            raise TimeoutError("ring init did not arrive")
        time.sleep(0.02)
    payload = init_box["payload"]
    from .protocol import decode_tensors
    params = decode_tensors(payload["parameters"], session.schema())
    session.load_parameters(params)
    if rank != n - 1:
        right.init({
            "protocol_version": PROTOCOL_VERSION,
            "run_id": config["run_id"],
            "sender_rank": rank,
            "request_id": f"init-{rank}",
            "schema_hash": session.schema_hash(),
            "parameter_hash": session.parameter_hash(),
            "parameter_version": 0,
            "parameters": payload["parameters"],
        })


def _exchange(rank, n, session, layout, gradients, step, right, inbox, config, stop):
    inbox.current_step = step
    flat = flatten_gradients(gradients, layout)
    # `padded_chunks` owns one reusable contiguous buffer.  The old path
    # copied that complete matrix once here and then copied it again after
    # every ScatterReduce/AllGather round.  Ring only changes one chunk per
    # round, so keep the matrix in place.
    chunks, _pad = padded_chunks(flat, n)
    timeout = float(config.get("timeout", 30))
    collective = f"{config['run_id']}:{step}"
    left = (rank - 1) % n
    for round_id in range(n - 1):
        send_id = scatter_send_chunk_id(rank, round_id, n)
        _send_chunk(right, config, rank, (rank + 1) % n, "SCATTER", round_id, send_id,
                    step, collective, chunks[send_id], session)
        recv_id = (rank - round_id - 1) % n
        incoming = _wait_chunk(inbox, collective, "SCATTER", round_id, recv_id, left, step, timeout, stop)
        chunks = scatter_reduce_apply(chunks, incoming, rank, round_id, n, inplace=True)
    for round_id in range(n - 1):
        send_id = allgather_send_chunk_id(rank, round_id, n)
        if rank == 0 and config.get('fault') == 'gather_crash' and step == int(config.get('fault_step', 0)):
            os._exit(18)
        _send_chunk(right, config, rank, (rank + 1) % n, "GATHER", round_id, send_id,
                    step, collective, chunks[send_id], session)
        recv_id = (rank - round_id) % n
        incoming = _wait_chunk(inbox, collective, "GATHER", round_id, recv_id, left, step, timeout, stop)
        chunks = allgather_apply(chunks, incoming, rank, round_id, n, inplace=True)
    restored = restore_gradients(chunks.reshape(-1)[:layout["numel"]], layout, copy=False)
    return restored


def _send_chunk(client, config, sender, receiver, stage, round_id, chunk_id, step, collective, chunk, session):
    data = np.ascontiguousarray(chunk, dtype="<f4").tobytes()
    digest = hashlib.sha256(data).hexdigest()
    message = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": config["run_id"],
        "collective_id": collective,
        "global_step": step,
        "parameter_version": session.local_version,
        "sender_rank": sender,
        "receiver_rank": receiver,
        "stage": stage,
        "round": round_id,
        "chunk_id": chunk_id,
        "valid_length": int(np.asarray(chunk).size),
        "schema_hash": session.schema_hash(),
        "payload_hash": digest,
        "data": data,
        "request_id": f"{collective}-{stage}-{round_id}-{sender}",
    }
    fault = config.get('fault') if sender == 0 and step == int(config.get('fault_step', 0)) else None
    if fault == 'schema':
        message['schema_hash'] = 'injected-invalid-schema'
    client.receive(message)
    if fault == 'duplicate':
        client.receive(message)
    if config.get('numerical_snapshots'):
        import json
        from pathlib import Path
        path = Path(config['artifact_dir']) / f'rank-{sender}' / 'messages.jsonl'
        with path.open('a', encoding='utf-8') as f:
            f.write(json.dumps({k:v for k,v in message.items() if k != 'data'})+'\n')


def _wait_chunk(inbox, collective, stage, round_id, chunk_id, sender, step, timeout, stop):
    key = (collective, stage, int(round_id), int(chunk_id), int(sender), int(step))
    deadline = time.monotonic() + timeout
    while True:
        if stop.is_set():
            raise RuntimeError("run stopped")
        try:
            message = inbox.wait(key, timeout=min(0.1, max(0.01, deadline - time.monotonic())))
            array = np.frombuffer(message["data"], dtype="<f4")
            # The protobuf response owns an immutable bytes object.  Keeping
            # a NumPy view avoids a second full chunk copy; the view remains
            # alive until the reduction has consumed it.
            return array
        except TimeoutError:
            if time.monotonic() >= deadline:
                raise
            continue


def _barrier_wait(client, message, stop, config):
    timeout = float(config.get("timeout", 30))
    deadline = time.perf_counter() + timeout
    while True:
        if stop.is_set():
            raise RuntimeError("run stopped")
        reply = client.call(message)
        if reply.get("status") == STATUS_OK:
            return reply
        if reply.get("status") == STATUS_WAITING:
            if time.perf_counter() >= deadline:
                raise TimeoutError(reply.get("error") or "barrier timeout")
            time.sleep(float(config.get('poll_interval_s', 0.001)))
            message = dict(message)
            message["request_id"] = message.get("request_id") or uuid.uuid4().hex
            continue
        raise RuntimeError(f"{reply.get('status')}: {reply.get('error')}")
