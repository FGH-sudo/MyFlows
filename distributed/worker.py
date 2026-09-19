"""PS worker process: GPU/CPU compute, Push/Pull, local optimizer update."""

from __future__ import annotations

import os
import threading
import time
import uuid

import numpy as np

from .constants import PROTOCOL_VERSION, STATUS_OK, STATUS_WAITING
from .data_meta import lightweight_data_meta
from .monitor import emit
from .protocol import decode_tensors, encode_tensors, payload_hash
from .schedule import BatchCursor, iter_steps, metric_row, resolve_budget
from .session import TrainSession
from .shards import split_global_batch
from .transport_socket import SocketClient


def worker_main(worker_id, config, shard, events, stop):
    client = None
    heartbeat = None
    try:
        emit(events, config, "worker", worker_id, -1, "worker_start")
        session = TrainSession(config, for_compute=True)
        emit(events, config, "worker", worker_id, -1, "session_ready", communication=True)
        client = _connect(config)
        hb_stop = threading.Event()
        heartbeat = threading.Thread(target=_heartbeat, args=(client, worker_id, config, hb_stop), daemon=True)
        heartbeat.start()
        _handshake(client, session, worker_id, config)
        emit(events, config, "worker", worker_id, -1, "handshake_ok", communication=True)
        data = session.task.make_data(config)
        emit(events, config, "worker", worker_id, -1, "data_meta",
             **_data_meta_payload(data, config, worker_id))
        budget = resolve_budget(config, data)
        cursor = BatchCursor(session.task, data, config)
        from .measurement import Recorder
        recorder = Recorder(session, data, config, worker_id, stop)
        recorder.clients = [client]
        recorder.prepare(cursor, shard)
        collect_history = bool(config.get("collect_history", True))
        history = [session.named_parameters_cpu()] if collect_history else []
        losses = []
        metrics = []
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
            fault = config.get("fault") if worker_id == 0 and step == int(config.get("fault_step", 0)) else None
            if fault == "crash":
                os._exit(17)
            if fault == "timeout":
                stop.wait(float(config.get("timeout", 1)) * 3)
                raise TimeoutError("injected worker timeout")
            if fault == 'heartbeat_loss':
                hb_stop.set()
                stop.wait(float(config.get('heartbeat_timeout_s', 5))*3)
                raise TimeoutError('injected heartbeat loss')
            started = time.perf_counter()
            computed = session.forward_backward(x, y, materialize_cpu_gradients=True)
            compute_s = time.perf_counter() - started
            sync_start = time.perf_counter()
            push = {
                "protocol_version": PROTOCOL_VERSION,
                "run_id": config["run_id"],
                "worker_id": worker_id,
                "request_id": f"push-{worker_id}-{step}",
                "type": "push",
                "epoch": epoch,
                "global_step": step,
                "batch_id": f"b{step}",
                "parameter_version": session.local_version,
                "schema_hash": session.schema_hash(),
                "n_samples": int(computed["n_samples"]),
                "loss": computed["loss"],
                "payload_hash": payload_hash(computed["gradients"]),
            }
            # Keep gRPC/protobuf tensors as NumPy arrays until the protobuf
            # adapter writes TensorBlob bytes.  The Socket/JSON path retains
            # its list representation for the protocol comparison.
            if config.get("transport") == "grpc_proto":
                push["tensor_arrays"] = computed["gradients"]
            else:
                push["tensors"] = encode_tensors(computed["gradients"])
            if fault == "schema":
                push["schema_hash"] = "injected-invalid-schema"
            _call_ok(client, push, stop, config)
            if fault == 'retry':
                _call_ok(client, push, stop, config)
            if fault == "duplicate":
                dup = dict(push)
                dup["request_id"] = f"push-{worker_id}-{step}-dup"
                _call_ok(client, dup, stop, config)
            pulled = _call_ok(client, {
                "protocol_version": PROTOCOL_VERSION,
                "run_id": config["run_id"],
                "worker_id": worker_id,
                "request_id": f"pull-{worker_id}-{step}",
                "type": "pull",
                "global_step": step,
                "parameter_version": session.local_version,
                "schema_hash": session.schema_hash(),
            }, stop, config, allow_wait=True)
            avg = decode_tensors(pulled["tensors"], session.schema())
            sync_s = time.perf_counter() - sync_start
            applied = session.apply_global_gradients(avg, pulled["update_id"])
            confirm_start = time.perf_counter()
            digest = session.digest()
            digest_s = time.perf_counter() - confirm_start
            confirm = _call_ok(client, {
                "protocol_version": PROTOCOL_VERSION,
                "run_id": config["run_id"],
                "worker_id": worker_id,
                "request_id": f"upd-{worker_id}-{step}",
                "type": "update_applied",
                "global_step": step,
                "parameter_version": session.local_version,
                "update_id": pulled["update_id"],
                "optimizer_step": int(session.optimizer_meta().get("t", step + 1)),
                "digest": digest,
            }, stop, config, allow_wait=True)
            if not confirm.get("round_ready"):
                raise RuntimeError("missing round_ready")
            recorder.record(epoch, step, step_start, data_s, computed, sync_s,
                            time.perf_counter()-confirm_start, digest_s, traffic_before, avg)
            row = metric_row(
                worker_id, epoch, step, computed["n_samples"], computed["loss"],
                computed.get("metrics"), computed.get("metric_stats"))
            losses.append(float(row["loss"]))
            metrics.append(row)
            if collect_history:
                history.append(session.named_parameters_cpu())
            emit(events, config, "worker", worker_id, step, "metrics",
                 **{k: v for k, v in row.items() if k != "worker_id"})
            emit(events, config, "worker", worker_id, step, "gradient_acknowledged",
                 n_samples=computed["n_samples"], compute_s=compute_s,
                 update_id=pulled["update_id"], applied=applied,
                 backends=computed.get("backends"),
                 gradient_payload_bytes=sum(v.nbytes for v in computed["gradients"].values()))
            if step_in_epoch + 1 == cursor.n_batches(epoch):
                recorder.end_epoch(epoch, step)
        recorder.finish()
        emit(events, config, "worker", worker_id, last_step + 1, "done",
             parameter_hash=session.parameter_hash(), digest=session.digest(),
             communication=True, cpu_grad_materializations=session.cpu_grad_materializations,
             epoch_index_builds=cursor.builds)
        emit(events, config, "worker", worker_id, last_step + 1, "history",
             history=_cpu_history(history), losses=losses, metrics=metrics)
    except BaseException as exc:
        if not stop.is_set():
            emit(events, config, "worker", worker_id, -1, "error", error=f"{type(exc).__name__}: {exc}")
        stop.set()
        raise SystemExit(1) from None
    finally:
        if heartbeat is not None:
            hb_stop.set()
        if client is not None:
            client.close()


def _connect(config):
    transport = config.get("transport", "socket_json")
    host = config.get("bind_host", "127.0.0.1")
    port = config["ps_port"]
    timeout = float(config.get("timeout", 30))
    if transport == "socket_json":
        deadline = time.perf_counter() + timeout
        last = None
        while time.perf_counter() < deadline:
            try:
                return SocketClient(host, port, timeout=timeout)
            except OSError as exc:
                last = exc
                time.sleep(0.05)
        raise ConnectionError(f"ps connect failed: {last}")
    if transport == "grpc_proto":
        from .transport_grpc import GrpcPsClient
        return GrpcPsClient(host, port, timeout=timeout)
    raise ValueError(f"unsupported transport {transport}")


def _handshake(client, session, worker_id, config):
    init = _call_ok(client, {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": config["run_id"],
        "worker_id": worker_id,
        "request_id": f"init-{worker_id}",
        "type": "init",
        "schema_hash": session.schema_hash(),
    }, None, config)
    params = decode_tensors(init["parameters"], init["schema"])
    session.load_parameters(params)
    session.load_optimizer_state(init.get("optimizer"))
    session.local_version = int(init.get("parameter_version", 0))
    deadline = time.perf_counter() + float(config.get("timeout", 30))
    while time.perf_counter() < deadline:
        applied = client.call({
            "protocol_version": PROTOCOL_VERSION,
            "run_id": config["run_id"],
            "worker_id": worker_id,
            "request_id": f"init-applied-{worker_id}-{uuid.uuid4().hex}",
            "type": "initial_state_applied",
            "parameter_hash": session.parameter_hash(),
        })
        if applied.get("status") == STATUS_OK and applied.get("ready"):
            return
        time.sleep(0.02)
    raise TimeoutError("init barrier timed out")


def _heartbeat(client, worker_id, config, stop):
    interval = float(config.get("heartbeat_interval_s", 1.0))
    while not stop.is_set():
        try:
            client.call({
                "protocol_version": PROTOCOL_VERSION,
                "run_id": config["run_id"],
                "worker_id": worker_id,
                "request_id": f"hb-{worker_id}-{uuid.uuid4().hex}",
                "type": "heartbeat",
            })
        except Exception:
            if stop.is_set():
                return
        stop.wait(interval)


def _call_ok(client, message, stop, config, allow_wait=False):
    timeout = float(config.get("timeout", 30))
    deadline = time.perf_counter() + timeout
    request_id = message.get("request_id")
    while True:
        if stop is not None and stop.is_set():
            raise RuntimeError("run stopped")
        reply = client.call(message)
        if reply.get("status") == STATUS_OK:
            return reply
        if allow_wait and reply.get("status") == STATUS_WAITING:
            if time.perf_counter() >= deadline:
                raise TimeoutError(reply.get("error") or "waiting timed out")
            time.sleep(float(config.get('poll_interval_s', 0.001)))
            message = dict(message)
            message["request_id"] = request_id
            continue
        raise RuntimeError(f"{reply.get('status')}: {reply.get('error')}")


def _data_meta_payload(data, config, worker_id):
    meta = lightweight_data_meta(data, config)
    overrides = config.get("test_data_meta_by_worker") or {}
    injected = overrides.get(worker_id, overrides.get(str(worker_id)))
    if injected:
        meta = dict(meta)
        meta["split_fingerprint"] = injected
    return meta


def shard_plan(config):
    sizes = split_global_batch(int(config["global_batch"]), int(config["train_workers"]),
                               config.get("shard_sizes"))
    shards = []
    offset = 0
    for size in sizes:
        shards.append(np.arange(offset, offset + size))
        offset += size
    return sizes, shards


def _cpu_history(history):
    converted = []
    for snapshot in history:
        converted.append({name: np.asarray(value, dtype=np.float32) for name, value in snapshot.items()})
    return converted
