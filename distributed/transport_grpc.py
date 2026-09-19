"""gRPC adapters for PS dict messages and Ring chunk exchange."""

from __future__ import annotations

from concurrent import futures
import threading
import time
import hashlib

import grpc
import numpy as np

from .constants import BIND_HOST, DEFAULT_MAX_FRAME, DEFAULT_RING_DIGEST_STEPS, PROTOCOL_VERSION, STATUS_OK

try:
    from generated.grpc import common_pb2, ps_pb2, ps_pb2_grpc, ring_pb2, ring_pb2_grpc
except ImportError as exc:  # pragma: no cover
    raise ImportError("run python tools/generate_distributed_proto.py") from exc


_PS_METHODS = {
    "init": "VariableWeightsInit",
    "push": "Push",
    "pull": "Pull",
    "heartbeat": "Heartbeat",
    "initial_state_applied": "InitialStateApplied",
    "update_applied": "UpdateApplied",
    "stop": "Stop",
    # Ring's coordinator carries only readiness and state digests, never gradients.
    "join_init": "InitialStateApplied",
    "confirm_update": "UpdateApplied",
}


def _array_to_blob(name, array):
    arr = np.ascontiguousarray(array, dtype="<f4")
    return common_pb2.TensorBlob(name=name, shape=list(arr.shape), dtype="float32", data=arr.tobytes())


def encoded_to_blobs(items):
    blobs = []
    if isinstance(items, dict):
        for name, value in sorted(items.items()):
            blobs.append(_array_to_blob(name, value))
        return blobs
    for item in items or []:
        array = np.asarray(item["data"], dtype=np.float32).reshape(item["shape"])
        blobs.append(_array_to_blob(item["name"], array))
    return blobs


def blobs_to_encoded(blobs, *, as_arrays=False):
    if as_arrays:
        arrays = {}
        for blob in blobs:
            array = np.frombuffer(blob.data, dtype="<f4")
            expected = int(np.prod(blob.shape) or 0)
            if int(array.size) != expected:
                raise ValueError(f"protobuf tensor length mismatch for {blob.name}")
            arrays[blob.name] = array.reshape(tuple(blob.shape))
        return arrays
    items = []
    for blob in blobs:
        array = np.frombuffer(blob.data, dtype="<f4")
        if int(array.size) != int(np.prod(blob.shape) or 0):
            raise ValueError(f"protobuf tensor length mismatch for {blob.name}")
        items.append({
            "name": blob.name,
            "shape": list(blob.shape),
            "dtype": "float32",
            "data": np.ascontiguousarray(array).reshape(blob.shape).reshape(-1).tolist(),
        })
    return items


def dict_to_ps_request(message):
    req = ps_pb2.PsRequest(
        protocol_version=message.get("protocol_version", PROTOCOL_VERSION),
        run_id=str(message.get("run_id") or ""),
        worker_id=int(message.get("worker_id") or 0),
        request_id=str(message.get("request_id") or ""),
        type=str(message.get("type") or ""),
        epoch=int(message.get("epoch") or 0),
        global_step=int(message.get("global_step") or 0),
        batch_id=str(message.get("batch_id") or ""),
        parameter_version=int(message.get("parameter_version") or 0),
        schema_hash=str(message.get("schema_hash") or ""),
        n_samples=int(message.get("n_samples") or 0),
        loss=float(message.get("loss") or 0.0),
        payload_hash=str(message.get("payload_hash") or ""),
        update_id=str(message.get("update_id") or ""),
        optimizer_step=int(message.get("optimizer_step") or 0),
        digest=str(message.get("digest") or ""),
        parameter_hash=str(message.get("parameter_hash") or ""),
    )
    req.tensors.extend(encoded_to_blobs(message.get("tensor_arrays")
                                       if message.get("tensor_arrays") is not None
                                       else message.get("tensors")))
    return req


def ps_request_to_dict(req):
    return {
        "protocol_version": req.protocol_version,
        "run_id": req.run_id,
        "worker_id": req.worker_id,
        "request_id": req.request_id,
        "type": req.type,
        "epoch": req.epoch,
        "global_step": req.global_step,
        "batch_id": req.batch_id,
        "parameter_version": req.parameter_version,
        "schema_hash": req.schema_hash,
        "n_samples": req.n_samples,
        "loss": req.loss,
        "payload_hash": req.payload_hash,
        "update_id": req.update_id,
        "optimizer_step": req.optimizer_step,
        "digest": req.digest,
        "parameter_hash": req.parameter_hash,
        "tensors": blobs_to_encoded(req.tensors, as_arrays=True),
    }


def dict_to_ps_response(reply):
    resp = ps_pb2.PsResponse(
        protocol_version=str(reply.get("protocol_version") or PROTOCOL_VERSION),
        run_id=str(reply.get("run_id") or ""),
        request_id=str(reply.get("request_id") or ""),
        status=str(reply.get("status") or ""),
        error=str(reply.get("error") or ""),
        payload_hash=str(reply.get("payload_hash") or ""),
        update_id=str(reply.get("update_id") or ""),
        base_parameter_version=int(reply.get("base_parameter_version") or 0),
        target_parameter_version=int(reply.get("target_parameter_version") or 0),
        parameter_version=int(reply.get("parameter_version") or 0),
        n_samples=int(reply.get("n_samples") or 0),
        loss=float(reply.get("loss") or 0.0),
        round_ready=bool(reply.get("round_ready")),
        ready=bool(reply.get("ready")),
        accepted=bool(reply.get("accepted")),
        schema_hash=str(reply.get("schema_hash") or ""),
        parameter_hash=str(reply.get("parameter_hash") or ""),
        fingerprint=str(reply.get("fingerprint") or ""),
        global_step=int(reply.get("global_step") or 0),
    )
    resp.tensors.extend(encoded_to_blobs(reply.get("tensor_arrays")
                                         if reply.get("tensor_arrays") is not None
                                         else reply.get("tensors")))
    if reply.get("parameter_arrays") is not None:
        resp.parameters.extend(encoded_to_blobs(reply.get("parameter_arrays")))
    elif reply.get("parameters"):
        resp.parameters.extend(encoded_to_blobs(reply.get("parameters")))
    optimizer = reply.get("optimizer") or {}
    if optimizer:
        resp.optimizer.impl = str(optimizer.get("impl") or "")
        resp.optimizer.impl_path = str(optimizer.get("impl_path") or "")
        resp.optimizer.learning_rate = float(optimizer.get("learning_rate") or 0.0)
        resp.optimizer.t = int(optimizer.get("t") or 0)
        resp.optimizer.beta_1 = float(optimizer.get("beta_1") or 0.0)
        resp.optimizer.beta_2 = float(optimizer.get("beta_2") or 0.0)
        resp.optimizer.eps = float(optimizer.get("eps") or 0.0)
    return resp


def ps_response_to_dict(resp):
    reply = {
        "protocol_version": resp.protocol_version,
        "run_id": resp.run_id,
        "request_id": resp.request_id,
        "status": resp.status,
        "error": resp.error,
        "payload_hash": resp.payload_hash,
        "update_id": resp.update_id,
        "base_parameter_version": resp.base_parameter_version,
        "target_parameter_version": resp.target_parameter_version,
        "parameter_version": resp.parameter_version,
        "n_samples": resp.n_samples,
        "loss": resp.loss,
        "round_ready": resp.round_ready,
        "ready": resp.ready,
        "accepted": resp.accepted,
        "schema_hash": resp.schema_hash,
        "parameter_hash": resp.parameter_hash,
        "fingerprint": resp.fingerprint,
        "global_step": resp.global_step,
        "tensors": blobs_to_encoded(resp.tensors, as_arrays=True),
    }
    if resp.parameters:
        reply["parameters"] = blobs_to_encoded(resp.parameters, as_arrays=True)
        reply["schema"] = [{"name": name, "shape": list(value.shape), "dtype": "float32"}
                           for name, value in sorted(reply["parameters"].items())]
    if resp.optimizer.impl:
        reply["optimizer"] = {
            "impl": resp.optimizer.impl,
            "impl_path": resp.optimizer.impl_path,
            "learning_rate": resp.optimizer.learning_rate,
            "t": resp.optimizer.t,
            "beta_1": resp.optimizer.beta_1,
            "beta_2": resp.optimizer.beta_2,
            "eps": resp.optimizer.eps,
        }
    return reply


class _PsServicer(ps_pb2_grpc.ParameterServiceServicer):
    def __init__(self, engine):
        self.engine = engine

    def _handle(self, request, default_type):
        message = ps_request_to_dict(request)
        message["type"] = message.get("type") or default_type
        message["_grpc_wire_arrays"] = True
        # Let the server-side state machine wait briefly for the peer of the
        # same synchronous round.  This removes the usual 1 ms polling RPC
        # loop while preserving the caller's deadline and retry semantics.
        return dict_to_ps_response(self.engine.handle(message, wait_s=0.05))

    def VariableWeightsInit(self, request, context):
        return self._handle(request, "init")

    def Push(self, request, context):
        return self._handle(request, "push")

    def Pull(self, request, context):
        return self._handle(request, "pull")

    def Heartbeat(self, request, context):
        return self._handle(request, "heartbeat")

    def InitialStateApplied(self, request, context):
        return self._handle(request, "initial_state_applied")

    def UpdateApplied(self, request, context):
        return self._handle(request, "update_applied")

    def Stop(self, request, context):
        return self._handle(request, "stop")


class GrpcPsClient:
    def __init__(self, host, port, timeout=30.0):
        self.timeout = timeout
        options = [
            ("grpc.max_send_message_length", DEFAULT_MAX_FRAME),
            ("grpc.max_receive_message_length", DEFAULT_MAX_FRAME),
        ]
        deadline = time.perf_counter() + timeout
        last = None
        while time.perf_counter() < deadline:
            self.channel = grpc.insecure_channel(f"{host}:{int(port)}", options=options)
            try:
                grpc.channel_ready_future(self.channel).result(timeout=0.5)
                last = None
                break
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(0.05)
        if last is not None:
            raise ConnectionError(f"grpc ps connect failed: {last}")
        self.stub = ps_pb2_grpc.ParameterServiceStub(self.channel)
        self.lock = threading.Lock()
        self.traffic = {}

    def call(self, message):
        method_name = _PS_METHODS[message["type"]]
        method = getattr(self.stub, method_name)
        with self.lock:
            from .measurement import account
            started = time.perf_counter()
            req = dict_to_ps_request(message)
            encode_s = time.perf_counter()-started
            started = time.perf_counter()
            resp = method(req, timeout=self.timeout)
            rpc_s = time.perf_counter()-started
            started = time.perf_counter()
            reply = ps_response_to_dict(resp)
            account(self, message['type'], req.ByteSize(), resp.ByteSize(), rpc_s, encode_s,
                    time.perf_counter()-started, sum(len(t.data) for t in req.tensors))
        return reply

    def close(self):
        self.channel.close()


def serve_ps_grpc(engine, port, stop, host=BIND_HOST, timeout=30.0):
    options = [
        ("grpc.max_send_message_length", DEFAULT_MAX_FRAME),
        ("grpc.max_receive_message_length", DEFAULT_MAX_FRAME),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=16), options=options)
    ps_pb2_grpc.add_ParameterServiceServicer_to_server(_PsServicer(engine), server)
    server.add_insecure_port(f"{host}:{int(port)}")
    server.start()
    try:
        while not stop.is_set() and not engine.stopped:
            time.sleep(0.05)
    finally:
        server.stop(grace=0.5)


class RingInbox:
    def __init__(self, lookahead=1, digest_steps=DEFAULT_RING_DIGEST_STEPS):
        self.lookahead = lookahead
        self.digest_steps = max(1, int(digest_steps))
        self.cv = threading.Condition()
        self.buffer = {}
        self.digests = {}
        self._digest_steps = {}
        self.contract = None
        self.current_step = 0

    def configure(self, run_id, rank, workers, schema, chunk_length):
        self.contract = (run_id, rank, workers, schema, chunk_length)

    def put(self, message):
        key = _ring_key(message)
        digest = message.get("payload_hash") or message.get("data")
        step = int(message.get("global_step", 0))
        with self.cv:
            if self.contract:
                run_id, rank, workers, schema, length = self.contract
                stage, round_id = message.get('stage'), int(message.get('round', -1))
                expected = (rank-round_id-1) % workers if stage == 'SCATTER' else (rank-round_id) % workers
                data = message.get('data', b'')
                valid = (message.get('run_id') == run_id and message.get('schema_hash') == schema
                         and message.get('receiver_rank') == rank
                         and message.get('sender_rank') == (rank-1) % workers
                         and stage in ('SCATTER','GATHER') and 0 <= round_id < workers-1
                         and message.get('chunk_id') == expected and message.get('valid_length') == length
                         and len(data) == 4*length and hashlib.sha256(data).hexdigest() == message.get('payload_hash')
                         and self.current_step <= step <= self.current_step + self.lookahead
                         and message.get('collective_id') == f'{run_id}:{step}')
                if not valid:
                    return {'status':'INVALID_MESSAGE', 'error':'ring message violates topology, step, shape or hash',
                            'request_id':message.get('request_id')}
            if key in self.digests:
                if self.digests[key] != digest:
                    return {"status": "DUPLICATE_CONFLICT", "error": "conflicting ring message",
                            "request_id": message.get("request_id"), "duplicate": True}
                return {"status": STATUS_OK, "error": "", "request_id": message.get("request_id"),
                        "duplicate": True}
            self.digests[key] = digest
            self._digest_steps[key] = step
            self.buffer[key] = message
            self._gc_digests(step)
            self.cv.notify_all()
            return {"status": STATUS_OK, "error": "", "request_id": message.get("request_id"),
                    "duplicate": False}

    def wait(self, key, timeout):
        deadline = time.monotonic() + timeout
        with self.cv:
            while key not in self.buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"ring message {key} timed out")
                self.cv.wait(remaining)
            return self.buffer.pop(key)

    def payload_bytes(self):
        total = 0
        for message in self.buffer.values():
            data = message.get("data") or b""
            total += len(data) if isinstance(data, (bytes, bytearray, memoryview)) else 0
        return int(total)

    def buffered_count(self):
        return len(self.buffer)

    def _gc_digests(self, current_step):
        cutoff = int(current_step) - self.digest_steps
        for key, step in list(self._digest_steps.items()):
            if step < cutoff and key not in self.buffer:
                self.digests.pop(key, None)
                self._digest_steps.pop(key, None)


def _ring_key(message):
    return (
        message.get("collective_id"),
        message.get("stage"),
        int(message.get("round", 0)),
        int(message.get("chunk_id", 0)),
        int(message.get("sender_rank", 0)),
        int(message.get("global_step", 0)),
    )


class _RingServicer(ring_pb2_grpc.RingAllReduceServiceServicer):
    def __init__(self, inbox, init_handler):
        self.inbox = inbox
        self.init_handler = init_handler

    def VariableWeightsInit(self, request, context):
        reply = self.init_handler({
            "protocol_version": request.protocol_version,
            "run_id": request.run_id,
            "sender_rank": request.sender_rank,
            "request_id": request.request_id,
            "schema_hash": request.schema_hash,
            "parameter_hash": request.parameter_hash,
            "parameter_version": request.parameter_version,
            "parameters": blobs_to_encoded(request.parameters),
        })
        return ring_pb2.RingInitResponse(status=reply.get("status", STATUS_OK),
                                         error=reply.get("error", ""),
                                         request_id=request.request_id)

    def Receive(self, request, context):
        message = {
            "protocol_version": request.protocol_version,
            "run_id": request.run_id,
            "collective_id": request.collective_id,
            "global_step": request.global_step,
            "parameter_version": request.parameter_version,
            "sender_rank": request.sender_rank,
            "receiver_rank": request.receiver_rank,
            "stage": request.stage,
            "round": request.round,
            "chunk_id": request.chunk_id,
            "valid_length": request.valid_length,
            "schema_hash": request.schema_hash,
            "payload_hash": request.payload_hash,
            "data": request.data,
            "request_id": request.request_id,
        }
        reply = self.inbox.put(message)
        return ring_pb2.RingReceiveResponse(
            status=reply["status"], error=reply.get("error", ""),
            request_id=request.request_id, duplicate=bool(reply.get("duplicate")),
        )


class GrpcRingClient:
    def __init__(self, host, port, timeout=30.0):
        options = [
            ("grpc.max_send_message_length", DEFAULT_MAX_FRAME),
            ("grpc.max_receive_message_length", DEFAULT_MAX_FRAME),
        ]
        deadline = time.perf_counter() + timeout
        last = None
        while time.perf_counter() < deadline:
            self.channel = grpc.insecure_channel(f"{host}:{int(port)}", options=options)
            try:
                grpc.channel_ready_future(self.channel).result(timeout=0.5)
                last = None
                break
            except Exception as exc:  # noqa: BLE001
                last = exc
                time.sleep(0.05)
        if last is not None:
            raise ConnectionError(f"ring connect failed: {last}")
        self.stub = ring_pb2_grpc.RingAllReduceServiceStub(self.channel)
        self.timeout = timeout
        self.traffic = {}

    def receive(self, message):
        from .measurement import account
        started = time.perf_counter()
        req = ring_pb2.RingReceiveRequest(
            protocol_version=message.get("protocol_version", PROTOCOL_VERSION),
            run_id=message["run_id"],
            collective_id=message["collective_id"],
            global_step=int(message["global_step"]),
            parameter_version=int(message.get("parameter_version") or 0),
            sender_rank=int(message["sender_rank"]),
            receiver_rank=int(message["receiver_rank"]),
            stage=message["stage"],
            round=int(message["round"]),
            chunk_id=int(message["chunk_id"]),
            valid_length=int(message.get("valid_length") or 0),
            schema_hash=str(message.get("schema_hash") or ""),
            payload_hash=str(message.get("payload_hash") or ""),
            data=message["data"],
            request_id=str(message.get("request_id") or ""),
        )
        encode_s = time.perf_counter()-started
        started = time.perf_counter()
        reply = self.stub.Receive(req, timeout=self.timeout)
        account(self, message['stage'], req.ByteSize(), reply.ByteSize(),
                time.perf_counter()-started, encode_s, payload=len(req.data))
        if reply.status != STATUS_OK:
            raise RuntimeError(f'Ring receive rejected: {reply.status}: {reply.error}')
        return reply

    def init(self, message):
        req = ring_pb2.RingInitRequest(
            protocol_version=message.get("protocol_version", PROTOCOL_VERSION),
            run_id=message["run_id"],
            sender_rank=int(message["sender_rank"]),
            request_id=str(message.get("request_id") or ""),
            schema_hash=str(message.get("schema_hash") or ""),
            parameter_hash=str(message.get("parameter_hash") or ""),
            parameter_version=int(message.get("parameter_version") or 0),
        )
        req.parameters.extend(encoded_to_blobs(message.get("parameters")))
        from .measurement import account
        started = time.perf_counter()
        reply = self.stub.VariableWeightsInit(req, timeout=self.timeout)
        account(self, 'init', req.ByteSize(), reply.ByteSize(), time.perf_counter()-started)
        return reply

    def close(self):
        self.channel.close()


def serve_ring_grpc(inbox, init_handler, port, stop, host=BIND_HOST):
    options = [
        ("grpc.max_send_message_length", DEFAULT_MAX_FRAME),
        ("grpc.max_receive_message_length", DEFAULT_MAX_FRAME),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8), options=options)
    ring_pb2_grpc.add_RingAllReduceServiceServicer_to_server(_RingServicer(inbox, init_handler), server)
    server.add_insecure_port(f"{host}:{int(port)}")
    server.start()
    try:
        while not stop.is_set():
            time.sleep(0.05)
    finally:
        server.stop(grace=0.5)
