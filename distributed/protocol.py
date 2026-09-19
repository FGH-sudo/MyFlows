"""JSON/FP32 tensor encoding, length-prefixed frames, hashes and weighted aggregation."""

from __future__ import annotations

import hashlib
import json
import struct

import numpy as np

from .constants import DEFAULT_MAX_FRAME, PROTOCOL_VERSION

__all__ = [
    "PROTOCOL_VERSION",
    "schema_from_arrays",
    "schema_hash",
    "payload_hash",
    "payload_bytes",
    "clone_arrays",
    "encode_tensors",
    "decode_tensors",
    "pack_frame",
    "unpack_frame",
    "recv_frame_from_buffer",
    "aggregate_weighted",
    "make_update_id",
    "digest_state",
]


def schema_from_arrays(arrays):
    return [{"name": name, "shape": list(np.asarray(value).shape), "dtype": "float32"}
            for name, value in sorted(arrays.items())]


def schema_hash(arrays_or_schema):
    if isinstance(arrays_or_schema, dict):
        spec = schema_from_arrays(arrays_or_schema)
    else:
        spec = list(arrays_or_schema)
    return hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _as_fp32(value, name="tensor"):
    array = np.ascontiguousarray(value)
    if array.dtype != np.float32:
        array = array.astype(np.float32, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"non-finite array {name}")
    return array


def payload_hash(arrays):
    schema = schema_from_arrays(arrays)
    sha = hashlib.sha256(schema_hash(schema).encode())
    for name, value in sorted(arrays.items()):
        array = _as_fp32(value, name)
        sha.update(name.encode())
        sha.update(array.tobytes())
    return sha.hexdigest()


def payload_bytes(arrays):
    return int(sum(np.asarray(value).nbytes for value in arrays.values()))


def clone_arrays(arrays):
    return {name: np.array(value, copy=True) for name, value in arrays.items()}


def encode_tensors(arrays):
    items = []
    for name, value in sorted(arrays.items()):
        array = _as_fp32(value, name)
        items.append({
            "name": name,
            "shape": list(array.shape),
            "dtype": "float32",
            "data": array.reshape(-1).tolist(),
        })
    return items


def decode_tensors(items, schema):
    expected = {entry["name"]: entry for entry in schema}
    # gRPC keeps tensor payloads as protobuf bytes.  Accepting a name->array
    # mapping lets that path avoid materializing every FP32 element as a
    # Python float before reconstructing the NumPy array.
    if isinstance(items, dict):
        if set(items) != set(expected):
            raise ValueError("tensor names differ from schema")
        iterable = ((name, {"name": name, "shape": expected[name]["shape"],
                            "dtype": expected[name]["dtype"], "data": value})
                    for name, value in items.items())
    elif isinstance(items, list):
        if {item.get("name") for item in items} != set(expected):
            raise ValueError("tensor names differ from schema")
        iterable = ((item["name"], item) for item in items)
    else:
        raise ValueError("tensor payload must be a list or mapping")
    arrays = {}
    for name, item in iterable:
        spec = expected[name]
        array = np.asarray(item.get("data"), dtype=np.float32)
        shape = tuple(spec["shape"])
        if int(array.size) != int(np.prod(shape)) or list(item.get("shape", [])) != list(shape):
            raise ValueError(f"invalid shape for {name}")
        if item.get("dtype") not in ("float32", "float32", spec["dtype"]):
            raise ValueError(f"invalid dtype for {name}")
        array = array.reshape(shape)
        if not np.isfinite(array).all():
            raise ValueError(f"non-finite array {name}")
        array = np.ascontiguousarray(array)
        # A protobuf bytes field is exposed through a read-only NumPy view;
        # model parameters and optimizer buffers must be writable after
        # initialization.
        arrays[name] = array.copy() if not array.flags.writeable else array
    return arrays


def pack_frame(obj, max_bytes=DEFAULT_MAX_FRAME):
    body = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    if len(body) > int(max_bytes):
        raise ValueError(f"JSON frame {len(body)} exceeds {max_bytes}")
    return struct.pack("!I", len(body)) + body


def unpack_frame(frame):
    if len(frame) < 4:
        raise ValueError("truncated frame header")
    length = struct.unpack("!I", frame[:4])[0]
    body = frame[4:]
    if length != len(body):
        raise ValueError("frame length mismatch")
    return json.loads(body.decode("utf-8"))


def recv_frame_from_buffer(buffer, max_bytes=DEFAULT_MAX_FRAME):
    buf = bytearray(buffer)
    if len(buf) < 4:
        return None, buf
    length = struct.unpack("!I", buf[:4])[0]
    if length > int(max_bytes):
        raise ValueError(f"frame length {length} exceeds {max_bytes}")
    if len(buf) < 4 + length:
        return None, buf
    return bytes(buf[:4 + length]), buf[4 + length:]


def aggregate_weighted(messages):
    if not messages:
        raise ValueError("no gradient contributions")
    total = sum(int(m["n_samples"]) for m in messages)
    if total <= 0 or any(int(m["n_samples"]) <= 0 for m in messages):
        raise ValueError("shards must be nonempty")
    first = messages[0]["gradients"]
    gradients = {name: np.zeros_like(value, dtype=np.float32) for name, value in first.items()}
    loss = 0.0
    for message in messages:
        weight = int(message["n_samples"]) / total
        loss += float(message["loss"]) * weight
        for name in gradients:
            gradients[name] += np.asarray(message["gradients"][name], dtype=np.float32) * np.float32(weight)
    return gradients, float(loss), int(total)


def make_update_id(run_id, global_step, base_parameter_version):
    return f"{run_id}:{int(global_step)}:{int(base_parameter_version)}"


def digest_state(parameters, optimizer_state=None, step=None):
    payload = dict(parameters)
    if optimizer_state:
        for name, value in sorted(optimizer_state.items()):
            if name in ("t", "impl", "learning_rate", "beta_1", "beta_2", "eps"):
                continue
            payload[f"opt.{name}"] = value
    digest = payload_hash(payload)
    extra = json.dumps({
        "t": None if optimizer_state is None else optimizer_state.get("t"),
        "step": step,
    }, sort_keys=True)
    return hashlib.sha256((digest + extra).encode()).hexdigest()
