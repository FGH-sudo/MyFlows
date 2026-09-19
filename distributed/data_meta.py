"""Lightweight, comparable data-split metadata for launcher/manifest."""

from __future__ import annotations

import hashlib

import numpy as np

META_FIELDS = ("source", "n_train", "n_val", "n_test", "seed", "split_fingerprint")
DEFAULT_FINGERPRINT_CHUNK_BYTES = 1 << 20
SPLIT_NAMES = ("train", "val", "test")


def fingerprint_arrays(*arrays, chunk_bytes=DEFAULT_FINGERPRINT_CHUNK_BYTES):
    sha = hashlib.sha256()
    for index, array in enumerate(arrays):
        sha.update(b"array=")
        sha.update(str(index).encode())
        sha.update(b"\0")
        _hash_array(sha, array, chunk_bytes)
    return sha.hexdigest()


def fingerprint_split(split, chunk_bytes=DEFAULT_FINGERPRINT_CHUNK_BYTES):
    sha = hashlib.sha256()
    for name in SPLIT_NAMES:
        sha.update(b"split=")
        sha.update(name.encode())
        sha.update(b"\0")
        part = (split or {}).get(name)
        if not part:
            sha.update(b"missing\0")
            continue
        sha.update(b"features\0")
        _hash_array(sha, part[0], chunk_bytes)
        sha.update(b"labels\0")
        _hash_array(sha, part[1], chunk_bytes)
    return sha.hexdigest()


def _hash_array(sha, array, chunk_bytes):
    array = np.asarray(array)
    sha.update(b"shape=")
    sha.update(str(tuple(array.shape)).encode())
    sha.update(b"\0dtype=")
    sha.update(str(array.dtype).encode())
    sha.update(b"\0")
    if array.size == 0:
        sha.update(b"empty\0")
        return
    itemsize = int(array.dtype.itemsize) or 1
    chunk_bytes = max(int(chunk_bytes), itemsize)
    if array.flags.c_contiguous:
        _hash_contiguous(sha, array, chunk_bytes)
        return
    _hash_buffered_c_order(sha, array, chunk_bytes)


def _hash_contiguous(sha, array, chunk_bytes):
    raw = array.reshape(-1).view(np.uint8)
    total = int(raw.size)
    for start in range(0, total, chunk_bytes):
        sha.update(memoryview(raw[start:start + chunk_bytes]))


def _hash_buffered_c_order(sha, array, chunk_bytes):
    itemsize = int(array.dtype.itemsize) or 1
    buffersize = max(1, int(chunk_bytes) // itemsize)
    iterator = np.nditer(
        array,
        flags=["buffered", "external_loop"],
        op_flags=["readonly"],
        order="C",
        buffersize=buffersize,
    )
    with iterator:
        for chunk in iterator:
            payload = np.asarray(chunk)
            if not payload.flags.c_contiguous:
                payload = np.ascontiguousarray(payload)
            _hash_contiguous(sha, payload, chunk_bytes)


def compact_meta(meta):
    payload = {}
    for key in META_FIELDS:
        value = (meta or {}).get(key)
        if hasattr(value, "item"):
            value = value.item()
        payload[key] = value
    return payload


def ensure_split_meta(split, config):
    meta = dict((split or {}).get("meta") or {})
    for name in SPLIT_NAMES:
        key = f"n_{name}"
        if key not in meta and name in (split or {}) and split[name] is not None:
            meta[key] = int(np.asarray(split[name][0]).shape[0])
    meta.setdefault("seed", int(config.get("seed", 0)))
    meta.setdefault("source", meta.get("source") or config.get("data_source") or "injected")
    if not meta.get("split_fingerprint"):
        meta["split_fingerprint"] = fingerprint_split(
            split, chunk_bytes=int(config.get("fingerprint_chunk_bytes") or DEFAULT_FINGERPRINT_CHUNK_BYTES))
        meta["fingerprint_source"] = "auto"
    else:
        meta.setdefault("fingerprint_source", "external")
    split = dict(split or {})
    split["meta"] = meta
    return split


def lightweight_data_meta(data, config):
    if isinstance(data, dict):
        data = ensure_split_meta(data, config)
        return compact_meta(data.get("meta"))
    x, y = data
    chunk = int(config.get("fingerprint_chunk_bytes") or DEFAULT_FINGERPRINT_CHUNK_BYTES)
    return compact_meta({
        "source": "synthetic",
        "n_train": int(np.asarray(x).shape[0]) * int(np.asarray(x).shape[1]),
        "n_val": 0,
        "n_test": 0,
        "seed": int(config.get("seed", 0)),
        "split_fingerprint": fingerprint_arrays(x, y, chunk_bytes=chunk),
    })


def merge_data_meta(metas, n_workers):
    metas = [compact_meta(item) for item in metas or []]
    if len(metas) != int(n_workers):
        raise ValueError(f"data_meta missing: got {len(metas)} worker records, expected {n_workers}")
    first = metas[0]
    if not first.get("split_fingerprint"):
        raise ValueError("data_meta split_fingerprint missing")
    for index, item in enumerate(metas[1:], 1):
        if item != first:
            raise ValueError(f"data_meta mismatch between worker 0 and {index}: {first} vs {item}")
    return first
