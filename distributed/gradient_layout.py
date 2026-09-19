"""Flatten named gradients and execute the two-phase Ring AllReduce chunk flow."""

from __future__ import annotations

import math

import numpy as np

from .protocol import schema_from_arrays


def build_layout(schema):
    entries = []
    offset = 0
    for item in sorted(schema, key=lambda row: row["name"]):
        shape = tuple(int(v) for v in item["shape"])
        numel = int(np.prod(shape)) if shape else 1
        entries.append({
            "name": item["name"],
            "shape": shape,
            "dtype": "float32",
            "offset": offset,
            "numel": numel,
        })
        offset += numel
    return {"entries": entries, "numel": offset}


def flatten_gradients(arrays, layout, out=None):
    flat = (np.zeros(layout["numel"], dtype=np.float32) if out is None
            else np.asarray(out, dtype=np.float32).reshape(-1))
    if flat.size != layout["numel"]:
        raise ValueError("gradient output buffer has the wrong size")
    for entry in layout["entries"]:
        value = np.ascontiguousarray(arrays[entry["name"]], dtype=np.float32).reshape(-1)
        if value.size != entry["numel"]:
            raise ValueError(f"gradient numel mismatch for {entry['name']}")
        start, stop = entry["offset"], entry["offset"] + entry["numel"]
        flat[start:stop] = value
    return flat


def restore_gradients(flat, layout, *, copy=True):
    arrays = {}
    vec = np.ascontiguousarray(flat, dtype=np.float32).reshape(-1)
    if vec.size < layout["numel"]:
        raise ValueError("reduced vector is shorter than layout")
    for entry in layout["entries"]:
        start, stop = entry["offset"], entry["offset"] + entry["numel"]
        view = vec[start:stop].reshape(entry["shape"])
        arrays[entry["name"]] = view.copy() if copy else view
    return arrays


def padded_chunks(vector, n_ranks):
    n_ranks = int(n_ranks)
    if n_ranks <= 0:
        raise ValueError("n_ranks must be positive")
    vec = np.ascontiguousarray(vector, dtype=np.float32).reshape(-1)
    chunk = int(math.ceil(vec.size / n_ranks)) if vec.size else 0
    padded_size = n_ranks * chunk
    padded = np.zeros(padded_size, dtype=np.float32)
    padded[:vec.size] = vec
    chunks = padded.reshape(n_ranks, chunk) if chunk else np.zeros((n_ranks, 0), dtype=np.float32)
    return chunks, padded_size - vec.size


def scatter_reduce_apply(local_chunks, recv_chunk, rank, round_id, n_ranks, *, inplace=False):
    out = (np.asarray(local_chunks, dtype=np.float32)
           if inplace else np.array(local_chunks, dtype=np.float32, copy=True))
    recv_id = (int(rank) - int(round_id) - 1) % int(n_ranks)
    out[recv_id] += np.asarray(recv_chunk, dtype=np.float32)
    return out


def allgather_apply(local_chunks, recv_chunk, rank, round_id, n_ranks, *, inplace=False):
    out = (np.asarray(local_chunks, dtype=np.float32)
           if inplace else np.array(local_chunks, dtype=np.float32, copy=True))
    recv_id = (int(rank) - int(round_id)) % int(n_ranks)
    out[recv_id] = np.asarray(recv_chunk, dtype=np.float32)
    return out


def scatter_send_chunk_id(rank, round_id, n_ranks):
    return (int(rank) - int(round_id)) % int(n_ranks)


def allgather_send_chunk_id(rank, round_id, n_ranks):
    return (int(rank) + 1 - int(round_id)) % int(n_ranks)


def simulate_ring_allreduce(vectors):
    n_ranks = len(vectors)
    if n_ranks == 0:
        return []
    prepared = [np.ascontiguousarray(vector, dtype=np.float32).reshape(-1) for vector in vectors]
    if n_ranks == 1:
        return [prepared[0].copy()]
    states = []
    sizes = []
    for vector in prepared:
        chunks, _pad = padded_chunks(vector, n_ranks)
        states.append(np.array(chunks, copy=True))
        sizes.append(vector.size)
    for round_id in range(n_ranks - 1):
        incoming = []
        for rank in range(n_ranks):
            left = (rank - 1) % n_ranks
            send_id = scatter_send_chunk_id(left, round_id, n_ranks)
            incoming.append(states[left][send_id].copy())
        states = [scatter_reduce_apply(states[rank], incoming[rank], rank, round_id, n_ranks)
                  for rank in range(n_ranks)]
    for round_id in range(n_ranks - 1):
        incoming = []
        for rank in range(n_ranks):
            left = (rank - 1) % n_ranks
            send_id = allgather_send_chunk_id(left, round_id, n_ranks)
            incoming.append(states[left][send_id].copy())
        states = [allgather_apply(states[rank], incoming[rank], rank, round_id, n_ranks)
                  for rank in range(n_ranks)]
    return [states[rank].reshape(-1)[:sizes[rank]].copy() for rank in range(n_ranks)]


def layout_from_arrays(arrays):
    return build_layout(schema_from_arrays(arrays))
