"""Deterministic global-batch sharding without overlapping samples."""

from __future__ import annotations

import numpy as np


def split_global_batch(global_batch, n_workers, shard_sizes=None):
    n_workers = int(n_workers)
    global_batch = int(global_batch)
    if n_workers <= 0 or global_batch < n_workers:
        raise ValueError("batch must give every worker a sample")
    if shard_sizes is None:
        parts = [len(part) for part in np.array_split(np.arange(global_batch), n_workers)]
    else:
        if any(type(size) is not int for size in shard_sizes):
            raise ValueError("shard_sizes must be positive integers")
        parts = list(shard_sizes)
    if len(parts) != n_workers or any(type(size) is not int or size <= 0 for size in parts) or sum(parts) != global_batch:
        raise ValueError("shard_sizes must be positive and sum to global_batch")
    return parts


def split_indices(indices, n_workers, global_batch=None, drop_last=False, shard_sizes=None):
    values = np.asarray(indices)
    dropped = 0
    if drop_last:
        if global_batch is None or int(global_batch) <= 0:
            raise ValueError("drop_last requires a positive global_batch")
        usable = (len(values) // int(global_batch)) * int(global_batch)
        dropped = int(len(values) - usable)
        values = values[:usable]
        sizes = split_global_batch(int(global_batch), n_workers, shard_sizes)
        shards = [[] for _ in range(n_workers)]
        offset = 0
        for start in range(0, len(values), int(global_batch)):
            batch = values[start:start + int(global_batch)]
            cursor = 0
            for worker, size in enumerate(sizes):
                shards[worker].append(batch[cursor:cursor + size])
                cursor += size
            offset += int(global_batch)
        return [np.concatenate(parts) if parts else np.zeros((0,), dtype=values.dtype) for parts in shards], dropped
    sizes = split_global_batch(len(values), n_workers, shard_sizes) if len(values) else [0] * n_workers
    shards = []
    cursor = 0
    for size in sizes:
        shards.append(values[cursor:cursor + size])
        cursor += size
    return shards, dropped
