"""Locked stage-one fixtures shared by correctness tests and experiment CLIs."""

import numpy as np


CONV_CASES = (
    ("T0", (1, 1, 5, 5), (1, 1, 3, 3), (1, 1), (0, 0)),
    ("T1", (2, 3, 7, 9), (4, 3, 3, 3), (2, 2), (1, 1)),
    ("T2", (1, 2, 6, 8), (3, 2, 1, 1), (1, 1), (0, 0)),
    ("T3", (1, 3, 16, 16), (4, 3, 7, 7), (2, 2), (3, 3)),
    ("T4", (1, 2, 7, 9), (3, 2, 2, 3), (1, 2), (0, 1)),
)


def conv_fixture(case, seed=0):
    name, x_shape, w_shape, stride, padding = case
    rng = np.random.default_rng(seed)
    shape = (x_shape[0], w_shape[0],
             (x_shape[2] + 2 * padding[0] - w_shape[2]) // stride[0] + 1,
             (x_shape[3] + 2 * padding[1] - w_shape[3]) // stride[1] + 1)
    arrays = [rng.normal(size=s).astype(np.float32) for s in (x_shape, w_shape, (w_shape[0],), shape)]
    if name == "T0":
        arrays[0] = np.arange(25, dtype=np.float32).reshape(x_shape)
        arrays[1] = np.ones(w_shape, dtype=np.float32)
        arrays[2] = np.zeros(w_shape[0], dtype=np.float32)
    return arrays
