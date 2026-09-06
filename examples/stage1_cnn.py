"""Locked 32-sample FP32 CNN fixture for CUDA/CuPy training equivalence."""

import numpy as np

from ..core.device import xp
from ..core.graph import Graph
from ..core.node import Variable
from ..layers.layer import Conv2D, Dense, Flatten, MaxPool2d
from ..ops.activation import ReLU
from ..ops.loss import CrossEntropy
from ..train.opt import Adam
from ..utils.initializers import make_initializer


def dataset(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.05, (32, 1, 8, 8)).astype(np.float32)
    y = np.arange(32, dtype=np.int64) % 2
    x[y == 0, :, :, 1:3] += 1
    x[y == 1, :, 1:3, :] += 1
    return x, y.reshape(-1, 1)


def build(backend, seed=0):
    x_data, y_data = dataset(seed)
    x, y = Variable(xp.asarray(x_data), name="images"), Variable(xp.asarray(y_data), name="labels")
    init = make_initializer(seed=seed)
    conv = Conv2D(1, 4, kernel_size=3, padding=1, fuse_activation=False,
                  initializer=init, dtype=np.float32, backend=backend, name="conv")
    pool = MaxPool2d(2, 2, backend=backend)
    fc1 = Dense(64, 16, activation=ReLU, initializer=init, dtype=np.float32, name="fc1")
    fc2 = Dense(16, 2, initializer=init, dtype=np.float32, name="fc2")
    logits = fc2(fc1(Flatten()(pool(ReLU(conv(x))))))
    loss = CrossEntropy(logits, y)
    graph = Graph(loss, optimize=False)
    optimizer = Adam(graph, learning_rate=0.01)
    return {"x": x, "y": y, "logits": logits, "loss": loss, "graph": graph,
            "optimizer": optimizer, "params": conv.params + fc1.params + fc2.params}


def assert_fp32(model):
    for node in model["graph"].nodes:
        if node is model["y"]:
            continue
        for name in ("value", "grad"):
            array = getattr(node, name)
            if array is not None and array.dtype != np.float32:
                raise AssertionError(f"{node.name}.{name} has dtype {array.dtype}")
    opt = model["optimizer"]
    for state in (opt.acc_gradient, opt.v, opt.s):
        for value in state.values():
            if value.dtype != np.float32:
                raise AssertionError(f"optimizer state has dtype {value.dtype}")
