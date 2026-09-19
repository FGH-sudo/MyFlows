"""Tiny FP32 MLP fixture used for protocol and numeric equivalence tests."""

import numpy as np

from ...core.device import xp
from ...core.graph import Graph
from ...core.node import Variable
from ...layers.layer import Dense
from ...ops.activation import ReLU
from ...ops.loss import MSELoss
from ...utils.initializers import make_initializer


def make_batches(seed, global_batch, steps):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(steps, global_batch, 4)).astype(np.float32)
    target_weight = np.array([[0.5, -0.3], [0.2, 0.8], [-0.6, 0.1], [0.4, 0.7]], dtype=np.float32)
    y = x @ target_weight + np.float32(0.1) * np.sin(x[..., :2]).astype(np.float32)
    return x, y


class SyntheticTask:
    name = "synthetic"
    kind = "regression"

    def __init__(self):
        self.epoch_index_builds = 0

    def build_model(self, config, device="cpu"):
        seed = int(config.get("seed", 0))
        init = make_initializer(seed=seed)
        dtype = np.float32
        x = Variable(xp.zeros((1, 4), dtype), name="input")
        y = Variable(xp.zeros((1, 2), dtype), name="target")
        fc1 = Dense(4, 8, activation=ReLU, initializer=init, dtype=dtype, name="fc1")
        fc2 = Dense(8, 2, initializer=init, dtype=dtype, name="fc2")
        logits = fc2(fc1(x))
        loss = MSELoss(logits, y)
        graph = Graph(loss, optimize=False)
        params = {
            "fc1.weight": fc1.W,
            "fc1.bias": fc1.b,
            "fc2.weight": fc2.W,
            "fc2.bias": fc2.b,
        }
        return {"x": x, "y": y, "logits": logits, "loss": loss, "graph": graph,
                "params": params, "kind": "regression"}

    def make_data(self, config):
        steps = int(config.get("steps", 20))
        batch = int(config.get("global_batch", 32))
        return make_batches(int(config.get("seed", 0)), batch, steps)

    def shard_arrays(self, data, step, indices):
        x, y = data
        if indices is None:
            return x[step], y[step]
        return x[step, indices], y[step, indices]

    def metrics(self, session, x_cpu, y_cpu):
        from ..metrics_reduce import values_from_stats
        return values_from_stats(self.metric_stats(session, x_cpu, y_cpu))

    def metric_stats(self, session, x_cpu, y_cpu):
        from ...core.device import asnumpy
        pred = asnumpy(session.logits.value, copy=False)
        err = np.asarray(pred) - np.asarray(y_cpu)
        n_elem = int(err.size)
        sq = float(np.sum(err ** 2))
        abs_sum = float(np.sum(np.abs(err)))
        return {
            "loss": {"sum": sq, "count": n_elem, "kind": "mean"},
            "mse": {"sum": sq, "count": n_elem, "kind": "mean"},
            "mae": {"sum": abs_sum, "count": n_elem, "kind": "mean"},
            "rmse": {"sum": sq, "count": n_elem, "kind": "rmse"},
        }
