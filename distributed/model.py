"""Fixed FP64 MLP, named parameters and shared batch manifest for the PS demo."""

import numpy as np

from ..core.device import set_device
from ..core.graph import Graph
from ..core.node import Variable
from ..layers.layer import Dense
from ..ops.activation import ReLU
from ..ops.loss import MSELoss
from ..train.opt import MBGD
from ..utils.initializers import make_initializer


def make_batches(seed, global_batch, steps):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(steps, global_batch, 4)).astype(np.float64)
    target_weight = np.array([[0.5, -0.3], [0.2, 0.8], [-0.6, 0.1], [0.4, 0.7]])
    y = x @ target_weight + 0.1 * np.sin(x[..., :2])
    return x, y


class SmallMLP:
    def __init__(self, seed=0):
        set_device("cpu")
        init = make_initializer(seed=seed)
        self.x = Variable(np.zeros((1, 4), np.float64), name="input")
        self.y = Variable(np.zeros((1, 2), np.float64), name="target")
        fc1 = Dense(4, 8, activation=ReLU, initializer=init, dtype=np.float64, name="fc1")
        fc2 = Dense(8, 2, initializer=init, dtype=np.float64, name="fc2")
        self.params = {"fc1.weight": fc1.W, "fc1.bias": fc1.b,
                       "fc2.weight": fc2.W, "fc2.bias": fc2.b}
        if len({id(p) for p in self.params.values()}) != len(self.params):
            raise AssertionError("parameter mapping must be unique")
        self.loss = MSELoss(fc2(fc1(self.x)), self.y)
        self.graph = Graph(self.loss, optimize=False)
        self.optimizer = MBGD(self.graph, learning_rate=0.01)

    def snapshot(self):
        return {name: node.value.copy() for name, node in self.params.items()}

    def load(self, params):
        if set(params) != set(self.params):
            raise ValueError("parameter names differ")
        for name, node in self.params.items():
            value = params[name]
            if value.shape != node.value.shape or value.dtype != np.float64 or not np.isfinite(value).all():
                raise ValueError(f"invalid parameter {name}")
        for name, node in self.params.items():
            node.value = params[name].copy()

    def gradients(self, x, y):
        self.x.value, self.y.value = x, y
        self.graph.forward()
        self.graph.backward()
        return float(self.loss.value), {name: node.grad.copy() for name, node in self.params.items()}

    def update(self, gradients):
        if self.optimizer.acc_no != 0 or self.optimizer.acc_gradient:
            raise RuntimeError("server optimizer cache must be empty before external gradients")
        self.optimizer.update({self.params[name]: gradient for name, gradient in gradients.items()})


def single_process(seed=0, global_batch=32, steps=20):
    model = SmallMLP(seed)
    batches = make_batches(seed, global_batch, steps)
    history = [model.snapshot()]
    losses = []
    for x, y in zip(*batches):
        loss, grads = model.gradients(x, y)
        model.update(grads)
        history.append(model.snapshot())
        losses.append(loss)
    return {"history": history, "losses": losses}
