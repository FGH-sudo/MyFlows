"""MNIST MLP adapter. Official digits are loaded via sklearn OpenML when needed."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ...core.device import asnumpy, xp
from ...core.graph import Graph
from ...core.node import Variable
from ...layers.layer import Dense
from ...ops.activation import ReLU
from ...ops.loss import CrossEntropy
from ...utils.initializers import make_initializer

OFFICIAL_MNIST_TRAIN = 60000
OFFICIAL_MNIST_TEST = 10000


def split_official_mnist(x, y, n_train, n_val, seed):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.shape[0] < OFFICIAL_MNIST_TRAIN + OFFICIAL_MNIST_TEST:
        raise ValueError("MNIST fixture must contain the official 70000 samples")
    n_train = int(n_train)
    n_val = int(n_val)
    if n_train < 0 or n_val < 0:
        raise ValueError("n_train and n_val must be non-negative")
    if n_train + n_val > OFFICIAL_MNIST_TRAIN:
        raise ValueError("n_train and n_val cannot exceed the official MNIST training split")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(OFFICIAL_MNIST_TRAIN)
    train_idx = np.asarray(order[:n_train], dtype=np.int64)
    val_idx = np.asarray(order[n_train:n_train + n_val], dtype=np.int64)
    test_idx = np.arange(OFFICIAL_MNIST_TRAIN, OFFICIAL_MNIST_TRAIN + OFFICIAL_MNIST_TEST, dtype=np.int64)
    split = {
        "train": (x[train_idx], y[train_idx]),
        "val": (x[val_idx], y[val_idx]),
        "test": (x[test_idx], y[test_idx]),
    }
    fingerprint = hashlib.sha256(train_idx.tobytes() + val_idx.tobytes() + test_idx.tobytes()).hexdigest()
    meta = {
        "source": "mnist_784",
        "official_train": OFFICIAL_MNIST_TRAIN,
        "official_test": OFFICIAL_MNIST_TEST,
        "n_train": n_train,
        "n_val": n_val,
        "n_test": int(test_idx.size),
        "seed": int(seed),
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
        "split_fingerprint": fingerprint,
    }
    return split, meta


class MnistMlpTask:
    name = "mnist_mlp"
    kind = "classification"

    def __init__(self):
        self.epoch_index_builds = 0

    def build_model(self, config, device="cpu"):
        seed = int(config.get("seed", 0))
        hidden = int(config.get("hidden", 64))
        init = make_initializer(seed=seed)
        dtype = np.float32
        x = Variable(xp.zeros((1, 784), dtype), name="input")
        y = Variable(xp.zeros((1, 1), np.int64), name="target")
        fc1 = Dense(784, hidden, activation=ReLU, initializer=init, dtype=dtype, name="fc1")
        fc2 = Dense(hidden, 10, initializer=init, dtype=dtype, name="fc2")
        logits = fc2(fc1(x))
        loss = CrossEntropy(logits, y)
        graph = Graph(loss, optimize=False)
        params = {
            "fc1.weight": fc1.W,
            "fc1.bias": fc1.b,
            "fc2.weight": fc2.W,
            "fc2.bias": fc2.b,
        }
        return {"x": x, "y": y, "logits": logits, "loss": loss, "graph": graph,
                "params": params, "kind": "classification"}

    def make_data(self, config):
        if config.get("prepared_dir"):
            from .prepared import load_prepared
            return load_prepared(config["prepared_dir"])
        if config.get("split_arrays"):
            from ..data_meta import ensure_split_meta
            split = {name: tuple(value) for name, value in config["split_arrays"].items()}
            split["meta"] = dict(config.get("split_meta") or {"source": "injected"})
            return ensure_split_meta(split, config)
        from sklearn.datasets import fetch_openml

        cache = Path(config.get("data_dir") or Path.home() / ".cache" / "myflows-mnist")
        cache.mkdir(parents=True, exist_ok=True)
        bundle = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto", data_home=str(cache))
        x = bundle.data.astype(np.float32) / 255.0
        y = bundle.target.astype(np.int64).reshape(-1, 1)
        n_train = int(config.get("n_train", 50000))
        n_val = int(config.get("n_val", 10000))
        split, meta = split_official_mnist(x, y, n_train=n_train, n_val=n_val, seed=int(config.get("seed", 0)))
        split["meta"] = meta
        from ..data_meta import ensure_split_meta
        return ensure_split_meta(split, config)

    def epoch_index(self, split, config, epoch=0, split_name="train"):
        self.epoch_index_builds += 1
        x, y = split[split_name]
        batch = int(config.get("global_batch", 128))
        drop_last = bool(config.get("drop_last", True))
        n = (len(x) // batch) * batch if drop_last else len(x)
        rng = np.random.default_rng(int(config.get("seed", 0)) + 17 + int(epoch))
        order = rng.permutation(len(x))[:n] if split_name == "train" else np.arange(n)
        return [order[start:start + batch] for start in range(0, n, batch)]

    def take_batch(self, split, indices, split_name="train"):
        x, y = split[split_name]
        return x[indices], y[indices]

    def epoch_batches(self, split, config, split_name="train"):
        batches = []
        for index in self.epoch_index(split, config, epoch=0, split_name=split_name):
            batches.append(self.take_batch(split, index, split_name=split_name))
        return batches

    def shard_arrays(self, batch, _step, indices):
        x, y = batch
        return x[indices], y[indices]

    def metrics(self, session, x_cpu, y_cpu):
        from ..metrics_reduce import values_from_stats
        return values_from_stats(self.metric_stats(session, x_cpu, y_cpu))

    def metric_stats(self, session, x_cpu, y_cpu):
        logits = asnumpy(session.logits.value, copy=False)
        pred = np.argmax(logits, axis=1)
        label = np.asarray(y_cpu).reshape(-1)
        n = int(label.size)
        correct = float(np.sum(pred == label))
        return {
            "accuracy": {"sum": correct, "count": n, "kind": "mean"},
        }
