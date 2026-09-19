"""Small native-CUDA CNN adapter for DonkeyCar road images."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ...core.device import asnumpy, get_device, xp
from ...core.graph import Graph
from ...core.node import Variable
from ...layers.layer import Conv2D, Dense, Flatten, MaxPool2d
from ...ops.activation import ReLU
from ...ops.loss import MSELoss
from ...utils.initializers import make_initializer


class DonkeyCnnTask:
    name = "donkey_cnn"
    kind = "regression"

    def __init__(self):
        self.epoch_index_builds = 0

    def build_model(self, config, device="cpu"):
        seed = int(config.get("seed", 0))
        init = make_initializer(seed=seed)
        dtype = np.float32
        height = int(config.get("image_h", 120))
        width = int(config.get("image_w", 160))
        backend = "cuda_native_cublas" if get_device() == "cuda" else "numpy"
        conv_backend = config.get("conv_backend")
        if conv_backend not in (None, ""):
            backend = str(conv_backend)
        x = Variable(xp.zeros((1, 3, height, width), dtype), name="input")
        y = Variable(xp.zeros((1, 2), dtype), name="target")
        conv1 = Conv2D(3, 8, kernel_size=5, stride=2, padding=2, fuse_activation=False,
                       initializer=init, dtype=dtype, backend=backend, name="conv1")
        pool1 = MaxPool2d(2, 2, backend=backend)
        conv2 = Conv2D(8, 16, kernel_size=3, stride=2, padding=1, fuse_activation=False,
                       initializer=init, dtype=dtype, backend=backend, name="conv2")
        pool2 = MaxPool2d(2, 2, backend=backend)
        fc1 = Dense(1120, 32, activation=ReLU, initializer=init, dtype=dtype, name="fc1")
        fc2 = Dense(32, 2, initializer=init, dtype=dtype, name="fc2")
        logits = fc2(fc1(Flatten()(pool2(ReLU(conv2(pool1(ReLU(conv1(x)))))))))
        loss = MSELoss(logits, y)
        graph = Graph(loss, optimize=False)
        params = {
            "conv1.kernel": conv1.kernel,
            "conv1.bias": conv1.b,
            "conv2.kernel": conv2.kernel,
            "conv2.bias": conv2.b,
            "fc1.weight": fc1.W,
            "fc1.bias": fc1.b,
            "fc2.weight": fc2.W,
            "fc2.bias": fc2.b,
        }
        return {"x": x, "y": y, "logits": logits, "loss": loss, "graph": graph,
                "params": params, "kind": "regression", "backend": backend}

    def make_data(self, config):
        if config.get("prepared_dir"):
            from .prepared import load_prepared
            return load_prepared(config["prepared_dir"])
        if config.get("split_arrays"):
            from ..data_meta import ensure_split_meta
            split = {name: tuple(value) for name, value in config["split_arrays"].items()}
            split["meta"] = dict(config.get("split_meta") or {"source": "injected"})
            return ensure_split_meta(split, config)
        from apps.common.donkey_data import load_donkey_index
        from apps.common.image_preprocess import imread_nchw
        from apps.common.splits import build_split

        data_dir = Path(config.get("data_dir") or Path("mycar/data"))
        rows = load_donkey_index(data_dir, fixed_throttle=float(config.get("fixed_throttle", 0.3)),
                                 angle_scale=float(config.get("angle_scale", 1.0)))
        if len(rows) < int(config.get("global_batch", 32)):
            raise RuntimeError(f"insufficient donkey samples in {data_dir}")
        split_idx = build_split(len(rows), val_ratio=0.1, test_ratio=0.1, seed=int(config.get("seed", 0)))
        height = int(config.get("image_h", 120))
        width = int(config.get("image_w", 160))

        def load_subset(indices):
            xs, ys = [], []
            for idx in indices:
                rel, angle, throttle = rows[idx]
                image = imread_nchw(data_dir / rel, (width, height), dtype=np.float32)[0]
                xs.append(image)
                ys.append((float(angle), float(throttle)))
            return np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.float32)

        split = {name: load_subset(split_idx[name]) for name in ("train", "val", "test")}
        split["meta"] = {
            "source": str(data_dir),
            "n_train": int(len(split["train"][0])),
            "n_val": int(len(split["val"][0])),
            "n_test": int(len(split["test"][0])),
            "seed": int(config.get("seed", 0)),
        }
        from ..data_meta import ensure_split_meta
        return ensure_split_meta(split, config)

    def epoch_index(self, split, config, epoch=0, split_name="train"):
        self.epoch_index_builds += 1
        x, y = split[split_name]
        batch = int(config.get("global_batch", 32))
        drop_last = bool(config.get("drop_last", True))
        n = (len(x) // batch) * batch if drop_last else len(x)
        rng = np.random.default_rng(int(config.get("seed", 0)) + 23 + int(epoch))
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
        pred = asnumpy(session.logits.value, copy=False)
        err = np.asarray(pred) - np.asarray(y_cpu)
        n = int(err.shape[0])
        n_elem = int(err.size)
        sq = float(np.sum(err ** 2))
        angle = err[:, 0]
        throttle = err[:, 1]
        return {
            "loss": {"sum": sq, "count": n_elem, "kind": "mean"},
            "mse": {"sum": sq, "count": n_elem, "kind": "mean"},
            "angle_mae": {"sum": float(np.sum(np.abs(angle))), "count": n, "kind": "mean"},
            "angle_rmse": {"sum": float(np.sum(angle ** 2)), "count": n, "kind": "rmse"},
            "throttle_mae": {"sum": float(np.sum(np.abs(throttle))), "count": n, "kind": "mean"},
        }
