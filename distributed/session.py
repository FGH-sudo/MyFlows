"""Local model session: named parameters, GPU/CPU forward-backward and optimizer apply."""

from __future__ import annotations

from .constants import OPTIMIZER_IMPL
from ..core.device import asnumpy, get_device, set_device, xp
from ..train.opt import Adam, MBGD
from .protocol import digest_state, payload_hash, schema_from_arrays, schema_hash
from .tasks import build_task


class TrainSession:
    def __init__(self, config, *, for_compute=True):
        self.config = dict(config)
        device = str(config.get("device", "cpu"))
        if for_compute:
            set_device("cuda" if device.startswith("cuda") or device == "gpu" else "cpu")
        else:
            set_device("cpu")
        self.device = get_device()
        self.task = build_task(config.get("task", "synthetic"))
        built = self.task.build_model(config, device=self.device)
        self.x = built["x"]
        self.y = built["y"]
        self.loss_node = built["loss"]
        self.logits = built.get("logits")
        self.graph = built["graph"]
        self.params = built["params"]
        self.buffers = built.get("buffers", {})
        self.kind = built.get("kind", "regression")
        opt_name = str(config.get("optimizer", "mbgd")).lower()
        lr = float(config.get("learning_rate", 0.01))
        if opt_name == "adam":
            self.optimizer = Adam(self.graph, learning_rate=lr)
        elif opt_name == "mbgd":
            self.optimizer = MBGD(self.graph, learning_rate=lr)
        else:
            raise ValueError(f"unsupported optimizer {opt_name}")
        self._applied_update = None
        self.local_version = 0
        self.cpu_grad_materializations = 0
        self._schema = [{"name": name, "shape": list(node.value.shape), "dtype": "float32"}
                        for name, node in sorted(self.params.items())]
        self._schema_hash = schema_hash(self._schema)
        if config.get("init_checkpoint"):
            from .measurement import load_state
            load_state(self, config["init_checkpoint"])

    def schema(self):
        return self._schema

    def schema_hash(self):
        return self._schema_hash

    def named_parameters_cpu(self):
        return {name: asnumpy(node.value, copy=True).astype("float32", copy=False)
                for name, node in self.params.items()}

    def load_parameters(self, arrays):
        if set(arrays) != set(self.params):
            raise ValueError("parameter names differ")
        for name, node in self.params.items():
            node.value = xp.asarray(arrays[name], dtype=node.value.dtype)

    def forward_backward(self, x_cpu, y_cpu, *, materialize_cpu_gradients=True):
        import time
        from .measurement import timed
        timings = {}
        x_cpu = asnumpy(x_cpu, copy=False)
        y_cpu = asnumpy(y_cpu, copy=False)
        def inputs():
            self.x.value = xp.asarray(x_cpu)
            self.y.value = xp.asarray(y_cpu)
        for label, operation in (("input_h2d", inputs), ("forward", self.graph.forward),
                                 ("backward", self.graph.backward)):
            _, measured = timed(self, label, operation)
            timings.update(measured)
        backward_end_s = time.perf_counter()
        metric_start = time.perf_counter()
        loss = float(asnumpy(self.loss_node.value))
        device_grads = {name: node.grad for name, node in self.params.items()}
        stats = {}
        if hasattr(self.task, "metric_stats"):
            stats.update(self.task.metric_stats(self, x_cpu, y_cpu) or {})
        n_samples = int(x_cpu.shape[0])
        if "loss" not in stats:
            stats["loss"] = {"sum": loss * n_samples, "count": n_samples, "kind": "mean"}
        metrics = self.task.metrics(self, x_cpu, y_cpu)
        timings['metrics_wall_s'] = time.perf_counter() - metric_start
        cpu_grads = None
        copy_start = time.perf_counter()
        if materialize_cpu_gradients:
            self.cpu_grad_materializations += 1
            cpu_grads = {name: asnumpy(grad, copy=True).astype("float32", copy=False)
                         for name, grad in device_grads.items()}
        timings['gradient_d2h_s'] = time.perf_counter() - copy_start if materialize_cpu_gradients else 0.
        backends = []
        for node in self.graph.nodes:
            backend = getattr(node, "actual_backend", None)
            if backend:
                backends.append(backend)
        return {
            "loss": loss,
            "gradients": cpu_grads,
            "device_gradients": device_grads,
            "n_samples": n_samples,
            "metrics": metrics,
            "metric_stats": stats,
            "backends": backends,
            "timings": timings,
            "backward_end_s": backward_end_s,
        }

    def apply_global_gradients(self, gradients, update_id):
        if self._applied_update == update_id:
            return False
        if self.optimizer.acc_no != 0 or self.optimizer.acc_gradient:
            raise RuntimeError("optimizer cache must be empty before external gradients")
        from .measurement import timed
        def copy_gradients():
            return {node: _copy_for_update(gradients[name], node.value.dtype)
                    for name, node in self.params.items()}
        mapped, copy_times = timed(self, 'gradient_to_device', copy_gradients)
        _, update_times = timed(self, 'optimizer', lambda: self.optimizer.update(mapped))
        self.update_timings = {**copy_times, **update_times}
        self._applied_update = update_id
        self.local_version += 1
        return True

    def optimizer_meta(self):
        meta = {
            "impl": type(self.optimizer).__name__,
            "impl_path": OPTIMIZER_IMPL if isinstance(self.optimizer, Adam) else "MyFlows.train.opt.MBGD",
            "learning_rate": float(self.optimizer.learning_rate),
            "t": int(getattr(self.optimizer, "t", 0)),
        }
        for key in ("beta_1", "beta_2", "eps"):
            if hasattr(self.optimizer, key):
                meta[key] = float(getattr(self.optimizer, key))
        return meta

    def optimizer_state_cpu(self):
        state = dict(self.optimizer_meta())
        state["v"] = {}
        state["s"] = {}
        for name, node in self.params.items():
            if hasattr(self.optimizer, "v") and node in self.optimizer.v:
                state["v"][name] = asnumpy(self.optimizer.v[node], copy=True).astype("float32", copy=False)
            if hasattr(self.optimizer, "s") and node in self.optimizer.s:
                state["s"][name] = asnumpy(self.optimizer.s[node], copy=True).astype("float32", copy=False)
        return state

    def load_optimizer_state(self, state):
        if not state:
            return
        self.optimizer.learning_rate = float(state.get("learning_rate", self.optimizer.learning_rate))
        if hasattr(self.optimizer, "t"):
            self.optimizer.t = int(state.get("t", 0))
        for key in ('beta_1', 'beta_2', 'eps'):
            if key in state and hasattr(self.optimizer, key):
                setattr(self.optimizer, key, float(state[key]))
        for name, node in self.params.items():
            if state.get("v") and name in state["v"]:
                self.optimizer.v[node] = xp.asarray(state["v"][name])
            if state.get("s") and name in state["s"]:
                self.optimizer.s[node] = xp.asarray(state["s"][name])

    def digest(self):
        params = self.named_parameters_cpu()
        opt = self.optimizer_state_cpu()
        named = dict(params)
        named.update({f"buffer.{name}": asnumpy(value, copy=True) for name, value in self.buffers.items()})
        for name, value in opt.get("v", {}).items():
            named[f"v.{name}"] = value
        for name, value in opt.get("s", {}).items():
            named[f"s.{name}"] = value
        return digest_state(named, {"t": opt.get("t")}, step=opt.get("t"))

    def parameter_hash(self):
        return payload_hash(self.named_parameters_cpu())

    def actual_conv_backends(self):
        return [getattr(node, "actual_backend", None) for node in self.graph.nodes
                if getattr(node, "actual_backend", None)]


def _copy_for_update(value, dtype):
    """Copy onto the active device so Adam state cannot alias node.grad."""
    return xp.array(value, dtype=dtype, copy=True)
