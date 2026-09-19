"""Training budget, epoch batch cursor, and the no-communication local loop."""

from __future__ import annotations

import time

import numpy as np

from .data_meta import lightweight_data_meta
from .metrics_reduce import reduce_step_metrics
from .session import TrainSession


def resolve_budget(config, _data=None):
    task = str(config.get("task", "synthetic"))
    if task == "synthetic":
        steps = int(config.get("steps", 20))
        return {
            "epochs": 1,
            "steps_per_epoch": steps,
            "total_steps": steps,
            "steps_semantics": "synthetic_total_steps",
        }
    epochs = int(config.get("epochs", 1))
    cap = int(config.get("steps", 20))
    return {
        "epochs": epochs,
        "steps_per_epoch": cap,
        "total_steps": epochs * cap,
        "steps_semantics": "per_epoch_batch_cap",
    }


def iter_steps(cursor, budget):
    global_step = 0
    for epoch in range(int(budget["epochs"])):
        for step_in_epoch in range(cursor.n_batches(epoch)):
            yield epoch, step_in_epoch, global_step
            global_step += 1


def metric_row(worker_id, epoch, global_step, n_samples, loss, metrics, stats=None):
    row = {
        "worker_id": int(worker_id),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "n_samples": int(n_samples),
        "loss": float(loss),
        "aggregation": "local",
        "scope": "train_batch",
    }
    for key, value in (metrics or {}).items():
        row[key] = float(value) if np.isscalar(value) else value
    if stats:
        row["stats"] = {
            name: {
                "sum": float(spec.get("sum") or 0.0),
                "count": int(spec.get("count") or 0),
                "kind": spec.get("kind") or "mean",
            }
            for name, spec in stats.items()
        }
    return row


class BatchCursor:
    """Build each epoch's batch index once; materialize only the current batch."""

    def __init__(self, task, data, config):
        self.task = task
        self.data = data
        self.config = config
        self.builds = 0
        self._epoch = None
        self._indices = None
        self._n_batches = 0
        self._synthetic = str(config.get("task", "synthetic")) == "synthetic"

    def prepare(self, epoch):
        epoch = int(epoch)
        if self._synthetic:
            if self._indices is not None:
                return
            self.builds += 1
            self._epoch = 0
            self._indices = ()
            self._n_batches = int(self.data[0].shape[0])
            return
        if self._epoch == epoch and self._indices is not None:
            return
        self.builds += 1
        self._epoch = epoch
        indices = self.task.epoch_index(self.data, self.config, epoch=epoch + int(self.config.get('data_epoch', 0)))
        cap = int(self.config.get("steps", 0) or 0)
        if cap > 0:
            indices = indices[:cap]
        self._indices = indices
        self._n_batches = len(indices)
        if self._n_batches <= 0:
            raise RuntimeError("no batches in epoch; check global_batch and dataset size")

    def n_batches(self, epoch):
        self.prepare(epoch)
        return self._n_batches

    def get(self, epoch, step_in_epoch, shard=None):
        self.prepare(epoch)
        if self._synthetic:
            return self.task.shard_arrays(self.data, int(step_in_epoch), shard)
        indices = self._indices[int(step_in_epoch)]
        if shard is not None:
            indices = indices[shard]
        return self.task.take_batch(self.data, indices)


def run_local_training(config):
    """Single-process baseline: same task/session/optimizer, no PS or Ring."""
    started = time.perf_counter()
    session = TrainSession(config, for_compute=True)
    data = session.task.make_data(config)
    budget = resolve_budget(config, data)
    cursor = BatchCursor(session.task, data, config)
    from .measurement import Recorder
    recorder = Recorder(session, data, config, 0)
    recorder.prepare(cursor)
    collect_history = bool(config.get("collect_history", True))
    history = [session.named_parameters_cpu()] if collect_history else []
    metrics = []
    step_offset = session.local_version
    for epoch, step_in_epoch, global_step in iter_steps(cursor, budget):
        global_step += step_offset
        if step_in_epoch == 0:
            recorder.start_epoch(epoch)
        step_start = time.perf_counter()
        x, y = cursor.get(epoch, step_in_epoch)
        data_s = time.perf_counter() - step_start
        computed = session.forward_backward(x, y, materialize_cpu_gradients=False)
        session.apply_global_gradients(computed["device_gradients"], f"single-{global_step}")
        recorder.record(epoch, global_step, step_start, data_s, computed,
                        gradients=computed['device_gradients'])
        row = metric_row(
            0, epoch, global_step, computed["n_samples"], computed["loss"],
            computed.get("metrics"), computed.get("metric_stats"))
        metrics.append(row)
        if collect_history:
            history.append(session.named_parameters_cpu())
        if step_in_epoch + 1 == cursor.n_batches(epoch):
            recorder.end_epoch(epoch, global_step)
    recorder.finish()
    global_rows = []
    global_losses = []
    issues = []
    for row in metrics:
        merged, step_issues = reduce_step_metrics([row], n_workers=1)
        if step_issues:
            issues.extend(step_issues)
        else:
            global_rows.append(merged)
            global_losses.append(float(merged["loss"]))
    data_meta = lightweight_data_meta(data, config)
    return {
        "status": "passed" if not issues else "failed",
        "error": "; ".join(issues) if issues else None,
        "history": history,
        "losses": global_losses,
        "metrics": global_rows,
        "worker_metrics": {0: metrics},
        "metrics_scope": "train_batch",
        "metrics_valid": not issues,
        "metrics_issues": issues,
        "events": [
            {
                "phase": "metrics",
                "worker_id": 0,
                "step_id": row["global_step"],
                **{k: v for k, v in row.items() if k != "worker_id"},
            }
            for row in metrics
        ] + [{
            "phase": "data_meta",
            "worker_id": 0,
            **data_meta,
        }],
        "communication": False,
        "cpu_grad_materializations": int(session.cpu_grad_materializations),
        "epoch_index_builds": int(cursor.builds),
        "data_meta": data_meta,
        "elapsed_s": time.perf_counter() - started,
        "cleanup_s": 0.0,
        "exitcodes": {},
        "alive_pids": [],
    }
