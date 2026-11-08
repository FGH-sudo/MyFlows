"""Opt-in NVTX scopes and a bounded cudaProfilerApi capture, with source maps."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from functools import lru_cache, wraps
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def parse_steps(value: str | None):
    if value is None:
        return None
    try:
        first, last = (int(part) for part in value.split(":"))
    except (TypeError, ValueError) as exc:
        raise ValueError("--profile-steps must be A:B (inclusive, 1-based)") from exc
    if first < 1 or last < first:
        raise ValueError("--profile-steps requires 1 <= A <= B")
    return first, last


@lru_cache(maxsize=1024)
def _location(filename, line):
    path = Path(filename).resolve()
    try:
        relative = path.relative_to(ROOT).as_posix()
    except ValueError:
        relative = path.name
    return {"file": relative, "line": int(line)}


class TrainingProfiler:
    def __init__(self, *, enabled=False, nodes=False, capture_steps=None, output=None, device="cuda", cuda=None):
        self.enabled = bool(enabled or nodes)
        self.nodes_enabled = bool(nodes)
        self.capture_steps = parse_steps(capture_steps) if isinstance(capture_steps, str) else capture_steps
        self.output = Path(output) if output else None
        self.cuda = cuda
        if self.enabled or self.capture_steps:
            if device != "cuda":
                raise ValueError("NVTX/cudaProfilerApi requires a CUDA training run")
            if self.cuda is None:
                from MyFlows.core.device import _lazy_cupy

                self.cuda = _lazy_cupy().cuda
        self.capturing = False
        self.ranges = {}
        self.node_labels = {}
        self._iterator = None

    @contextmanager
    def _push(self, label):
        if self.enabled:
            self.cuda.nvtx.RangePush(label)
        try:
            yield
        finally:
            if self.enabled:
                self.cuda.nvtx.RangePop()

    def range(self, name, *, source=None):
        if not self.enabled:
            return nullcontext()
        if source is None:
            frame = sys._getframe(1)
            source = _location(frame.f_code.co_filename, frame.f_lineno)
        label = f"{name} @{source['file']}:{source['line']}"
        self.ranges[label] = {**source, "phase": name}
        return self._push(label)

    def annotate(self, name):
        def decorate(function):
            if not self.enabled:
                return function
            source = _location(inspect.getsourcefile(function), inspect.getsourcelines(function)[1])

            @wraps(function)
            def wrapped(*args, **kwargs):
                with self.range(name, source=source):
                    return function(*args, **kwargs)

            return wrapped
        return decorate

    def bind_graph(self, graph):
        if not self.nodes_enabled:
            return
        for index, node in enumerate((*graph.sorted_nodes, *getattr(graph, "_bn_groups", {}).values())):
            for direction in ("forward", "backward"):
                method = getattr(type(node), direction)
                label = f"node:{direction}:{index}:{node.name}"
                self.node_labels[node, direction] = label
                self.ranges[label] = {**_location(inspect.getsourcefile(method), inspect.getsourcelines(method)[1]),
                                      "node_class": type(node).__name__, "node_name": node.name, "direction": direction}
        graph.profiler = self

    def node(self, node, direction):
        return self._push(self.node_labels[node, direction])

    def batches(self, loader, *, start_step: int):
        if not self.enabled and self.capture_steps is None:
            return loader
        frame = sys._getframe(1)
        source = _location(frame.f_code.co_filename, frame.f_lineno)
        self._iterator = self._batches(loader, start_step, source)
        return self._iterator

    def _batches(self, loader, start_step, source):
        iterator = iter(loader)
        try:
            for offset in range(len(loader)):
                step = start_step + offset
                if self.capture_steps and step == self.capture_steps[0]:
                    self.cuda.runtime.deviceSynchronize()
                    self.cuda.profiler.start()
                    self.capturing = True
                label = f"step {step}"
                self.ranges[label] = {**source, "phase": "step", "step": step}
                try:
                    with self._push(label):
                        with self.range("data", source=source):
                            batch = next(iterator)
                        yield batch
                finally:
                    if self.capture_steps and step == self.capture_steps[1]:
                        self._stop_capture()
        finally:
            close = getattr(iterator, "close", None)
            if close:
                close()

    def _stop_capture(self):
        if self.capturing:
            try:
                self.cuda.runtime.deviceSynchronize()
            finally:
                self.cuda.profiler.stop()
                self.capturing = False

    def close(self):
        try:
            if self._iterator is not None:
                self._iterator.close()
                self._iterator = None
        finally:
            self._stop_capture()
            if self.output and self.enabled:
                self.output.parent.mkdir(parents=True, exist_ok=True)
                self.output.write_text(json.dumps({"schema_version": 1, "capture_steps": self.capture_steps,
                                                   "ranges": self.ranges}, ensure_ascii=False, indent=2), encoding="utf-8")
