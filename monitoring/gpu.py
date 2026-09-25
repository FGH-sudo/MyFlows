"""GPU/CPU resource sampling: NVML first, nvidia-smi as fallback.

Unsupported or unreadable metrics are reported as ``None`` together with a
reason in ``missing``; they are never replaced by 0.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from collections.abc import Callable
from pathlib import Path

GPU_FIELDS = (
    "utilization_pct",
    "memory_used_mib",
    "memory_total_mib",
    "temperature_c",
    "power_w",
    "power_limit_w",
)

_SMI_QUERY = (
    "index,name,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit"
)
_SMI_COLUMNS = ("index", "name", "uuid") + GPU_FIELDS
_MIB = 1024 * 1024
_WDDM_REASON = "NVML 未提供每进程显存（Windows WDDM 模式常见）"


def _no_window() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


class _NvmlBackend:
    name = "nvml"

    def __init__(self):
        import pynvml

        self.nv = pynvml
        pynvml.nvmlInit()
        self.count = pynvml.nvmlDeviceGetCount()
        self.handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.count)]
        self.static = []
        for index, handle in enumerate(self.handles):
            self.static.append({
                "index": index,
                "name": self._try(lambda: _text(pynvml.nvmlDeviceGetName(handle)))[0],
                "uuid": self._try(lambda: _text(pynvml.nvmlDeviceGetUUID(handle)))[0],
            })
        self.driver_version = self._try(lambda: _text(pynvml.nvmlSystemGetDriverVersion()))[0]

    @staticmethod
    def _try(fn: Callable):
        try:
            return fn(), None
        except Exception as exc:  # NVMLError or missing symbol
            return None, f"{type(exc).__name__}: {exc}"

    def read(self) -> list[dict]:
        nv = self.nv
        gpus = []
        for handle, static in zip(self.handles, self.static):
            gpu = dict(static)
            missing = {}

            def put(field, fn):
                value, reason = self._try(fn)
                gpu[field] = value
                if reason:
                    missing[field] = reason

            put("utilization_pct", lambda: float(nv.nvmlDeviceGetUtilizationRates(handle).gpu))
            memory, reason = self._try(lambda: nv.nvmlDeviceGetMemoryInfo(handle))
            gpu["memory_used_mib"] = memory.used / _MIB if memory else None
            gpu["memory_total_mib"] = memory.total / _MIB if memory else None
            if reason:
                missing["memory_used_mib"] = missing["memory_total_mib"] = reason
            put("temperature_c", lambda: float(nv.nvmlDeviceGetTemperature(handle, nv.NVML_TEMPERATURE_GPU)))
            put("power_w", lambda: nv.nvmlDeviceGetPowerUsage(handle) / 1000.0)
            put("power_limit_w", lambda: nv.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000.0)
            procs, reason = self._try(lambda: nv.nvmlDeviceGetComputeRunningProcesses(handle))
            if reason:
                gpu["processes"] = None
                missing["processes"] = reason
            else:
                gpu["processes"] = [
                    {"pid": int(p.pid),
                     "used_memory_mib": (p.usedGpuMemory / _MIB) if p.usedGpuMemory is not None else None}
                    for p in procs
                ]
                if any(p["used_memory_mib"] is None for p in gpu["processes"]):
                    missing["process_memory"] = _WDDM_REASON
            gpu["missing"] = missing
            gpus.append(gpu)
        return gpus

    def close(self):
        try:
            self.nv.nvmlShutdown()
        except Exception:
            pass


class _SmiBackend:
    name = "nvidia-smi"

    def __init__(self, executable: str = "nvidia-smi", runner: Callable | None = None):
        self.executable = executable
        self.runner = runner or self._run
        self.driver_version = None
        rows = self.read()
        if not rows:
            raise RuntimeError("nvidia-smi 未返回任何 GPU")
        try:
            self.driver_version = self.runner(["--query-gpu=driver_version", "--format=csv,noheader"]).splitlines()[0].strip()
        except Exception:
            self.driver_version = None

    def _run(self, args: list[str]) -> str:
        result = subprocess.run([self.executable, *args], capture_output=True, text=True, timeout=4,
                                creationflags=_no_window())
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or f"nvidia-smi exit {result.returncode}")
        return result.stdout

    def read(self) -> list[dict]:
        text = self.runner([f"--query-gpu={_SMI_QUERY}", "--format=csv,noheader,nounits"])
        gpus = []
        for line in text.strip().splitlines():
            values = [v.strip() for v in line.split(",")]
            if len(values) != len(_SMI_COLUMNS):
                continue
            gpu, missing = {}, {}
            for column, raw in zip(_SMI_COLUMNS, values):
                if column in ("name", "uuid"):
                    gpu[column] = raw
                elif column == "index":
                    gpu[column] = int(raw)
                else:
                    try:
                        gpu[column] = float(raw)
                    except ValueError:
                        gpu[column] = None
                        missing[column] = f"nvidia-smi 返回 {raw}"
            gpu["processes"] = None
            missing["processes"] = "nvidia-smi 回退模式不采集每进程显存"
            gpu["missing"] = missing
            gpus.append(gpu)
        return gpus

    def close(self):
        pass


def _text(value) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


class GpuSampler:
    """Read all visible GPUs. ``backend`` is ``auto``, ``nvml`` or ``nvidia-smi``."""

    def __init__(self, backend: str = "auto", smi_executable: str = "nvidia-smi"):
        self._lock = threading.Lock()
        self.backend = None
        self.errors: dict[str, str] = {}
        order = {"auto": ("nvml", "nvidia-smi"), "nvml": ("nvml",), "nvidia-smi": ("nvidia-smi",)}[backend]
        for name in order:
            try:
                self.backend = _NvmlBackend() if name == "nvml" else _SmiBackend(smi_executable)
                break
            except Exception as exc:
                self.errors[name] = f"{type(exc).__name__}: {exc}"

    @property
    def backend_name(self) -> str | None:
        return self.backend.name if self.backend else None

    @property
    def driver_version(self) -> str | None:
        return getattr(self.backend, "driver_version", None)

    def sample(self) -> dict:
        row = {"monotonic_s": time.perf_counter(), "unix_s": time.time(), "backend": self.backend_name}
        if self.backend is None:
            row.update(status="unavailable", gpus=[], error="; ".join(f"{k}: {v}" for k, v in self.errors.items()))
            return row
        try:
            with self._lock:
                gpus = self.backend.read()
        except Exception as exc:
            row.update(status="error", gpus=[], error=f"{type(exc).__name__}: {exc}")
            return row
        partial = any(set(g["missing"]) - {"process_memory", "processes"} for g in gpus)
        row.update(status="partial" if partial else "ok", gpus=gpus, error=None)
        return row

    def close(self):
        if self.backend is not None:
            self.backend.close()
            self.backend = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def sample_host(root_pid: int | None = None) -> dict:
    """CPU/RAM for the host plus RSS of ``root_pid`` and its descendants."""
    import psutil

    row = {"cpu_utilization_pct": psutil.cpu_percent(), "ram_used_bytes": psutil.virtual_memory().used,
           "ram_total_bytes": psutil.virtual_memory().total}
    if root_pid is not None:
        rss = {}
        try:
            root = psutil.Process(root_pid)
            for proc in [root, *root.children(recursive=True)]:
                try:
                    rss[str(proc.pid)] = proc.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except psutil.NoSuchProcess:
            pass
        row["process_rss_bytes"] = rss
    return row


def flat_resource_row(sample: dict, host: dict | None = None, gpu_index: int = 0) -> dict:
    """One ``resources.jsonl`` row, field-compatible with ``benchmark.distributed_experiment``."""
    row = {"monotonic_s": sample["monotonic_s"], "unix_s": sample["unix_s"], "sampler_backend": sample["backend"]}
    row.update(host or {})
    gpu = next((g for g in sample["gpus"] if g.get("index") == gpu_index), None)
    if gpu is None:
        row["sampling_error"] = sample.get("error") or f"GPU {gpu_index} 不可用"
        for key in ("gpu_utilization_pct", "gpu_memory_mib", "gpu_temperature_c", "gpu_power_w"):
            row[key] = None
    else:
        row.update(gpu_utilization_pct=gpu["utilization_pct"], gpu_memory_mib=gpu["memory_used_mib"],
                   gpu_temperature_c=gpu["temperature_c"], gpu_power_w=gpu["power_w"])
        if gpu["missing"]:
            row["gpu_missing"] = gpu["missing"]
    row["gpus"] = sample["gpus"]
    return row


class ResourceRecorder:
    """Background sampling thread; keeps a bounded in-memory history and optionally appends JSONL rows."""

    def __init__(self, sampler: GpuSampler, *, interval_s: float = 0.5, jsonl_path: str | Path | None = None,
                 root_pid: int | None = None, history: int = 1200, on_sample: Callable[[dict], None] | None = None):
        self.sampler = sampler
        self.interval_s = float(interval_s)
        self.jsonl_path = Path(jsonl_path) if jsonl_path else None
        self.root_pid = root_pid
        self.history_size = int(history)
        self.on_sample = on_sample
        self._history: list[dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="resource-recorder", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self, timeout: float = 5.0):
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def history(self, since_unix_s: float | None = None) -> list[dict]:
        with self._lock:
            rows = list(self._history)
        return rows if since_unix_s is None else [r for r in rows if r["unix_s"] > since_unix_s]

    def latest(self) -> dict | None:
        with self._lock:
            return self._history[-1] if self._history else None

    def _loop(self):
        handle = None
        try:
            if self.jsonl_path:
                self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                handle = self.jsonl_path.open("a", encoding="utf-8", buffering=1)
            while not self._stop.is_set():
                started = time.perf_counter()
                row = flat_resource_row(self.sampler.sample(), sample_host(self.root_pid))
                with self._lock:
                    self._history.append(row)
                    del self._history[:-self.history_size]
                if handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                if self.on_sample:
                    self.on_sample(row)
                self._stop.wait(max(0.0, self.interval_s - (time.perf_counter() - started)))
        finally:
            if handle:
                handle.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
