"""Resource recorder process tied to a training PID and its creation time.

Usage: python -m MyFlows.monitoring.record --pid PID --out resources.jsonl --interval-s 0.5
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from .schema import EXTERNAL_MONITOR_ENV


def monitoring_enabled(mode: str, run_dir, environ=None) -> bool:
    if mode not in ("auto", "on", "off"):
        raise ValueError(f"unknown monitoring mode: {mode}")
    environ = os.environ if environ is None else environ
    # One writer per resources.jsonl, including explicit --monitor on under the console.
    return bool(run_dir) and mode != "off" and environ.get(EXTERNAL_MONITOR_ENV) != "1"


def watch(pid: int, out: Path, interval_s: float, *, create_time: float | None = None,
          stop_file: Path | None = None, ready_file: Path | None = None, sampler=None):
    """Stop on parent exit, PID reuse, or the owner's graceful stop request."""
    import psutil

    from .gpu import GpuSampler, ResourceRecorder

    if not math.isfinite(interval_s) or interval_s <= 0:
        raise ValueError("interval_s must be finite and positive")
    try:
        parent = psutil.Process(pid)
        if create_time is not None and abs(parent.create_time() - create_time) > 0.001:
            return
    except psutil.NoSuchProcess:
        return
    owned_sampler = sampler is None
    sampler = GpuSampler() if owned_sampler else sampler
    try:
        with ResourceRecorder(sampler, interval_s=interval_s, jsonl_path=out, root_pid=pid,
                              history=2) as recorder:
            while parent.is_running() and parent.status() != psutil.STATUS_ZOMBIE:
                if stop_file is not None and stop_file.exists():
                    break
                if recorder.error is not None:
                    raise RuntimeError(recorder.error)
                if ready_file is not None and recorder.latest() is not None:
                    ready_file.write_text("ready\n", encoding="ascii")
                    ready_file = None
                time.sleep(min(interval_s, 0.1))
    except psutil.NoSuchProcess:
        pass
    finally:
        if owned_sampler:
            sampler.close()


class MonitorProcess:
    """Parent-side lifecycle. No NVML calls or sampling thread in the trainer."""

    def __init__(self, run_dir: str | Path, interval_s: float = 0.5):
        if not math.isfinite(interval_s) or interval_s <= 0:
            raise ValueError("monitor interval must be finite and positive")
        self.run_dir = Path(run_dir)
        self.interval_s = interval_s
        self.process = None
        self.stop_file = self.run_dir / f".monitor-{os.getpid()}.stop"
        self.ready_file = self.run_dir / f".monitor-{os.getpid()}.ready"

    def start(self):
        import psutil

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.stop_file.unlink(missing_ok=True)
        self.ready_file.unlink(missing_ok=True)
        command = [sys.executable, "-m", "MyFlows.monitoring.record", "--pid", str(os.getpid()),
                   "--create-time", str(psutil.Process().create_time()), "--out", str(self.run_dir / "resources.jsonl"),
                   "--interval-s", str(self.interval_s), "--stop-file", str(self.stop_file),
                   "--ready-file", str(self.ready_file)]
        with (self.run_dir / "monitor.log").open("ab") as log:
            self.process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        # Initialization is outside training timing; even a short run gets at least one sample.
        deadline = time.perf_counter() + 20
        while not self.ready_file.exists():
            if self.process.poll() is not None or time.perf_counter() > deadline:
                self.stop()
                raise RuntimeError("resource monitor did not become ready; see monitor.log")
            time.sleep(0.02)
        return self

    def stop(self):
        if self.process is None:
            return
        try:
            self.stop_file.write_text("stop\n", encoding="ascii")
            try:
                self.process.wait(timeout=6)
            except subprocess.TimeoutExpired:
                # On Windows a venv launcher can own another Python process.
                import psutil

                try:
                    children = psutil.Process(self.process.pid).children(recursive=True)
                except psutil.NoSuchProcess:
                    children = []
                for child in children:
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                self.process.kill()
                self.process.wait(timeout=5)
        finally:
            self.stop_file.unlink(missing_ok=True)
            self.ready_file.unlink(missing_ok=True)
            self.process = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval-s", type=float, default=0.5)
    parser.add_argument("--create-time", type=float)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--ready-file", type=Path)
    args = parser.parse_args()
    watch(args.pid, args.out, args.interval_s, create_time=args.create_time,
          stop_file=args.stop_file, ready_file=args.ready_file)


if __name__ == "__main__":
    main()
