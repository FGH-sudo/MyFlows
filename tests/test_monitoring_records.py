"""Monitoring lifecycle and incremental ingestion without a GPU or live NVML."""

import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock

from MyFlows.monitoring.gpu import GpuSampler, ResourceRecorder, _CpuMeter, _NvmlBackend
from MyFlows.monitoring.record import monitoring_enabled, watch
from MyFlows.monitoring.schema import EXTERNAL_MONITOR_ENV, JsonlCursor
from MyFlows.utils.training_dashboard import TrainingDashboard


class FakeSampler:
    def sample(self):
        return {"backend": "fake", "monotonic_s": time.perf_counter(), "unix_s": time.time(),
                "status": "partial", "gpus": [{"index": 0, "name": "fixture", "uuid": "fixture",
                "utilization_pct": 50., "memory_used_mib": 100., "memory_total_mib": 8000.,
                "temperature_c": None, "power_w": 20., "power_limit_w": 50., "processes": [],
                "missing": {"temperature_c": "not supported by fixture"}}]}


class MonitoringRecordsTest(unittest.TestCase):
    def test_modes_and_external_ownership(self):
        self.assertTrue(monitoring_enabled("auto", "run", {}))
        self.assertFalse(monitoring_enabled("auto", None, {}))
        self.assertFalse(monitoring_enabled("off", "run", {}))
        for mode in ("on", "auto"):
            self.assertFalse(monitoring_enabled(mode, "run", {EXTERNAL_MONITOR_ENV: "1"}))

    def test_cursor_waits_for_complete_line_and_detects_truncation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resources.jsonl"
            cursor = JsonlCursor(path)
            self.assertEqual(cursor.poll(), [])
            path.write_bytes(b'{"n": 1}\n{"n": ')
            self.assertEqual(cursor.poll(), [{"n": 1}])
            self.assertEqual(cursor.poll(), [])
            with path.open("ab") as handle:
                handle.write(b'2}\ninvalid\n[]\n')
            self.assertEqual(cursor.poll(), [{"n": 2}])
            path.write_bytes(b'{"n": 3}\n')
            self.assertEqual(cursor.poll(), [{"n": 3}])

    def test_fake_recorder_writes_null_reason_and_distinct_memory(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resources.jsonl"
            sampled = threading.Event()
            host = {"cpu_utilization_pct": 20., "ram_used_bytes": 1000, "ram_total_bytes": 2000,
                    "process_rss_bytes": {"123": 100, "456": 200}}
            with mock.patch("MyFlows.monitoring.gpu.sample_host", return_value=host):
                with ResourceRecorder(FakeSampler(), interval_s=0.01, jsonl_path=path, root_pid=123,
                                      on_sample=lambda row: sampled.set()) as recorder:
                    self.assertTrue(sampled.wait(3))
                self.assertIsNone(recorder.error)
            row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(row["schema_version"], 1)
            self.assertIsNone(row["gpu_temperature_c"])
            self.assertIn("gpu_temperature_c", row["missing"])
            self.assertEqual(row["gpu_device_memory_mib"], 100.)
            self.assertEqual(row["gpu_device_memory_peak_mib"], 100.)
            self.assertEqual(row["process_rss_total_bytes"], 300)
            self.assertEqual(row["root_pid"], 123)
            self.assertEqual(row["interval_s"], 0.01)

    def test_watch_stops_without_leaving_sampling_thread(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stop, ready = root / "stop", root / "ready"
            errors = []

            def run():
                try:
                    watch(os.getpid(), root / "resources.jsonl", 0.01, stop_file=stop,
                          ready_file=ready, sampler=FakeSampler())
                except Exception as exc:
                    errors.append(exc)

            thread = threading.Thread(target=run)
            thread.start()
            try:
                deadline = time.monotonic() + 5
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertTrue(ready.exists())
            finally:
                stop.write_text("stop", encoding="ascii")
                thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])

    def test_reused_parent_pid_does_not_start_recorder(self):
        import psutil

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resources.jsonl"
            watch(os.getpid(), path, 0.01, create_time=psutil.Process().create_time() - 10,
                  sampler=FakeSampler())
            self.assertFalse(path.exists())

    def test_resource_ingestion_uses_original_time_and_never_replays_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "resources.jsonl"
            dashboard = TrainingDashboard(root / "tb", enabled=False, resources_path=path)
            dashboard.tb = mock.Mock(active=True)
            row = {"schema_version": 1, "unix_s": 123.25, "gpu_power_w": 20., "gpu_temperature_c": None,
                   "gpu_device_memory_mib": 100., "cpu_utilization_pct": 5.}
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            dashboard.log_resources()
            self.assertIn(mock.call("resources/gpu_power_w", 20., 0, wall_time=123.25),
                          dashboard.tb.log_scalar.call_args_list)
            self.assertFalse(any(call.args[0].endswith("temperature_c") for call in dashboard.tb.log_scalar.call_args_list))
            count = dashboard.tb.log_scalar.call_count
            dashboard.log_resources()
            self.assertEqual(dashboard.tb.log_scalar.call_count, count)
            dashboard.close()

    def test_phase_events_survive_noncoincident_logging_intervals(self):
        with tempfile.TemporaryDirectory() as tmp:
            dashboard = TrainingDashboard(Path(tmp), enabled=False, log_interval=10)
            dashboard.tb = mock.Mock(active=True)
            dashboard.log_train_step(21, loss=1., data_load_ms=1., train_step_ms=3., step_time_ms=4.,
                                     batch_size=2, labels=None, task="regression",
                                     extra={"forward_ms": 1.5, "backward_ms": 1.5, "optimizer_ms": .2})
            calls = dashboard.tb.log_scalar.call_args_list
            self.assertIn(mock.call("train/backward_ms", 1.5, 21), calls)
            self.assertNotIn(mock.call("train/loss_step", 1., 21), calls)
            dashboard.close()

    def test_nvml_partial_init_failure_calls_shutdown(self):
        fake_nvml = types.SimpleNamespace(
            nvmlInit=mock.Mock(),
            nvmlShutdown=mock.Mock(),
            nvmlDeviceGetCount=mock.Mock(side_effect=RuntimeError("nvml query fail")),
        )
        with mock.patch.dict(sys.modules, {"pynvml": fake_nvml}):
            with self.assertRaisesRegex(RuntimeError, "nvml query fail"):
                _NvmlBackend()
        fake_nvml.nvmlInit.assert_called_once()
        fake_nvml.nvmlShutdown.assert_called_once()

    def test_gpu_sampler_close_is_synchronized_and_idempotent(self):
        sampler = GpuSampler.__new__(GpuSampler)
        sampler._lock = threading.Lock()
        sampler.errors = {}
        sampler.backend = mock.Mock()
        sampler.backend.name = "fake"
        sampler.backend.read.return_value = []
        sampler.close()
        self.assertIsNone(sampler.backend)
        row = sampler.sample()
        self.assertEqual(row["status"], "unavailable")

    def test_cpu_meters_are_independent_across_recorders(self):
        from collections import namedtuple

        CpuTimes = namedtuple("scputimes", ("user", "system", "idle"))
        meter_a = _CpuMeter()
        meter_b = _CpuMeter()
        ticks = iter([
            CpuTimes(user=10.0, system=0.0, idle=90.0),
            CpuTimes(user=12.0, system=0.0, idle=98.0),
            CpuTimes(user=15.0, system=0.0, idle=105.0),
        ])
        fake_psutil = types.SimpleNamespace(LINUX=False, cpu_times=lambda: next(ticks))
        self.assertIsNone(meter_a.sample(fake_psutil))
        self.assertIsNone(meter_b.sample(fake_psutil))
        # meter_a's interval runs from (10, 100) to (15, 120): busy +5 / total +20 = 25.0%
        self.assertAlmostEqual(meter_a.sample(fake_psutil), 25.0, places=1)

    def test_non_finite_metrics_are_sanitized_to_null_with_reason(self):
        class _NaNSampler:
            backend_name = "fake"

            def sample(self):
                return {
                    "backend": "fake",
                    "monotonic_s": 1.0,
                    "unix_s": 2.0,
                    "status": "ok",
                    "gpus": [{
                        "index": 0, "name": "fixture", "uuid": "fixture",
                        "utilization_pct": float("nan"), "memory_used_mib": float("inf"),
                        "memory_total_mib": 8000.0, "temperature_c": 40.0,
                        "power_w": 20.0, "power_limit_w": 50.0, "processes": [], "missing": {},
                    }],
                    "error": None,
                }

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resources.jsonl"
            sampled = threading.Event()
            host = {"cpu_utilization_pct": float("nan"), "ram_used_bytes": 1000, "ram_total_bytes": 2000,
                    "process_rss_bytes": {"1": 100}}
            with mock.patch("MyFlows.monitoring.gpu.sample_host", return_value=host):
                with ResourceRecorder(_NaNSampler(), interval_s=0.01, jsonl_path=path,
                                      on_sample=lambda _row: sampled.set()) as recorder:
                    self.assertTrue(sampled.wait(3))
                self.assertIsNone(recorder.error)
            row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
            self.assertIsNone(row["gpu_utilization_pct"])
            self.assertIsNone(row["gpu_memory_mib"])
            self.assertIsNone(row["cpu_utilization_pct"])
            self.assertEqual(row["missing"]["gpu_utilization_pct"], "non-finite metric value")
            self.assertEqual(row["missing"]["cpu_utilization_pct"], "non-finite metric value")
            self.assertEqual(row["gpus"][0]["missing"]["utilization_pct"], "non-finite metric value")


if __name__ == "__main__":
    unittest.main()
