"""Monitoring lifecycle and incremental ingestion without a GPU or live NVML."""

import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

from MyFlows.monitoring.gpu import ResourceRecorder
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


if __name__ == "__main__":
    unittest.main()
