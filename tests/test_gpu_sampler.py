import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from MyFlows.monitoring import gpu as gpu_mod
from MyFlows.monitoring.gpu import (
    GpuSampler,
    ResourceRecorder,
    _SmiBackend,
    flat_resource_row,
)


def _fake_smi(line):
    def run(args):
        if args[0].startswith("--query-gpu=driver_version"):
            return "591.74\n"
        return line + "\n"
    return run


class SmiBackendTest(unittest.TestCase):
    def test_unsupported_fields_are_null_with_reason(self):
        backend = _SmiBackend(runner=_fake_smi("0, RTX, GPU-1, 37, 1500, 8188, [N/A], [Not Supported], 80"))
        gpu = backend.read()[0]
        self.assertEqual(gpu["utilization_pct"], 37.0)
        self.assertEqual(gpu["memory_total_mib"], 8188.0)
        self.assertIsNone(gpu["temperature_c"])
        self.assertIsNone(gpu["power_w"])
        self.assertIn("temperature_c", gpu["missing"])
        self.assertIn("power_w", gpu["missing"])
        self.assertIsNone(gpu["processes"])
        self.assertEqual(backend.driver_version, "591.74")


class GpuSamplerFallbackTest(unittest.TestCase):
    def test_falls_back_to_smi_when_nvml_fails(self):
        smi = _SmiBackend(runner=_fake_smi("0, RTX, GPU-1, 10, 100, 8188, 40, 12.5, 80"))
        with mock.patch.object(gpu_mod, "_NvmlBackend", side_effect=RuntimeError("no nvml")), \
                mock.patch.object(gpu_mod, "_SmiBackend", return_value=smi):
            sampler = GpuSampler()
        self.assertEqual(sampler.backend_name, "nvidia-smi")
        self.assertIn("nvml", sampler.errors)
        row = sampler.sample()
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["gpus"][0]["power_w"], 12.5)

    def test_unavailable_reports_reason_and_no_values(self):
        with mock.patch.object(gpu_mod, "_NvmlBackend", side_effect=RuntimeError("no nvml")), \
                mock.patch.object(gpu_mod, "_SmiBackend", side_effect=FileNotFoundError("nvidia-smi")):
            sampler = GpuSampler()
        row = sampler.sample()
        self.assertEqual(row["status"], "unavailable")
        self.assertEqual(row["gpus"], [])
        self.assertIn("no nvml", row["error"])
        flat = flat_resource_row(row, {"cpu_utilization_pct": 1.0})
        self.assertIsNone(flat["gpu_utilization_pct"])
        self.assertIn("sampling_error", flat)

    def test_nvml_backend_reports_null_process_memory(self):
        fake = mock.MagicMock()
        fake.nvmlDeviceGetCount.return_value = 1
        fake.nvmlDeviceGetName.return_value = b"RTX"
        fake.nvmlDeviceGetUUID.return_value = "GPU-1"
        fake.nvmlSystemGetDriverVersion.return_value = "591.74"
        fake.nvmlDeviceGetUtilizationRates.return_value = mock.Mock(gpu=55)
        fake.nvmlDeviceGetMemoryInfo.return_value = mock.Mock(used=1024 * 1024 * 100, total=1024 * 1024 * 8188)
        fake.nvmlDeviceGetTemperature.return_value = 50
        fake.nvmlDeviceGetPowerUsage.side_effect = RuntimeError("not supported")
        fake.nvmlDeviceGetEnforcedPowerLimit.return_value = 80000
        fake.nvmlDeviceGetComputeRunningProcesses.return_value = [mock.Mock(pid=42, usedGpuMemory=None)]
        with mock.patch.dict(sys.modules, {"pynvml": fake}):
            sampler = GpuSampler(backend="nvml")
        row = sampler.sample()
        gpu = row["gpus"][0]
        self.assertEqual(sampler.backend_name, "nvml")
        self.assertEqual(gpu["name"], "RTX")
        self.assertEqual(gpu["utilization_pct"], 55.0)
        self.assertAlmostEqual(gpu["memory_used_mib"], 100.0)
        self.assertIsNone(gpu["power_w"])
        self.assertIn("power_w", gpu["missing"])
        self.assertEqual(gpu["processes"], [{"pid": 42, "used_memory_mib": None}])
        self.assertIn("process_memory", gpu["missing"])
        self.assertEqual(row["status"], "partial")


class ResourceRecorderTest(unittest.TestCase):
    def test_writes_compatible_jsonl_rows(self):
        smi = _SmiBackend(runner=_fake_smi("0, RTX, GPU-1, 10, 100, 8188, 40, 12.5, 80"))
        with mock.patch.object(gpu_mod, "_NvmlBackend", side_effect=RuntimeError("no nvml")), \
                mock.patch.object(gpu_mod, "_SmiBackend", return_value=smi):
            sampler = GpuSampler()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resources.jsonl"
            with ResourceRecorder(sampler, interval_s=0.02, jsonl_path=path, history=5) as recorder:
                deadline = time.time() + 3
                while len(recorder.history()) < 3 and time.time() < deadline:
                    time.sleep(0.02)
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertGreaterEqual(len(rows), 3)
            self.assertLessEqual(len(recorder.history()), 5)
            for key in ("monotonic_s", "cpu_utilization_pct", "ram_used_bytes", "gpu_utilization_pct",
                        "gpu_memory_mib", "gpu_temperature_c", "gpu_power_w"):
                self.assertIn(key, rows[0])
            self.assertEqual(rows[0]["gpu_power_w"], 12.5)


if __name__ == "__main__":
    unittest.main()
