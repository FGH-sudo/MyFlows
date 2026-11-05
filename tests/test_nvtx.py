"""Bounded inclusive capture and balanced NVTX scopes, including errors and early returns."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock

from MyFlows.monitoring.nvtx import TrainingProfiler, parse_steps


class NvtxTest(TestCase):
    def test_capture_boundaries_and_source_map(self):
        cuda = Mock()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "map.json"
            profiler = TrainingProfiler(enabled=True, capture_steps="2:3", cuda=cuda, output=path)
            snapshots = []
            for value in profiler.batches([1, 2, 3, 4], start_step=1):
                with profiler.range("forward"):
                    snapshots.append((value, profiler.capturing))
            profiler.close()
            self.assertEqual(snapshots, [(1, False), (2, True), (3, True), (4, False)])
            cuda.profiler.start.assert_called_once()
            cuda.profiler.stop.assert_called_once()
            self.assertEqual(cuda.nvtx.RangePush.call_count, cuda.nvtx.RangePop.call_count)
            locations = json.loads(path.read_text(encoding="utf-8"))["ranges"]
            self.assertEqual(locations["step 2"]["file"], "MyFlows/tests/test_nvtx.py")
            self.assertTrue(all(row["line"] > 0 for row in locations.values()))

    def test_early_stop_balances_range_before_stopping_capture(self):
        cuda = Mock()
        profiler = TrainingProfiler(nodes=True, capture_steps="1:3", cuda=cuda)
        for _ in profiler.batches([1, 2, 3], start_step=1):
            break
        profiler.close()
        self.assertFalse(profiler.capturing)
        cuda.profiler.stop.assert_called_once()
        self.assertEqual(cuda.nvtx.RangePush.call_count, cuda.nvtx.RangePop.call_count)
        self.assertLess([call[0] for call in cuda.mock_calls].index("nvtx.RangePop"),
                        [call[0] for call in cuda.mock_calls].index("profiler.stop"))

    def test_disabled_path_preserves_loader_and_invalid_range_is_rejected(self):
        loader = [1]
        self.assertIs(TrainingProfiler(device="cpu").batches(loader, start_step=1), loader)
        for value in ("0:1", "3:2", "x", "1:2:3"):
            with self.assertRaises(ValueError):
                parse_steps(value)
