import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from MyFlows.utils.training_dashboard import TrainingDashboard


class TrainingDashboardJsonlTest(unittest.TestCase):
    def test_jsonl_written_without_tensorboard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run" / "metrics.jsonl"
            dashboard = TrainingDashboard(Path(tmp) / "tb", enabled=False, log_interval=10, jsonl_path=path)
            labels = np.zeros((2, 2))
            for step in (1, 2):
                dashboard.log_train_step(step, loss=1.0 / step, running_loss=0.7, data_load_ms=1.0,
                                         train_step_ms=4.0, step_time_ms=5.0, batch_size=2,
                                         labels=labels, task="regression", epoch=1)
            dashboard.log_epoch(1, loss=0.6, learning_rate=1e-3, epoch_time_s=2.0, validation_loss=0.8)
            dashboard.close()
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([r["kind"] for r in rows], ["step", "step", "epoch"])
        self.assertEqual(rows[1]["step"], 2)
        self.assertEqual(rows[1]["epoch"], 1)
        self.assertAlmostEqual(rows[0]["samples_per_sec"], 400.0)
        self.assertEqual(rows[2]["val_loss"], 0.8)
        self.assertIn("monotonic_s", rows[0])

    def test_no_jsonl_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            dashboard = TrainingDashboard(Path(tmp) / "tb", enabled=False)
            dashboard.log_epoch(1, loss=0.5, learning_rate=1e-3, epoch_time_s=1.0)
            dashboard.close()
            self.assertEqual(list(Path(tmp).rglob("*.jsonl")), [])


if __name__ == "__main__":
    unittest.main()
