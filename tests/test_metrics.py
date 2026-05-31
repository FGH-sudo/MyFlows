import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from MyFlows.utils.metrics import (
    DonkeyRegressionEvaluator,
    accuracy,
    angle_sign_accuracy,
    classification_report,
    confusion_matrix,
    mae,
    mse,
    multivariate_regression_metrics,
    precision_recall_f1,
    r2_score,
)
from MyFlows.utils.viz import TrainingHistory, plot_training_curves


class MetricsTest(unittest.TestCase):
    def test_regression_metrics(self):
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([0.1, 0.9, 2.2])
        self.assertAlmostEqual(mse(y_true, y_pred), np.mean((y_true - y_pred) ** 2))
        self.assertAlmostEqual(mae(y_true, y_pred), np.mean(np.abs(y_true - y_pred)))
        self.assertGreater(r2_score(y_true, y_pred), 0.9)

    def test_multivariate_and_sign_accuracy(self):
        y_true = np.array([[0.2, 0.5], [-0.3, 0.5], [0.0, 0.5]])
        y_pred = np.array([[0.1, 0.4], [-0.2, 0.6], [0.01, 0.5]])
        mv = multivariate_regression_metrics(y_true, y_pred, target_names=["angle", "throttle"])
        self.assertIn("angle_mse", mv)
        self.assertIn("throttle_mse", mv)
        sign = angle_sign_accuracy(y_true[:, 0], y_pred[:, 0])
        self.assertEqual(sign["angle_sign_accuracy"], 1.0)

    def test_classification_metrics(self):
        y_true = np.array([0, 1, 2, 1, 0])
        logits = np.array(
            [
                [2.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 2.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
        self.assertEqual(accuracy(y_true, logits), 1.0)
        cm = confusion_matrix(y_true, logits, num_classes=3)
        self.assertEqual(int(cm.sum()), 5)
        prf = precision_recall_f1(y_true, logits, num_classes=3, average="macro")
        self.assertAlmostEqual(prf["f1"], 1.0)
        report = classification_report(y_true, logits, num_classes=3)
        self.assertAlmostEqual(report["accuracy"], 1.0)

    def test_donkey_evaluator(self):
        ev = DonkeyRegressionEvaluator()
        ev.update(np.array([[0.1, 0.5]]), np.array([[0.2, 0.4]]))
        ev.update(np.array([[-0.2, 0.5]]), np.array([[-0.1, 0.6]]))
        m = ev.compute()
        self.assertIn("angle_mse", m)
        self.assertIn("angle_sign_accuracy", m)
        summary = ev.format_summary(m)
        self.assertIn("Eval Summary", summary)

    def test_training_history_plot(self, tmp_path=None):
        import tempfile

        out = Path(tempfile.mkdtemp())
        hist = TrainingHistory()
        hist.record_step(1, 1.0)
        hist.record_step(2, 0.5)
        hist.record_epoch(1, 0.75)
        json_path = hist.save_json(out / "history.json")
        self.assertTrue(json_path.is_file())
        paths = plot_training_curves(hist, out, prefix="unit")
        self.assertTrue(paths)
        for p in paths:
            self.assertTrue(p.is_file())


if __name__ == "__main__":
    unittest.main()
