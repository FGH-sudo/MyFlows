import math
import unittest

from MyFlows.distributed.launcher import _outputs_from_events
from MyFlows.distributed.metrics_reduce import reduce_step_metrics


def _local_row(worker, n_samples=4, epoch=0, step=0, **stats):
    return {
        "worker_id": worker,
        "epoch": epoch,
        "global_step": step,
        "n_samples": n_samples,
        "loss": 1.0,
        "aggregation": "local",
        "scope": "train_batch",
        "stats": stats,
    }


def _loss_acc(n, correct):
    return {
        "loss": {"sum": 1.0 * n, "count": n, "kind": "mean"},
        "accuracy": {"sum": float(correct), "count": n, "kind": "mean"},
    }


class MetricsReduceTest(unittest.TestCase):
    def test_uneven_shards_use_correct_denominators(self):
        rows = [
            {
                "worker_id": 0, "epoch": 0, "global_step": 0, "n_samples": 5,
                "loss": 1.0, "aggregation": "local", "scope": "train_batch",
                "stats": {
                    "loss": {"sum": 5.0, "count": 5, "kind": "mean"},
                    "accuracy": {"sum": 5.0, "count": 5, "kind": "mean"},
                    "mae": {"sum": 1.0, "count": 5, "kind": "mean"},
                    "rmse": {"sum": 4.0, "count": 5, "kind": "rmse"},
                    "mse": {"sum": 4.0, "count": 5, "kind": "mean"},
                },
            },
            {
                "worker_id": 1, "epoch": 0, "global_step": 0, "n_samples": 3,
                "loss": 3.0, "aggregation": "local", "scope": "train_batch",
                "stats": {
                    "loss": {"sum": 9.0, "count": 3, "kind": "mean"},
                    "accuracy": {"sum": 0.0, "count": 3, "kind": "mean"},
                    "mae": {"sum": 9.0, "count": 3, "kind": "mean"},
                    "rmse": {"sum": 27.0, "count": 3, "kind": "rmse"},
                    "mse": {"sum": 27.0, "count": 3, "kind": "mean"},
                },
            },
        ]
        merged, issues = reduce_step_metrics(rows, n_workers=2)
        self.assertEqual(issues, [])
        self.assertEqual(merged["n_samples"], 8)
        self.assertEqual(merged["aggregation"], "global")
        self.assertEqual(merged["scope"], "train_batch")
        self.assertAlmostEqual(merged["loss"], 14.0 / 8.0)
        self.assertAlmostEqual(merged["accuracy"], 5.0 / 8.0)
        self.assertAlmostEqual(merged["mae"], 10.0 / 8.0)
        self.assertAlmostEqual(merged["mse"], 31.0 / 8.0)
        self.assertAlmostEqual(merged["rmse"], math.sqrt(31.0 / 8.0))
        naive_rmse = (math.sqrt(4.0 / 5.0) * 5 + math.sqrt(27.0 / 3.0) * 3) / 8.0
        self.assertNotAlmostEqual(merged["rmse"], naive_rmse)

    def test_missing_worker_is_not_silently_reduced(self):
        rows = [{
            "worker_id": 0, "epoch": 0, "global_step": 1, "n_samples": 4,
            "loss": 0.5, "aggregation": "local", "scope": "train_batch",
            "stats": {"loss": {"sum": 2.0, "count": 4, "kind": "mean"}},
        }]
        merged, issues = reduce_step_metrics(rows, n_workers=2)
        self.assertIsNone(merged)
        self.assertTrue(any("missing" in item for item in issues))

    def test_does_not_reaggregate_already_global_rows(self):
        row = {
            "worker_id": 0, "epoch": 0, "global_step": 0, "n_samples": 32,
            "loss": 1.25, "aggregation": "global", "scope": "train_batch",
            "stats": {"loss": {"sum": 40.0, "count": 32, "kind": "mean"}},
        }
        merged, issues = reduce_step_metrics([row], n_workers=1)
        self.assertEqual(issues, [])
        self.assertAlmostEqual(merged["loss"], 1.25)
        self.assertEqual(merged["aggregation"], "global")

    def test_missing_accuracy_on_one_worker_is_not_global(self):
        rows = [
            _local_row(0, 4, **_loss_acc(4, 4)),
            _local_row(1, 4, loss={"sum": 4.0, "count": 4, "kind": "mean"}),
        ]
        merged, issues = reduce_step_metrics(rows, n_workers=2)
        self.assertIsNone(merged)
        self.assertTrue(any("accuracy" in item for item in issues))

    def test_missing_loss_is_rejected(self):
        rows = [
            _local_row(0, 4, accuracy={"sum": 4.0, "count": 4, "kind": "mean"}),
            _local_row(1, 4, accuracy={"sum": 2.0, "count": 4, "kind": "mean"}),
        ]
        merged, issues = reduce_step_metrics(rows, n_workers=2)
        self.assertIsNone(merged)
        self.assertTrue(any("loss" in item for item in issues))

    def test_kind_conflict_is_rejected(self):
        rows = [
            _local_row(0, 4, loss={"sum": 4.0, "count": 4, "kind": "mean"}),
            _local_row(1, 4, loss={"sum": 4.0, "count": 4, "kind": "rmse"}),
        ]
        merged, issues = reduce_step_metrics(rows, n_workers=2)
        self.assertIsNone(merged)
        self.assertTrue(any("kind" in item for item in issues))

    def test_nonfinite_sum_and_illegal_count_are_rejected(self):
        nan_rows = [
            _local_row(0, 4, loss={"sum": float("nan"), "count": 4, "kind": "mean"}),
            _local_row(1, 4, loss={"sum": 4.0, "count": 4, "kind": "mean"}),
        ]
        merged, issues = reduce_step_metrics(nan_rows, n_workers=2)
        self.assertIsNone(merged)
        self.assertTrue(any("finite" in item or "non-finite" in item or "nan" in item for item in issues))
        bad_count = [
            _local_row(0, 4, loss={"sum": 4.0, "count": 0, "kind": "mean"}),
            _local_row(1, 4, loss={"sum": 4.0, "count": 4, "kind": "mean"}),
        ]
        merged, issues = reduce_step_metrics(bad_count, n_workers=2)
        self.assertIsNone(merged)
        self.assertTrue(any("count" in item for item in issues))

    def test_epoch_or_step_mismatch_is_rejected(self):
        rows = [
            _local_row(0, 4, epoch=0, step=0, **_loss_acc(4, 1)),
            _local_row(1, 4, epoch=1, step=0, **_loss_acc(4, 1)),
        ]
        merged, issues = reduce_step_metrics(rows, n_workers=2)
        self.assertIsNone(merged)
        self.assertTrue(any("epoch" in item for item in issues))

    def test_multi_output_uneven_element_counts_still_reduce(self):
        rows = [
            _local_row(
                0, 5,
                loss={"sum": 10.0, "count": 10, "kind": "mean"},
                mse={"sum": 10.0, "count": 10, "kind": "mean"},
                mae={"sum": 4.0, "count": 10, "kind": "mean"},
                rmse={"sum": 10.0, "count": 10, "kind": "rmse"},
            ),
            _local_row(
                1, 3,
                loss={"sum": 6.0, "count": 6, "kind": "mean"},
                mse={"sum": 6.0, "count": 6, "kind": "mean"},
                mae={"sum": 3.0, "count": 6, "kind": "mean"},
                rmse={"sum": 6.0, "count": 6, "kind": "rmse"},
            ),
        ]
        merged, issues = reduce_step_metrics(rows, n_workers=2)
        self.assertEqual(issues, [])
        self.assertEqual(merged["n_samples"], 8)
        self.assertAlmostEqual(merged["mse"], 16.0 / 16.0)
        self.assertAlmostEqual(merged["mae"], 7.0 / 16.0)
        self.assertAlmostEqual(merged["rmse"], math.sqrt(16.0 / 16.0))

    def test_outputs_from_events_marks_partial_accuracy_invalid(self):
        events = [
            {
                "phase": "history", "worker_id": 0,
                "metrics": [_local_row(0, 4, **_loss_acc(4, 4))],
                "losses": [1.0],
            },
            {
                "phase": "history", "worker_id": 1,
                "metrics": [_local_row(1, 4, loss={"sum": 4.0, "count": 4, "kind": "mean"})],
                "losses": [1.0],
            },
            {
                "phase": "data_meta", "worker_id": 0, "source": "injected",
                "n_train": 8, "n_val": 0, "n_test": 0, "seed": 0,
                "split_fingerprint": "same-fp",
            },
            {
                "phase": "data_meta", "worker_id": 1, "source": "injected",
                "n_train": 8, "n_val": 0, "n_test": 0, "seed": 0,
                "split_fingerprint": "same-fp",
            },
        ]
        outputs = _outputs_from_events(events, 2)
        self.assertFalse(outputs["metrics_valid"])
        self.assertTrue(outputs["metrics_issues"])
        self.assertFalse(outputs["metrics"])
        self.assertFalse(outputs["losses"])
        self.assertTrue(any("accuracy" in item for item in outputs["metrics_issues"]))


if __name__ == "__main__":
    unittest.main()
