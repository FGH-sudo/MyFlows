import unittest

import numpy as np

from MyFlows.distributed.launcher import run_ps
from MyFlows.distributed.model import SmallMLP, single_process
from MyFlows.distributed.protocol import aggregate_weighted, payload_hash, schema_hash


class ParameterServerTest(unittest.TestCase):
    def test_one_worker_one_step_matches_single_process(self):
        reference = single_process(steps=1)
        result = run_ps(workers=1, steps=1, timeout=15, shard_sizes=[32])
        self.assertEqual(result["status"], "passed", result.get("error"))
        self.assertEqual(len(result["history"]), 2)
        for name in reference["history"][-1]:
            np.testing.assert_allclose(result["history"][-1][name], reference["history"][-1][name],
                                       atol=1e-4, rtol=1e-3)

    def test_one_two_workers_equal_single_process_all_twenty_updates(self):
        reference = single_process()
        for workers, shards in ((1, [32]), (2, [16, 16]), (2, [20, 12])):
            with self.subTest(workers=workers, shards=shards):
                result = run_ps(workers=workers, shard_sizes=shards, timeout=15)
                self.assertEqual(result["status"], "passed", result.get("error"))
                self.assertEqual(len(result["history"]), 21)
                for actual, expected in zip(result["history"], reference["history"]):
                    for name in actual:
                        np.testing.assert_allclose(actual[name], expected[name], atol=1e-4, rtol=1e-3)
                np.testing.assert_allclose(result["losses"], reference["losses"], atol=1e-4, rtol=1e-3)
                self.assertFalse(result["alive_pids"])
                self.assertLess(result["cleanup_s"], 5)
                self.assertTrue(all(code == 0 for code in result["exitcodes"].values()))
                acks = [e for e in result["events"] if e["phase"] == "gradient_acknowledged"]
                self.assertEqual(len(acks), 20 * workers)
                server_updates = [e for e in result["events"] if e.get("phase") == "optimizer_gpu"]
                self.assertEqual(server_updates, [])

    def test_faults_fail_and_cleanup(self):
        for fault in ("crash", "timeout", "duplicate", "schema"):
            with self.subTest(fault=fault):
                result = run_ps(workers=2, steps=2, timeout=5, fault=fault)
                self.assertEqual(result["status"], "failed")
                self.assertFalse(result["alive_pids"])
                self.assertLess(result["cleanup_s"], 5)
                self.assertTrue(any(code != 0 for code in result["exitcodes"].values()))

    def test_duplicate_in_final_single_worker_step_is_not_accepted(self):
        result = run_ps(workers=1, steps=1, timeout=1, fault="duplicate")
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["alive_pids"])

    def test_invalid_shards_rejected_before_spawn(self):
        for sizes in ([32, 0], [20, 10], [16.0, 16], [-1, 33]):
            with self.assertRaises(ValueError):
                run_ps(workers=2, shard_sizes=sizes)

    def test_external_update_requires_empty_optimizer_cache(self):
        model = SmallMLP()
        model.optimizer.acc_no = 1
        with self.assertRaises(RuntimeError):
            model.update({name: np.ones_like(p.value) for name, p in model.params.items()})

    def test_gpu_worker_one_step_when_cuda_available(self):
        from MyFlows.core.device import cuda_available
        if not cuda_available():
            self.skipTest("CUDA not available")
        reference = single_process(steps=1, device="cuda")
        result = run_ps(workers=1, steps=1, timeout=30, shard_sizes=[32], device="cuda")
        self.assertEqual(result["status"], "passed", result.get("error"))
        for name in reference["history"][-1]:
            np.testing.assert_allclose(result["history"][-1][name], reference["history"][-1][name],
                                       atol=1e-4, rtol=1e-3)

    def test_weighted_mean_and_schema_hash(self):
        reference = SmallMLP().snapshot()
        gradients = {name: np.ones_like(value) for name, value in reference.items()}
        message = {"n_samples": 20, "loss": 1.0, "gradients": gradients}
        second = {"n_samples": 12, "loss": 3.0,
                  "gradients": {name: value * 3 for name, value in gradients.items()}}
        mean, loss, count = aggregate_weighted([message, second])
        self.assertEqual(count, 32)
        self.assertEqual(loss, 1.75)
        for value in mean.values():
            np.testing.assert_allclose(value, np.full_like(value, 1.75), atol=1e-6)
        self.assertEqual(schema_hash(reference), schema_hash(dict(reference)))
        self.assertEqual(payload_hash(gradients), payload_hash(dict(gradients)))
