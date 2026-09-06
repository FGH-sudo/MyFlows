import unittest

import numpy as np

from MyFlows.distributed.launcher import run_ps
from MyFlows.distributed.model import SmallMLP, single_process
from MyFlows.distributed.protocol import aggregate, payload_hash, schema_hash, validate_gradient


class ParameterServerTest(unittest.TestCase):
    def test_one_two_workers_equal_single_process_all_twenty_updates(self):
        reference = single_process()
        for workers, shards in ((1, [32]), (2, [16, 16]), (2, [20, 12])):
            with self.subTest(workers=workers, shards=shards):
                result = run_ps(workers=workers, shard_sizes=shards)
                self.assertEqual(result["status"], "passed", result.get("error"))
                self.assertEqual(len(result["history"]), 21)
                for actual, expected in zip(result["history"], reference["history"]):
                    for name in actual:
                        np.testing.assert_allclose(actual[name], expected[name], atol=1e-8, rtol=1e-6)
                np.testing.assert_allclose(result["losses"], reference["losses"], atol=1e-8, rtol=1e-6)
                self.assertFalse(result["alive_pids"])
                self.assertLess(result["cleanup_s"], 5)
                self.assertTrue(all(code == 0 for code in result["exitcodes"].values()))
                updates = [e for e in result["events"] if e["phase"] == "updated"]
                self.assertEqual([(e["step_id"], e["parameter_version"], e["n_samples"]) for e in updates],
                                 [(i, i + 1, 32) for i in range(20)])
                for worker in range(workers):
                    acks = [e for e in result["events"] if e["phase"] == "gradient_acknowledged" and e["worker_id"] == worker]
                    self.assertEqual(len(acks), 20)
                    self.assertTrue(all(e["n_samples"] == shards[worker] and e["upload_ack_wait_s"] >= 0 for e in acks))

    def test_faults_fail_and_cleanup(self):
        for fault in ("crash", "timeout", "duplicate", "schema"):
            with self.subTest(fault=fault):
                result = run_ps(workers=2, steps=2, timeout=1, fault=fault)
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

    def test_weighted_mean_and_message_validation(self):
        reference = SmallMLP().snapshot()
        gradients = {name: np.ones_like(value) for name, value in reference.items()}
        message = {"kind": "gradient", "run_id": "test", "worker_id": 0, "step_id": 0,
                   "parameter_version": 0, "parameter_hash": payload_hash(reference),
                   "schema_hash": schema_hash(reference), "n_samples": 20, "loss": 1.,
                   "gradients": gradients, "gradient_hash": payload_hash(gradients)}
        kwargs = dict(run_id="test", step=0, parameter_hash=payload_hash(reference), reference=reference,
                      shard_sizes=[20, 12], received={})
        self.assertEqual(validate_gradient(message, **kwargs), 0)
        for key, value in (("run_id", "other"), ("step_id", -1), ("parameter_version", 1),
                           ("schema_hash", "bad"), ("parameter_hash", "bad"), ("n_samples", 0),
                           ("gradient_hash", "bad"), ("worker_id", 3)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_gradient({**message, key: value}, **kwargs)
        with self.assertRaises(ValueError):
            validate_gradient(message, **{**kwargs, "received": {0: message}})
        for array in (np.zeros((1,), np.float64), gradients["fc1.weight"].astype(np.float32),
                      np.full_like(gradients["fc1.weight"], np.nan)):
            with self.assertRaises(ValueError):
                validate_gradient({**message, "gradients": {**gradients, "fc1.weight": array}}, **kwargs)
        second = {**message, "n_samples": 12, "loss": 3.,
                  "gradients": {name: value * 3 for name, value in gradients.items()}}
        mean, loss, count = aggregate([message, second])
        self.assertEqual(count, 32)
        self.assertEqual(loss, 1.75)
        for value in mean.values():
            np.testing.assert_array_equal(value, np.full_like(value, 1.75))
