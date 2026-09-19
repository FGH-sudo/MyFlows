import unittest

from MyFlows.distributed.launcher import run_training
from MyFlows.distributed.model import single_process

import numpy as np


class RingTrainingTest(unittest.TestCase):
    def test_ring_two_workers_match_single_process(self):
        reference = single_process(steps=4)
        result = run_training(mode="ring", transport="grpc_proto", train_workers=2,
                              steps=4, timeout=20, shard_sizes=[16, 16])
        self.assertEqual(result["status"], "passed", result.get("error"))
        self.assertEqual(len(result["history"]), 5)
        for actual, expected in zip(result["history"], reference["history"]):
            for name in actual:
                np.testing.assert_allclose(actual[name], expected[name], atol=1e-4, rtol=1e-3)

    def test_ps_grpc_one_worker_matches_single_process(self):
        reference = single_process(steps=3)
        result = run_training(mode="ps", transport="grpc_proto", train_workers=1,
                              steps=3, timeout=20, shard_sizes=[32])
        self.assertEqual(result["status"], "passed", result.get("error"))
        for actual, expected in zip(result["history"], reference["history"]):
            for name in actual:
                np.testing.assert_allclose(actual[name], expected[name], atol=1e-4, rtol=1e-3)
