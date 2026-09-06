import unittest

import numpy as np

from pathlib import Path
import sys

PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from MyFlows.core.node import Variable
from MyFlows.train.opt import AdaGrad, Adam, MBGD, Momentum, RMSProp


class _DummyGraph:
    def __init__(self, nodes):
        self.nodes = nodes


class OptimizerNoneGradientTest(unittest.TestCase):
    def _nodes(self):
        trained = Variable(np.ones((2, 2), dtype=np.float64), trainable=True, name="trained")
        unused = Variable(np.full((2, 2), 3.0, dtype=np.float64), trainable=True, name="unused")
        return trained, unused

    def _assert_skips_missing_grad(self, opt_cls, **kwargs):
        trained, unused = self._nodes()
        unused_before = unused.value.copy()
        trained_before = trained.value.copy()
        opt = opt_cls(_DummyGraph([trained, unused]), learning_rate=0.1, **kwargs)
        opt.acc_gradient[trained] = np.ones_like(trained.value)
        opt.acc_no = 1
        opt._update()
        np.testing.assert_array_equal(unused.value, unused_before)
        self.assertFalse(np.allclose(trained.value, trained_before))

    def test_mbgd_skips_none_gradient(self):
        self._assert_skips_missing_grad(MBGD)

    def test_momentum_skips_none_gradient(self):
        self._assert_skips_missing_grad(Momentum)

    def test_adagrad_skips_none_gradient(self):
        self._assert_skips_missing_grad(AdaGrad)

    def test_rmsprop_skips_none_gradient(self):
        self._assert_skips_missing_grad(RMSProp)

    def test_adam_skips_none_gradient(self):
        self._assert_skips_missing_grad(Adam)


if __name__ == "__main__":
    unittest.main()
