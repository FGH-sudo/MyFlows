import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from MyFlows.core.device import cuda_available, get_device, set_device, use_cuda, asnumpy
from MyFlows.core.graph import Graph
from MyFlows.core.node import Variable
from MyFlows.layers.layer import Dense
from MyFlows.ops.activation import ReLU
from MyFlows.ops.loss import MSELoss
from MyFlows.train.opt import Adam


class DeviceSmokeTest(unittest.TestCase):
    def tearDown(self):
        set_device("cpu")

    def test_cpu_dense_step(self):
        set_device("cpu")
        x = Variable(np.random.randn(8, 4).astype(np.float64), name="x")
        y = Variable(np.random.randn(8, 2).astype(np.float64), name="y")
        pred = Dense(4, 2, activation=ReLU)(x)
        loss = MSELoss(pred, y)
        opt = Adam(Graph(loss), learning_rate=0.05)
        opt.one_step()
        opt.update()
        self.assertEqual(get_device(), "cpu")

    @unittest.skipUnless(cuda_available(), "CUDA/CuPy 不可用")
    def test_cuda_dense_step(self):
        use_cuda()
        self.assertEqual(get_device(), "cuda")
        x = Variable(np.random.randn(8, 4).astype(np.float64), name="x")
        y = Variable(np.random.randn(8, 2).astype(np.float64), name="y")
        pred = Dense(4, 2, activation=ReLU)(x)
        loss = MSELoss(pred, y)
        graph = Graph(loss)
        opt = Adam(graph, learning_rate=0.05)
        graph.forward()
        before = float(asnumpy(loss.value))
        for _ in range(5):
            opt.one_step()
            opt.update()
        graph.forward()
        after = float(asnumpy(loss.value))
        self.assertLess(after, before)


if __name__ == "__main__":
    unittest.main()
