import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from MyFlows.core.graph import Graph
from MyFlows.core.node import Variable
from MyFlows.core.tensor import Tensor
from MyFlows.train.opt import MBGD


class TensorWrapperTest(unittest.TestCase):
    def test_variable_uses_tensor_storage_for_value_and_grad(self):
        value = np.array([[1.0, 2.0], [3.0, 4.0]])
        node = Variable(value, trainable=True, name="weights")

        self.assertIsInstance(node.tensor, Tensor)
        self.assertTrue(np.allclose(node.value, value))

        node.clear_grad()
        self.assertIsInstance(node.grad_tensor, Tensor)
        self.assertTrue(np.allclose(node.grad, np.zeros_like(value)))

        node.grad = Tensor.ones_like(node.tensor)
        self.assertTrue(np.allclose(node.grad, np.ones_like(value)))

    def test_optimizer_accepts_tensor_gradients(self):
        node = Variable(np.array([[1.0, -2.0]]), trainable=True, name="weights")
        optimizer = MBGD(Graph(node), learning_rate=0.1)

        optimizer.update({node: Tensor([[0.5, -1.0]])})

        self.assertTrue(np.allclose(node.value, np.array([[0.95, -1.9]])))


if __name__ == "__main__":
    unittest.main()
