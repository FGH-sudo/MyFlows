import unittest

import numpy as np

import MyFlows as ms
from MyFlows.core.device import get_device, set_device
from MyFlows.core.node import Node


class LazyGradientTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(set_device, get_device())
        set_device("cpu")

    def test_first_write_owns_its_buffer_and_rounds_each_contribution(self):
        p = ms.Variable(np.zeros((2, 3), np.float32), trainable=True)
        signal = np.array([[1., 2., 3.]], np.float64)
        p.accumulate_grad(signal)
        signal[:] = 10.
        expected = np.array([[1., 2., 3.]] * 2, np.float32)
        np.testing.assert_array_equal(p.grad, expected)
        p.accumulate_grad(signal)
        np.testing.assert_array_equal(p.grad, expected + 10.)
        self.assertEqual(p.grad.dtype, np.float32)

    def test_shared_branches_repeated_backward_and_explicit_retention(self):
        x = ms.Variable(np.array([-2., 3.], np.float32), trainable=True)
        y = ms.ReLU(x)
        output = ms.Add(y, y)
        graph = ms.Graph(output)
        graph.forward()
        for _ in range(2):
            graph.backward()
            np.testing.assert_array_equal(x.grad, [0., 2.])
            self.assertIsNone(y.grad)
            self.assertIsNone(output.grad)
        graph.retain_grad(y)
        graph.forward()
        graph.backward()
        np.testing.assert_array_equal(y.grad, [2., 2.])
        x.grad[:] = 7.
        np.testing.assert_array_equal(y.grad, [2., 2.])

    def test_custom_legacy_node_can_still_accumulate_in_place(self):
        class Double(Node):
            def forward(self, x):
                self.value = x * 2

            def backward(self):
                self.parents[0].grad += self.grad * 2

        x = ms.Variable(np.ones(3, np.float32))
        y = Double(x)
        graph = ms.Graph(ms.Add(y, x))
        graph.forward()
        graph.backward()
        np.testing.assert_array_equal(x.grad, np.full(3, 3.))
        self.assertIsNone(y.grad)
