import unittest

import numpy as np

import sys
from pathlib import Path

PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from MyFlows.core.graph import Graph
from MyFlows.core.node import Variable
from MyFlows.ops.basic import Add


class GraphOptTest(unittest.TestCase):
    def test_deep_graph_can_run_with_and_without_optimizations(self):
        depth = sys.getrecursionlimit() + 25
        start = Variable(np.array(0.0))
        increment = Variable(np.array(1.0))
        result = start
        for _ in range(depth):
            result = Add(result, increment)
        for optimize in (False, True):
            with self.subTest(optimize=optimize):
                graph = Graph(result, optimize=optimize, fold_constants=True)
                self.assertEqual(len(graph.nodes), depth + 2)
                graph.forward()
                graph.backward()
                self.assertEqual(float(result.value), depth)
                self.assertEqual(float(start.grad), 1.)
                self.assertEqual(float(increment.grad), depth)

    def test_graph_rejects_a_cycle_before_execution(self):
        value = Variable(np.array(1.0))
        first = Add(value, value, name="first")
        last = Add(first, value)
        first.parents[0] = last
        for optimize in (False, True):
            with self.subTest(optimize=optimize), self.assertRaisesRegex(ValueError, "cycle in computation graph"):
                Graph(last, optimize=optimize, fold_constants=True)

    def test_constant_fold_add_chain(self):
        a = Variable(np.array(1.0), constant=True)
        b = Variable(np.array(2.0), constant=True)
        c = Variable(np.array(3.0), constant=True)
        self.assertFalse(a.requires_grad)
        s = Add(Add(a, b), c)
        g = Graph(s, optimize=True, fold_constants=True)
        self.assertEqual(g._opt_stats["constant_folds"], 2)
        self.assertEqual(len(g.sorted_nodes), 1)
        self.assertFalse(g.target_node.requires_grad)
        g.forward()
        self.assertAlmostEqual(float(g.target_node.value), 6.0)

        x = Variable(np.array(4.0), trainable=True)
        folded_plus_x = Add(Add(a, b), x)
        g2 = Graph(folded_plus_x, optimize=True, fold_constants=True)
        self.assertEqual(g2._opt_stats["constant_folds"], 1)
        folded_leaf = next(p for p in g2.target_node.parents if p is not x)
        self.assertFalse(folded_leaf.requires_grad)
        g2.forward()
        g2.backward()
        self.assertIsNone(folded_leaf.grad)
        self.assertEqual(float(x.grad), 1.0)

    def test_parameter_requires_grad_cannot_contradict_trainable(self):
        param = Variable(np.array(1.0), trainable=True)
        with self.assertRaisesRegex(ValueError, "trainable"):
            param.requires_grad = False
        with self.assertRaisesRegex(ValueError, "trainable"):
            Variable(np.array(1.0), trainable=True, requires_grad=False)

    def test_conv2d_fused_bias_single_op(self):
        from MyFlows.layers.layer import Conv2D
        from MyFlows.ops.convolution import Conv2D_Op

        x = Variable(np.random.randn(1, 2, 4, 4))
        layer = Conv2D(2, 3, kernel_size=3, padding=1, name="c")
        out = layer(x)
        self.assertIsInstance(out, Conv2D_Op)
        self.assertIsNotNone(out.bias)

    def test_dense_fused_linear_single_op(self):
        from MyFlows.layers.layer import Dense
        from MyFlows.ops.basic import Linear

        x = Variable(np.random.randn(4, 3))
        layer = Dense(3, 2, name="d")
        out = layer(x)
        self.assertIsInstance(out, Linear)

    def test_graph_fuses_conv_relu_when_enabled(self):
        from MyFlows.layers.layer import Conv2D
        from MyFlows.ops.activation import ReLU
        from MyFlows.ops.convolution import Conv2D_ReLU_Op

        x = Variable(np.random.randn(1, 2, 5, 5))
        conv = Conv2D(2, 3, kernel_size=3, padding=1, activation=ReLU, name="c")
        y = conv(x)  # ReLU(Conv2D_Op(...))
        g = Graph(y, optimize=True, fuse_activations=True)
        self.assertIsInstance(g.target_node, Conv2D_ReLU_Op)

    def test_convtranspose2d_fused_bias_single_op(self):
        from MyFlows.layers.layer import ConvTranspose2D
        from MyFlows.ops.convolution import ConvTranspose2D_Op

        x = Variable(np.random.randn(1, 2, 4, 4))
        layer = ConvTranspose2D(2, 3, kernel_size=3, stride=2, padding=1, output_padding=1, name="t")
        out = layer(x)
        self.assertIsInstance(out, ConvTranspose2D_Op)
        self.assertIsNotNone(out.bias)

    def test_graph_fuses_convtranspose_relu_when_enabled(self):
        from MyFlows.layers.layer import ConvTranspose2D
        from MyFlows.ops.activation import ReLU
        from MyFlows.ops.convolution import ConvTranspose2D_ReLU_Op

        x = Variable(np.random.randn(1, 2, 4, 4))
        deconv = ConvTranspose2D(2, 3, kernel_size=3, stride=2, padding=1, output_padding=1, activation=ReLU, name="t")
        y = deconv(x)
        g = Graph(y, optimize=True, fuse_activations=True)
        self.assertIsInstance(g.target_node, ConvTranspose2D_ReLU_Op)

    def test_graph_fuses_conv_leaky_relu_when_enabled(self):
        from MyFlows.layers.layer import Conv2D
        from MyFlows.ops.activation import LeakyReLU
        from MyFlows.ops.convolution import Conv2D_LeakyReLU_Op

        x = Variable(np.random.randn(1, 2, 5, 5))
        conv = Conv2D(2, 3, kernel_size=3, padding=1, name="c")
        y = LeakyReLU(conv(x), alpha=0.2)
        g = Graph(y, optimize=True, fuse_activations=True)
        self.assertIsInstance(g.target_node, Conv2D_LeakyReLU_Op)
        self.assertAlmostEqual(g.target_node.alpha, 0.2)

    def test_graph_fuses_convtranspose_leaky_relu_when_enabled(self):
        from MyFlows.layers.layer import ConvTranspose2D
        from MyFlows.ops.activation import LeakyReLU
        from MyFlows.ops.convolution import ConvTranspose2D_LeakyReLU_Op

        x = Variable(np.random.randn(1, 2, 4, 4))
        deconv = ConvTranspose2D(2, 3, kernel_size=3, stride=2, padding=1, output_padding=1, name="t")
        y = LeakyReLU(deconv(x), alpha=0.15)
        g = Graph(y, optimize=True, fuse_activations=True)
        self.assertIsInstance(g.target_node, ConvTranspose2D_LeakyReLU_Op)
        self.assertAlmostEqual(g.target_node.alpha, 0.15)


if __name__ == "__main__":
    unittest.main()
