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
    def test_constant_fold_add_chain(self):
        a = Variable(np.array(1.0), constant=True)
        b = Variable(np.array(2.0), constant=True)
        c = Variable(np.array(3.0), constant=True)
        s = Add(Add(a, b), c)
        g = Graph(s, optimize=True, fold_constants=True)
        self.assertEqual(g._opt_stats["constant_folds"], 2)
        self.assertEqual(len(g.sorted_nodes), 1)
        g.forward()
        self.assertAlmostEqual(float(g.target_node.value), 6.0)

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
