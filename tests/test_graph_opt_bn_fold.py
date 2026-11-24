# -*- coding: utf-8 -*-
"""Conv+BN 推理态折叠数值与节点数测试。"""

import unittest

import numpy as np

from MyFlows.core.device import asnumpy, xp
from MyFlows.core.graph import Graph
from MyFlows.core.graph_opt import apply_graph_optimizations, fold_bn_into_conv
from MyFlows.core.node import Variable
from MyFlows.layers.resnet import BatchNorm2d, Conv2D, ResNet18
from MyFlows.ops.activation import ReLU
from MyFlows.ops.batchnorm import BatchNorm2d_Op
from MyFlows.ops.convolution import Conv2D_Op, Conv2D_ReLU_Op


class TestGraphOptBnFold(unittest.TestCase):
  def test_resnet18_eval_fold_matches(self):
    model = ResNet18(in_channels=3, output_dim=5, stem="cifar", base_width=16)
    model.eval()
    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((1, 3, 32, 32), dtype=np.float64) * 0.1
    x = Variable(xp.asarray(x_np), name="x")

    out1 = model(x)
    g1 = Graph(out1)
    g1.forward()
    y1 = np.asarray(asnumpy(out1.value))

    out2 = model(x)
    root2, stats = apply_graph_optimizations(out2, mode="inference")
    self.assertGreater(stats["bn_folds"], 0)
    g2 = Graph(root2)
    g2.forward()
    y2 = np.asarray(asnumpy(root2.value))
    np.testing.assert_allclose(y1, y2, rtol=1e-4, atol=1e-4)

  def test_fold_does_not_mutate_original_conv_weights(self):
    model = ResNet18(in_channels=3, output_dim=5, stem="cifar", base_width=16)
    model.eval()
    original = np.asarray(asnumpy(model.stem_conv.kernel.value)).copy()
    x = Variable(xp.zeros((1, 3, 32, 32)), name="x")
    out = model(x)
    fold_bn_into_conv(out)
    after = np.asarray(asnumpy(model.stem_conv.kernel.value))
    np.testing.assert_array_equal(original, after)

  def test_node_count_decreases(self):
    model = ResNet18(in_channels=3, output_dim=5, stem="cifar", base_width=16)
    model.eval()
    x = Variable(xp.zeros((1, 3, 32, 32)), name="x")
    out = model(x)
    g_before = Graph(out)
    n_bn_before = sum(1 for n in g_before.nodes if isinstance(n, BatchNorm2d_Op))
    self.assertGreater(n_bn_before, 0)
    root, n_folded = fold_bn_into_conv(out)
    g_after = Graph(root)
    n_bn_after = sum(1 for n in g_after.nodes if isinstance(n, BatchNorm2d_Op))
    self.assertGreater(n_folded, 0)
    self.assertLess(n_bn_after, n_bn_before)

  def test_inference_mode_folds_bn_before_fusing_conv_relu(self):
    conv = Conv2D(2, 3, kernel_size=3, padding=1, bias=False, name="c")
    bn = BatchNorm2d(3, name="bn")
    bn.running_mean[:] = xp.asarray([0.2, -0.3, 0.4])
    bn.running_var[:] = xp.asarray([0.8, 1.2, 0.5])
    bn.eval()

    x = Variable(xp.asarray(np.random.default_rng(1).standard_normal((1, 2, 5, 5))), name="x")
    ref = ReLU(bn(conv(x)))
    Graph(ref).forward()
    expected = np.asarray(asnumpy(ref.value))

    out = ReLU(bn(conv(x)))
    root, stats = apply_graph_optimizations(out, mode="inference")
    self.assertEqual(stats["bn_folds"], 1)
    self.assertEqual(stats["conv_act_fusions"], 1)
    self.assertIsInstance(root, Conv2D_ReLU_Op)
    Graph(root).forward()
    np.testing.assert_allclose(np.asarray(asnumpy(root.value)), expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
  unittest.main()
