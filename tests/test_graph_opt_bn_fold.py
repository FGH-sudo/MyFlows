# -*- coding: utf-8 -*-
"""Conv+BN 推理态折叠数值与节点数测试。"""

import unittest

import numpy as np

from MyFlows.core.device import asnumpy, xp
from MyFlows.core.graph import Graph
from MyFlows.core.graph_opt import apply_graph_optimizations, fold_bn_into_conv
from MyFlows.core.node import Variable
from MyFlows.layers.resnet import ResNet18
from MyFlows.ops.batchnorm import BatchNorm2d_Op
from MyFlows.ops.convolution import Conv2D_Op


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


if __name__ == "__main__":
  unittest.main()
