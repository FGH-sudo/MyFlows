# -*- coding: utf-8 -*-
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

import MyFlows as ms
from MyFlows.core.graph import Graph
from MyFlows.core.node import Variable
from MyFlows.layers.vgg import VGG11, vgg_fc_input_dim
from MyFlows.ops.loss import MSELoss
from MyFlows.train.opt import Adam


class VGGSmokeTest(unittest.TestCase):
    def test_vgg_exports_do_not_include_discrete_label_helper(self):
        self.assertFalse(hasattr(ms, "angle_to_class"))

    def test_vgg_regression_forward_backward(self):
        h, w = 32, 32
        B, output_dim = 2, 2
        np.random.seed(0)
        x = Variable(np.random.randn(B, 3, h, w).astype(np.float64) * 0.1, name="X")
        y = Variable(np.array([[0.1, 0.5], [-0.2, 0.5]], dtype=np.float64), name="y")
        model = VGG11(3, output_dim=output_dim, image_h=h, image_w=w)
        self.assertEqual(model.output_dim, output_dim)
        self.assertEqual(vgg_fc_input_dim(h, w, 512), model._fc_in)
        pred = model(x)
        loss = MSELoss(pred, y)
        graph = Graph(loss)
        graph.forward()
        self.assertEqual(pred.value.shape, (B, output_dim))
        self.assertTrue(np.isfinite(float(loss.value)))
        opt = Adam(graph, learning_rate=0.01)
        opt.one_step()
        self.assertIsNotNone(model.fc3.W.grad)
        opt.update()


if __name__ == "__main__":
    unittest.main()
