# -*- coding: utf-8 -*-
import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from MyFlows.core.graph import Graph
from MyFlows.core.node import Variable
from MyFlows.layers.vgg import VGG11, angle_to_class, vgg_fc_input_dim
from MyFlows.ops.loss import CrossEntropy
from MyFlows.train.opt import Adam


class VGGSmokeTest(unittest.TestCase):
    def test_angle_to_class(self):
        self.assertEqual(angle_to_class(-0.3, 5), 0)
        self.assertEqual(angle_to_class(0.0, 3), 1)

    def test_vgg_forward_backward(self):
        h, w = 32, 32
        B, K = 2, 5
        np.random.seed(0)
        x = Variable(np.random.randn(B, 3, h, w).astype(np.float64) * 0.1, name="X")
        y = Variable(np.array([[0], [2]], dtype=np.float64), name="y")
        model = VGG11(3, K, image_h=h, image_w=w)
        self.assertEqual(vgg_fc_input_dim(h, w, 512), model._fc_in)
        logits = model(x)
        loss = CrossEntropy(logits, y)
        graph = Graph(loss)
        graph.forward()
        self.assertTrue(np.isfinite(float(loss.value)))
        opt = Adam(graph, learning_rate=0.01)
        opt.one_step()
        self.assertIsNotNone(model.fc3.W.grad)
        opt.update()


if __name__ == "__main__":
    unittest.main()
