import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from MyFlows.core.graph import Graph
from MyFlows.core.node import Variable
from MyFlows.layers.layer import Conv2D, Dense, Flatten
from MyFlows.ops.activation import ReLU
from MyFlows.ops.loss import CrossEntropy, LogLoss
from MyFlows.train.opt import Adam, MBGD


def binary_accuracy(logits, labels):
    preds = np.where(logits >= 0, 1, -1)
    return np.mean(preds == labels)


def multiclass_accuracy(logits, labels):
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == labels.reshape(-1))


class FrameworkIntegrationTest(unittest.TestCase):
    def test_dense_model_learns_linearly_separable_data(self):
        rng = np.random.default_rng(0)
        negative = rng.normal(loc=(-2.0, -2.0), scale=0.35, size=(32, 2))
        positive = rng.normal(loc=(2.0, 2.0), scale=0.35, size=(32, 2))
        x_data = np.vstack([negative, positive])
        y_data = np.vstack([
            -np.ones((32, 1), dtype=np.float64),
            np.ones((32, 1), dtype=np.float64),
        ])

        np.random.seed(0)
        x_node = Variable(x_data, name="x")
        y_node = Variable(y_data, name="y")
        dense = Dense(2, 1, name="linear")
        logits = dense(x_node)
        loss = LogLoss(logits, y_node)
        graph = Graph(loss)
        optimizer = MBGD(graph, learning_rate=0.2)

        graph.forward()
        initial_loss = float(loss.value)

        for _ in range(120):
            optimizer.one_step()
            optimizer.update()

        graph.forward()
        final_loss = float(loss.value)
        final_acc = binary_accuracy(logits.value, y_data)

        self.assertLess(final_loss, initial_loss)
        self.assertLess(final_loss, 0.01)
        self.assertGreater(final_acc, 0.98)

    def test_tiny_cnn_overfits_simple_patterns(self):
        x_data = np.array(
            [
                [[[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]]],
                [[[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0]]],
                [[[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]],
                [[[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]]],
                [[[1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1], [0, 0, 0, 0]]],
                [[[0, 1, 1, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]]],
                [[[1, 1, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]],
                [[[0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]]],
            ],
            dtype=np.float64,
        )
        y_data = np.array([[0], [0], [1], [1], [0], [0], [1], [1]], dtype=np.int64)

        np.random.seed(1)
        x_node = Variable(x_data, name="images")
        y_node = Variable(y_data, name="labels")

        conv = Conv2D(1, 2, kernel_size=2, stride=1, padding=0, activation=ReLU, name="conv")
        flatten = Flatten(name="flatten")
        dense = Dense(18, 2, name="classifier")

        logits = dense(flatten(conv(x_node)))
        loss = CrossEntropy(logits, y_node)
        graph = Graph(loss)
        optimizer = Adam(graph, learning_rate=0.03)

        graph.forward()
        initial_loss = float(loss.value)

        for _ in range(160):
            optimizer.one_step()
            optimizer.update()

        graph.forward()
        final_loss = float(loss.value)
        final_acc = multiclass_accuracy(logits.value, y_data)

        self.assertLess(final_loss, initial_loss * 0.35)
        self.assertGreater(final_acc, 0.95)


if __name__ == "__main__":
    unittest.main()
