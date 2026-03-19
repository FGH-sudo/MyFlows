import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

try:
    from sklearn.datasets import load_digits
except ImportError:  # pragma: no cover
    load_digits = None

from MyFlows.core.graph import Graph
from MyFlows.core.node import Variable
from MyFlows.layers.layer import Conv2D, Dense, Flatten, MaxPool2d
from MyFlows.ops.activation import ReLU
from MyFlows.ops.loss import CrossEntropy
from MyFlows.train.opt import Adam


def accuracy(logits, labels):
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == labels.reshape(-1))


@unittest.skipUnless(load_digits is not None, "scikit-learn is required for the digits smoke test")
class DigitsConvSmokeTest(unittest.TestCase):
    def test_cnn_can_fit_small_digits_subset(self):
        digits = load_digits()
        mask = np.isin(digits.target, [0, 1])
        x_data = digits.images[mask].astype(np.float64) / 16.0
        y_data = digits.target[mask].astype(np.int64)

        train_indices = []
        test_indices = []
        for cls in [0, 1]:
            cls_indices = np.where(y_data == cls)[0]
            train_indices.extend(cls_indices[:60])
            test_indices.extend(cls_indices[60:100])

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        x_train = x_data[train_indices][:, None, :, :]
        y_train = y_data[train_indices].reshape(-1, 1)
        x_test = x_data[test_indices][:, None, :, :]
        y_test = y_data[test_indices].reshape(-1, 1)

        np.random.seed(0)
        x_node = Variable(x_train, name="digits")
        y_node = Variable(y_train, name="labels")

        conv = Conv2D(1, 4, kernel_size=3, stride=1, padding=1, activation=ReLU, name="conv")
        pool = MaxPool2d(2, 2, name="pool")
        flatten = Flatten(name="flatten")
        dense1 = Dense(4 * 4 * 4, 16, activation=ReLU, name="dense1")
        dense2 = Dense(16, 2, name="dense2")

        logits = dense2(dense1(flatten(pool(conv(x_node)))))
        loss = CrossEntropy(logits, y_node)
        graph = Graph(loss)
        optimizer = Adam(graph, learning_rate=0.01)

        graph.forward()
        initial_loss = float(loss.value)

        for _ in range(40):
            optimizer.one_step()
            optimizer.update()

        graph.forward()
        train_loss = float(loss.value)
        train_acc = accuracy(logits.value, y_train)

        x_node.value = x_test
        y_node.value = y_test
        graph.forward()
        test_acc = accuracy(logits.value, y_test)

        self.assertLess(train_loss, initial_loss * 0.1)
        self.assertGreater(train_acc, 0.99)
        self.assertGreater(test_acc, 0.95)


if __name__ == "__main__":
    unittest.main()
