import unittest

import numpy as np

from MyFlows.core.graph import Graph
from MyFlows.core.node import Variable
from MyFlows.layers.layer import Dense, Dropout
from MyFlows.ops.basic import MatMul
from MyFlows.train.regularization import RegularizationConfig, apply_regularization
from MyFlows.utils.initializers import make_initializer
from MyFlows.utils.model_inspector import inspect_graph, model_summary


class InitializersRegularizationDropoutInspectorTest(unittest.TestCase):
    def test_constant_and_xavier_initializers(self):
        const = make_initializer("constant", value=0.25)
        np.testing.assert_allclose(const((2, 3)), np.full((2, 3), 0.25))

        xavier = make_initializer("xavier_uniform", seed=7)
        values = xavier((4, 8))
        limit = np.sqrt(6.0 / (4 + 8))
        self.assertEqual(values.shape, (4, 8))
        self.assertLessEqual(float(np.max(values)), limit)
        self.assertGreaterEqual(float(np.min(values)), -limit)

    def test_regularization_adds_weight_gradient_and_skips_bias_by_default(self):
        weight = Variable(np.array([[1.0, -2.0]], dtype=np.float64), trainable=True, name="dense_W")
        bias = Variable(np.array([[3.0]], dtype=np.float64), trainable=True, name="dense_b")

        class DummyOpt:
            acc_gradient = {
                weight: np.zeros_like(weight.value),
                bias: np.zeros_like(bias.value),
            }

        config = RegularizationConfig(l2=0.1, l1=0.05, regularize_bias_bn=False)
        applied = apply_regularization(DummyOpt(), [weight, bias], config)

        self.assertEqual(applied, 1)
        np.testing.assert_allclose(DummyOpt.acc_gradient[weight], [[0.15, -0.25]])
        np.testing.assert_allclose(DummyOpt.acc_gradient[bias], [[0.0]])

    def test_dropout_train_eval_and_seed_behavior(self):
        x = Variable(np.ones((2, 4), dtype=np.float64), name="x")
        dropout = Dropout(p=0.5, seed=123)
        y = dropout(x)
        graph = Graph(y)

        dropout.train(True)
        graph.forward()
        first = y.value.copy()
        self.assertTrue(np.any(first == 0.0))
        nonzero = first[first != 0.0]
        self.assertTrue(nonzero.size > 0)
        np.testing.assert_allclose(nonzero, np.full(nonzero.shape, 2.0))

        dropout2 = Dropout(p=0.5, seed=123)
        y2 = dropout2(x)
        Graph(y2).forward()
        np.testing.assert_allclose(first, y2.value)

        dropout.eval()
        graph.forward()
        np.testing.assert_allclose(y.value, np.ones((2, 4)))

    def test_model_inspector_reports_summary_and_bad_content(self):
        x = Variable(np.ones((2, 3), dtype=np.float64), name="x")
        dense = Dense(3, 2, name="dense")
        y = dense(x)
        graph = Graph(y)
        graph.forward()

        summary = model_summary(dense)
        self.assertEqual(summary["total_params"], 8)
        self.assertTrue(any(row["name"] == "dense_W" for row in summary["parameters"]))

        bad = MatMul(x, Variable(np.full((3, 2), np.nan), name="nan_w"))
        bad_graph = Graph(bad)
        bad_graph.forward()
        report = inspect_graph(bad_graph, check_shape=True, check_content=True)
        self.assertFalse(report["ok"])
        self.assertTrue(any(item["has_nan"] for item in report["nodes"]))


if __name__ == "__main__":
    unittest.main()
