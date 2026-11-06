"""Per-group Adam math, LR rescaling, frozen-step counters and checkpoint resume."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

import MyFlows as ms
from MyFlows.core.device import get_device, set_device
from MyFlows.layers.layer import Layer


class Pair(Layer):
    def __init__(self):
        super().__init__("pair")
        self.left = ms.Variable(np.array([1., 2.], np.float32), trainable=True, name="left")
        self.right = ms.Variable(np.array([-1., 3.], np.float32), trainable=True, name="right")
        self.params = [self.left, self.right]


class ParameterGroupTest(unittest.TestCase):
    def setUp(self):
        self.device = get_device()
        set_device("cpu")

    def tearDown(self):
        set_device(self.device)

    def build(self, grouped=True, decay=0.):
        pair = Pair()
        graph = ms.Graph(ms.Add(pair.left, pair.right))
        groups = [{"name": "backbone", "params": [pair.left], "lr_mult": .1, "weight_decay": decay},
                  {"name": "head", "params": [pair.right], "lr": .02}] if grouped else None
        return pair, ms.Adam(graph, learning_rate=.01, param_groups=groups)

    def test_adam_groups_match_hand_calculation_with_coupled_decay(self):
        model, opt = self.build(decay=.03)
        params = [p.value.astype(np.float64) for p in model.params]
        first = [np.zeros(2) for _ in params]
        second = [np.zeros(2) for _ in params]
        for step in range(1, 4):
            gradients = [np.array([step * .2, -.3], np.float32), np.array([-.4, step * .1], np.float32)]
            opt.update(dict(zip(model.params, gradients)))
            for i, (lr, decay) in enumerate(((.001, .03), (.02, 0.))):
                grad = gradients[i].astype(np.float64) + decay * params[i]
                first[i] = .9 * first[i] + .1 * grad
                second[i] = .999 * second[i] + .001 * grad**2
                params[i] -= lr * (first[i]/(1-.9**step)) / (np.sqrt(second[i]/(1-.999**step)) + 1e-8)
                np.testing.assert_allclose(model.params[i].value, params[i], atol=3e-7, rtol=2e-7)
        opt.learning_rate = .1
        self.assertEqual(opt.group_learning_rates(), {"backbone": .010000000000000002, "head": .02})

    def test_all_multipliers_one_are_bitwise_equivalent(self):
        a, plain = self.build(False)
        b, grouped = self.build(False)
        grouped.set_param_groups([{"name": "all", "params": b.params, "lr_mult": 1.}])
        for _ in range(4):
            g = np.array([.3, -.8], np.float32)
            plain.update({p: g for p in a.params})
            grouped.update({p: g for p in b.params})
        for x, y in zip(a.params, b.params):
            self.assertEqual(x.value.tobytes(), y.value.tobytes())

    def test_resume_across_unfreeze_keeps_group_rates_and_per_parameter_steps(self):
        model, opt = self.build()
        model.right.trainable = False
        g = np.array([.2, -.5], np.float32)
        for _ in range(3):
            opt.update({model.left: g})
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "state"
            ms.save_checkpoint([model], opt, 0, filepath=str(path))
            resumed, restored = self.build()
            ms.load_checkpoint([resumed], restored, str(path))
            model.right.trainable = True
            for _ in range(2):
                opt.update({p: g for p in model.params})
                restored.update({p: g for p in resumed.params})
            self.assertEqual(opt.parameter_steps[model.right], 2)
            self.assertEqual(restored.parameter_steps[resumed.right], 2)
            self.assertEqual(restored.group_learning_rates(), opt.group_learning_rates())
            for a, b in zip(model.params, resumed.params):
                self.assertEqual(a.value.tobytes(), b.value.tobytes())

    def test_group_validation_rejects_duplicate_members_and_invalid_lr(self):
        model, opt = self.build(False)
        for groups in ([{"params": [model.left, model.left]}], [{"params": model.params, "lr": -1}],
                       [{"params": model.params, "lr": .1, "lr_mult": .1}]):
            with self.assertRaises(ValueError):
                opt.set_param_groups(groups)
