import unittest

import numpy as np

from MyFlows.core.device import set_device
from MyFlows.layers.layer import Conv2D, Dense
from MyFlows.utils.initializers import make_initializer


class SeededLayersTest(unittest.TestCase):
    def test_shared_initializer_reproduces_model_without_repeating_draws(self):
        set_device("cpu")

        def build(seed):
            init = make_initializer(seed=seed)
            layers = [Conv2D(1, 4, initializer=init),
                      Dense(16, 16, initializer=init), Dense(16, 16, initializer=init)]
            return [np.asarray(p.value).copy() for layer in layers for p in layer.params]

        first, second, different = build(0), build(0), build(1)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
        self.assertFalse(np.array_equal(first[2], first[4]))
        self.assertFalse(np.array_equal(first[0], different[0]))
