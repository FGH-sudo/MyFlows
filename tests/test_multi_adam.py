from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from MyFlows.core.device import asnumpy, cuda_available, get_device, set_device, xp
from MyFlows.core.node import Variable
from MyFlows.ops import fused
from MyFlows.train.opt import Adam


@unittest.skipUnless(cuda_available(), "multi-tensor Adam uses CUDA")
class MultiAdamTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(set_device, get_device())
        set_device("cuda")

    def test_groups_partial_steps_accumulation_and_moments_are_bitwise_equal(self):
        def trajectory(multi):
            rng = np.random.default_rng(7)
            nodes = [Variable(xp.asarray(rng.normal(size=n).astype(np.float32)), trainable=True)
                     for n in (1, 255, 256, 257, 8195)]
            groups = [{"name": "head", "params": nodes[:2], "lr_mult": .3},
                      {"name": "trunk", "params": nodes[2:], "lr": .002, "weight_decay": .001}]
            opt = Adam(SimpleNamespace(nodes=nodes), learning_rate=.001, param_groups=groups, multi_tensor=multi)
            for step in range(7):
                nodes[2].trainable = step >= 2
                opt.acc_no = 1 + step % 3
                for i, node in enumerate(nodes):
                    g = rng.normal(size=node.value.shape).astype(np.float32)
                    if i != step % 5:
                        opt.acc_gradient[node] = xp.asarray(g)
                opt.update()
            return [(asnumpy(n.value), asnumpy(opt.v[n]), asnumpy(opt.s[n]), opt.parameter_steps[n]) for n in nodes]

        with patch.object(fused, "_multi_adam_kernel", wraps=fused._multi_adam_kernel) as kernel:
            expected, batched = trajectory(False), trajectory(True)
            self.assertEqual(kernel.call_count, 7)
        for left, right in zip(expected, batched):
            self.assertEqual(left[-1], right[-1])
            for a, b in zip(left[:3], right[:3]):
                self.assertEqual(a.tobytes(), b.tobytes())

    def test_overlapping_writes_and_strided_gradients_fall_back(self):
        arr = xp.arange(20, dtype=xp.float32)
        def update(p, g):
            return p, g, xp.zeros_like(p), xp.zeros_like(p), .01, 0., 1
        self.assertIsNone(fused._multi_adam_table([update(arr[:10], arr[10:]), update(arr[5:15], arr[:10])], .9, .999))
        self.assertIsNone(fused._multi_adam_table([update(arr[:10], arr[::2])], .9, .999))
