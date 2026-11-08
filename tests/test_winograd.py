import unittest
import numpy as np

import MyFlows as ms
from MyFlows.core.device import asnumpy, cuda_available, get_device, set_device
from MyFlows.ops.convolution import Conv2D_Op


class WinogradTest(unittest.TestCase):
    def compare(self, device):
        self.addCleanup(set_device, get_device())
        set_device(device)
        rng = np.random.default_rng(89)
        for h, w, pad in ((4, 4, (0, 0)), (5, 6, (1, 0)), (8, 7, (2, 1))):
            data = [rng.normal(size=s).astype(np.float32) for s in ((2, 3, h, w), (5, 3, 3, 3), (5,))]
            target = rng.normal(size=(2, 5, h + 2 * pad[0] - 2, w + 2 * pad[1] - 2)).astype(np.float32)

            def run(mode, dtype):
                x, k, b = (ms.Variable(a.astype(dtype), trainable=True) for a in data)
                op = Conv2D_Op(x, k, bias=b, padding=pad, backend=mode)
                g = ms.Graph(ms.MSELoss(op, ms.Variable(target.astype(dtype), requires_grad=False)))
                g.forward()
                g.backward()
                return [asnumpy(v).copy() for v in (op.value, x.grad, k.grad, b.grad)]

            for ref, actual in zip(run("auto", np.float64), run("winograd", np.float32)):
                np.testing.assert_allclose(actual, ref, atol=8e-6, rtol=7e-6)

    def test_cpu_padded_odd_tiles_and_both_adjoints(self):
        self.compare("cpu")

    @unittest.skipUnless(cuda_available(), "CUDA is required to check device Winograd products")
    def test_cuda_padded_odd_tiles_and_both_adjoints(self):
        self.compare("cuda")

    def test_other_strides_preserve_standard_convolution(self):
        self.addCleanup(set_device, get_device())
        set_device("cpu")
        x = ms.Variable(np.ones((1, 1, 5, 5), np.float32))
        k = ms.Variable(np.ones((1, 1, 3, 3), np.float32))
        op = Conv2D_Op(x, k, stride=2, backend="winograd")
        ms.Graph(op).forward()
        self.assertEqual(op.actual_backend, "numpy")
        np.testing.assert_array_equal(op.value, np.full((1, 1, 2, 2), 9.0))
