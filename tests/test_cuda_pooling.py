import unittest

import numpy as np

from MyFlows.core.device import cuda_available, set_device
from MyFlows.ops.cuda.kernels import maxpool2d_forward, maxpool2d_backward
from MyFlows.tests.test_convolution import naive_maxpool_forward, naive_maxpool_backward


@unittest.skipUnless(cuda_available(), "CUDA is required")
class CudaPoolingTest(unittest.TestCase):
    def setUp(self):
        set_device("cuda")
        import cupy
        self.cp = cupy

    def tearDown(self):
        self.cp.cuda.get_current_stream().synchronize()
        set_device("cpu")

    def compare(self, x, kernel, stride, view=False):
        cp = self.cp
        expected = naive_maxpool_forward(x.astype(np.float64), kernel, stride)
        dy = np.random.default_rng(12).normal(size=expected.shape).astype(np.float32)
        expected_dx = naive_maxpool_backward(x.astype(np.float64), dy.astype(np.float64), kernel, stride)
        gx, gdy = cp.asarray(x), cp.asarray(dy)
        if view:
            gx = cp.ascontiguousarray(gx.swapaxes(2, 3)).swapaxes(2, 3)
            gdy = cp.ascontiguousarray(gdy.swapaxes(2, 3)).swapaxes(2, 3)
        y, context = maxpool2d_forward(gx, kernel_size=kernel, stride=stride)
        dx = maxpool2d_backward(gdy, context)
        self.assertEqual(dx.dtype, np.float32)
        np.testing.assert_array_equal(cp.asnumpy(y), expected)
        np.testing.assert_allclose(cp.asnumpy(dx), expected_dx, atol=1e-4, rtol=1e-3)

    def test_t5_pool_matrix(self):
        x = np.random.default_rng(0).normal(size=(2, 3, 7, 9)).astype(np.float32)
        for kernel, stride in (((2, 2), (2, 2)), ((3, 3), (2, 2)), ((3, 3), (1, 1)), ((2, 3), (1, 2))):
            with self.subTest(kernel=kernel, stride=stride):
                self.compare(x, kernel, stride, view=True)

    def test_ties_zero_negative_overlap(self):
        for value in (0, -5):
            self.compare(np.full((2, 3, 7, 9), value, np.float32), (3, 3), (1, 1))
        x = np.full((1, 1, 3, 3), -2, np.float32)
        x[0, 0, 1, 1] = -1
        y, ctx = maxpool2d_forward(self.cp.asarray(x), kernel_size=2, stride=1)
        dx = self.cp.asnumpy(maxpool2d_backward(self.cp.ones_like(y), ctx))
        expected = np.zeros_like(x)
        expected[0, 0, 1, 1] = 4
        np.testing.assert_array_equal(dx, expected)

    def test_directional_difference_without_ties(self):
        cp = self.cp
        x = np.random.default_rng(5).normal(size=(1, 2, 5, 6)).astype(np.float32)
        y, ctx = maxpool2d_forward(cp.asarray(x), kernel_size=3, stride=1)
        rng = np.random.default_rng(6)
        dy = rng.normal(size=y.shape).astype(np.float32)
        direction = rng.normal(size=x.shape)
        dx = cp.asnumpy(maxpool2d_backward(cp.asarray(dy), ctx))
        eps = 1e-6
        plus = naive_maxpool_forward(x.astype(np.float64) + eps * direction, (3, 3), (1, 1))
        minus = naive_maxpool_forward(x.astype(np.float64) - eps * direction, (3, 3), (1, 1))
        np.testing.assert_allclose(np.sum(dx * direction), np.sum((plus - minus) * dy) / (2 * eps), atol=1e-4, rtol=1e-3)

    def test_invalid_pool(self):
        cp = self.cp
        x = cp.ones((1, 1, 4, 4), cp.float32)
        for kwargs in ({"stride": 0}, {"kernel_size": 9}, {"stride": (1, 1.5)}):
            with self.assertRaises((ValueError, TypeError)):
                maxpool2d_forward(x, **kwargs)
        with self.assertRaises(TypeError):
            maxpool2d_forward(x.astype(cp.float64))
        y, ctx = maxpool2d_forward(x)
        with self.assertRaises(ValueError):
            maxpool2d_backward(y[:, :, :1], ctx)
        with self.assertRaises(TypeError):
            maxpool2d_backward(y.astype(cp.float64), ctx)
