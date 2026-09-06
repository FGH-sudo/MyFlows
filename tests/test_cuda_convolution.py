import unittest

import numpy as np

from MyFlows.core.device import cuda_available, set_device
from MyFlows.ops.cuda.kernels import (
    conv2d_forward, conv2d_backward, conv2d_im2col_forward, conv2d_im2col_backward,
    conv2d_im2col_gemm_forward, conv2d_im2col_gemm_backward,
)
from MyFlows.tests.cuda_fixtures import CONV_CASES, conv_fixture
from MyFlows.tests.test_convolution import naive_conv_forward, naive_conv_backward


@unittest.skipUnless(cuda_available(), "CUDA is required")
class CudaConvolutionTest(unittest.TestCase):
    def setUp(self):
        set_device("cuda")
        import cupy
        self.cp = cupy

    def tearDown(self):
        self.cp.cuda.get_current_stream().synchronize()
        set_device("cpu")

    def compare(self, case, bias=True, views=False, zero=False):
        cp = self.cp
        x, w, b, dy = conv_fixture(case)
        if zero:
            x.fill(0)
        stride, padding = case[3:]
        expected = naive_conv_forward(x.astype(np.float64), w.astype(np.float64), stride, padding)
        if bias:
            expected += b[None, :, None, None]
        dx, dw = naive_conv_backward(x.astype(np.float64), w.astype(np.float64), dy.astype(np.float64), stride, padding)
        arrays = [cp.asarray(a) for a in (x, w, b, dy)]
        if views:
            strided = []
            for a in arrays:
                backing = cp.zeros((*a.shape[:-1], a.shape[-1] * 2), dtype=cp.float32)
                backing[..., ::2] = a
                strided.append(backing[..., ::2])
            arrays = strided
        gx, gw, gb, gdy = arrays
        result = conv2d_forward(gx, gw, gb if bias else None, stride=stride, padding=padding)
        grads = conv2d_backward(gx, gw, gdy, stride=stride, padding=padding, need_bias_grad=bias)
        for actual, reference in zip((result, *grads[:2]), (expected, dx, dw)):
            self.assertEqual(actual.dtype, np.float32)
            np.testing.assert_allclose(cp.asnumpy(actual), reference, atol=1e-4, rtol=1e-3)
        if bias:
            np.testing.assert_allclose(cp.asnumpy(grads[2]), dy.astype(np.float64).sum(axis=(0, 2, 3)), atol=1e-4, rtol=1e-3)
        else:
            self.assertIsNone(grads[2])

    def test_t0_hand_calculation(self):
        x, w, b, _ = map(self.cp.asarray, conv_fixture(CONV_CASES[0]))
        actual = self.cp.asnumpy(conv2d_forward(x, w, b))
        np.testing.assert_array_equal(actual[0, 0], [[54, 63, 72], [99, 108, 117], [144, 153, 162]])

    def test_t0_t4_forward_and_all_gradients(self):
        for case in CONV_CASES:
            with self.subTest(case=case[0]):
                self.compare(case)

    def test_no_bias_zero_and_strided_arrays(self):
        self.compare(CONV_CASES[1], bias=False)
        self.compare(CONV_CASES[4], views=True)
        self.compare(CONV_CASES[1], zero=True)

    def test_gradients_against_independent_directional_difference(self):
        cp = self.cp
        x, w, b, dy = conv_fixture(CONV_CASES[1])
        dx, dw, db = conv2d_backward(cp.asarray(x), cp.asarray(w), cp.asarray(dy), stride=2, padding=1)
        rng = np.random.default_rng(99)
        directions = [rng.normal(size=a.shape) for a in (x, w, b)]
        x, w, b, dy = [a.astype(np.float64) for a in (x, w, b, dy)]

        def objective(sign):
            vx, vw, vb = [a + sign * 1e-5 * d for a, d in zip((x, w, b), directions)]
            return np.sum((naive_conv_forward(vx, vw, (2, 2), (1, 1)) + vb[None, :, None, None]) * dy)

        numerical = (objective(1) - objective(-1)) / 2e-5
        analytic = sum(np.sum(cp.asnumpy(g) * d) for g, d in zip((dx, dw, db), directions))
        np.testing.assert_allclose(analytic, numerical, atol=1e-4, rtol=1e-3)

    def test_rejects_invalid_input_before_launch(self):
        from unittest.mock import patch
        cp = self.cp
        x, w, b, dy = map(cp.asarray, conv_fixture(CONV_CASES[1]))
        bad_calls = [lambda: conv2d_forward(np.zeros(x.shape, np.float32), w),
                     lambda: conv2d_forward(x.astype(cp.float64), w),
                     lambda: conv2d_forward(x, w.astype(cp.float64)),
                     lambda: conv2d_forward(x, w, b.astype(cp.float64)),
                     lambda: conv2d_forward(x, w[:, :2]),
                     lambda: conv2d_forward(x[:0], w),
                     lambda: conv2d_forward(x, w, b[None]),
                     lambda: conv2d_forward(x, w, stride=(1.5, 2)),
                     lambda: conv2d_forward(x, w, padding=-1),
                     lambda: conv2d_forward(x, cp.empty((4, 3, 99, 99), cp.float32)),
                     lambda: conv2d_backward(x, w, dy[:, :, :1]),
                     lambda: conv2d_backward(x, w, dy.astype(cp.float64))]
        with patch("MyFlows.ops.cuda.kernels._launch") as launch:
            for call in bad_calls:
                with self.subTest(call=call), self.assertRaises((TypeError, ValueError)):
                    call()
            launch.assert_not_called()

    def test_current_stream(self):
        cp = self.cp
        stream = cp.cuda.Stream(non_blocking=True)
        with stream:
            x = cp.zeros((1, 1, 5, 5), cp.float32)
            x += 2
            y = conv2d_forward(x, cp.ones((1, 1, 3, 3), cp.float32))
            got = y + 1
        stream.synchronize()
        np.testing.assert_array_equal(cp.asnumpy(got), np.full((1, 1, 3, 3), 19))

    def test_im2col_cuda_path_matches_reference(self):
        cp = self.cp
        for case in CONV_CASES:
            x, w, b, dy = conv_fixture(case)
            stride, padding = case[3:]
            expected_y = naive_conv_forward(x.astype(np.float64), w.astype(np.float64), stride, padding)
            expected_y += b[None, :, None, None]
            expected_dx, expected_dw = naive_conv_backward(
                x.astype(np.float64), w.astype(np.float64), dy.astype(np.float64), stride, padding)
            y, cols = conv2d_im2col_forward(cp.asarray(x), cp.asarray(w), cp.asarray(b),
                                            stride=stride, padding=padding)
            dx, dw, db = conv2d_im2col_backward(cp.asarray(x), cp.asarray(w), cp.asarray(dy), cols,
                                                stride=stride, padding=padding)
            with self.subTest(case=case[0]):
                np.testing.assert_allclose(cp.asnumpy(y), expected_y, atol=1e-4, rtol=1e-3)
                np.testing.assert_allclose(cp.asnumpy(dx), expected_dx, atol=1e-4, rtol=1e-3)
                np.testing.assert_allclose(cp.asnumpy(dw), expected_dw, atol=1e-4, rtol=1e-3)
                np.testing.assert_allclose(cp.asnumpy(db), dy.astype(np.float64).sum(axis=(0, 2, 3)), atol=1e-4, rtol=1e-3)

    def test_im2col_cuda_path_supports_no_bias_and_views(self):
        cp = self.cp
        case = CONV_CASES[4]
        x, w, b, dy = conv_fixture(case)
        stride, padding = case[3:]
        expected_y = naive_conv_forward(x.astype(np.float64), w.astype(np.float64), stride, padding)
        expected_dx, expected_dw = naive_conv_backward(
            x.astype(np.float64), w.astype(np.float64), dy.astype(np.float64), stride, padding)

        def strided_view(array):
            backing = cp.zeros((*array.shape[:-1], array.shape[-1] * 2), dtype=cp.float32)
            backing[..., ::2] = cp.asarray(array)
            return backing[..., ::2]

        gx, gw, gdy = [strided_view(a) for a in (x, w, dy)]
        y, cols = conv2d_im2col_forward(gx, gw, None, stride=stride, padding=padding)
        dx, dw, db = conv2d_im2col_backward(
            gx, gw, gdy, cols, stride=stride, padding=padding, need_bias_grad=False)
        np.testing.assert_allclose(cp.asnumpy(y), expected_y, atol=1e-4, rtol=1e-3)
        np.testing.assert_allclose(cp.asnumpy(dx), expected_dx, atol=1e-4, rtol=1e-3)
        np.testing.assert_allclose(cp.asnumpy(dw), expected_dw, atol=1e-4, rtol=1e-3)
        self.assertIsNone(db)

    def test_cuda_gemm_path_matches_reference(self):
        cp = self.cp
        for case in CONV_CASES:
            x, w, b, dy = conv_fixture(case)
            stride, padding = case[3:]
            expected_y = naive_conv_forward(x.astype(np.float64), w.astype(np.float64), stride, padding)
            expected_y += b[None, :, None, None]
            expected_dx, expected_dw = naive_conv_backward(
                x.astype(np.float64), w.astype(np.float64), dy.astype(np.float64), stride, padding)
            y, cols = conv2d_im2col_gemm_forward(cp.asarray(x), cp.asarray(w), cp.asarray(b),
                                                 stride=stride, padding=padding)
            dx, dw, db = conv2d_im2col_gemm_backward(cp.asarray(x), cp.asarray(w), cp.asarray(dy), cols,
                                                     stride=stride, padding=padding)
            with self.subTest(case=case[0]):
                np.testing.assert_allclose(cp.asnumpy(y), expected_y, atol=2e-4, rtol=2e-3)
                np.testing.assert_allclose(cp.asnumpy(dx), expected_dx, atol=2e-4, rtol=2e-3)
                np.testing.assert_allclose(cp.asnumpy(dw), expected_dw, atol=2e-4, rtol=2e-3)
                np.testing.assert_allclose(cp.asnumpy(db), dy.astype(np.float64).sum(axis=(0, 2, 3)), atol=2e-4, rtol=2e-3)
