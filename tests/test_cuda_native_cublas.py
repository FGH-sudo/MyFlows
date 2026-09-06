import unittest

import cupy as cp
import numpy as np

from MyFlows.core.device import set_device
from MyFlows.core.node import Variable
from MyFlows.ops.convolution import Conv2D_Op
from MyFlows.ops.cuda.kernels import conv2d_im2col_backward, conv2d_im2col_forward
from MyFlows.ops.cuda_native.native import is_available
from MyFlows.tests.cuda_fixtures import conv_fixture


@unittest.skipUnless(is_available(), "native CUDA scheduler DLL is not built")
class NativeCublasConvolutionTests(unittest.TestCase):
    def setUp(self):
        set_device("cuda")

    def tearDown(self):
        set_device("cpu")

    def test_forward_backward_matches_cupy_im2col(self):
        cases = (
            ("P0", (1, 3, 16, 16), (8, 3, 3, 3), (1, 1), (1, 1)),
            ("P1", (4, 16, 32, 32), (32, 16, 3, 3), (1, 1), (1, 1)),
            ("P2", (1, 3, 120, 160), (8, 3, 7, 7), (2, 2), (3, 3)),
        )
        rng = np.random.default_rng(0)
        for case in cases:
            _, x_shape, w_shape, stride, padding = case
            x, w, b, dy = [cp.asarray(value) for value in conv_fixture(case, 0)]
            x_node, w_node, b_node = Variable(x), Variable(w), Variable(b)
            op = Conv2D_Op(x_node, w_node, stride=stride, padding=padding,
                           bias=b_node, backend="cuda_native_cublas")
            op.forward(x, w, b)
            native_y = op.value.copy()
            op.grad = dy
            op.backward()
            native_dx, native_dw, native_db = x_node.grad.copy(), w_node.grad.copy(), b_node.grad.copy()

            ref_y, cols = conv2d_im2col_forward(x, w, b, stride=stride, padding=padding)
            ref_dx, ref_dw, ref_db = conv2d_im2col_backward(
                x, w, dy, cols, stride=stride, padding=padding)
            cp.cuda.get_current_stream().synchronize()
            np.testing.assert_allclose(cp.asnumpy(native_y), cp.asnumpy(ref_y), rtol=3e-4, atol=3e-4)
            np.testing.assert_allclose(cp.asnumpy(native_dx), cp.asnumpy(ref_dx), rtol=3e-4, atol=3e-4)
            np.testing.assert_allclose(cp.asnumpy(native_dw), cp.asnumpy(ref_dw), rtol=3e-4, atol=3e-4)
            np.testing.assert_allclose(cp.asnumpy(native_db), cp.asnumpy(ref_db), rtol=3e-4, atol=3e-4)


if __name__ == "__main__":
    unittest.main()
