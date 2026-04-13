import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from MyFlows.core.node import Variable
from MyFlows.ops.basic import Linear
from MyFlows.ops.convolution import (
    Conv2D_Op,
    Conv2D_ReLU_Op,
    Conv2D_LeakyReLU_Op,
    ConvTranspose2D_Op,
    ConvTranspose2D_ReLU_Op,
    ConvTranspose2D_LeakyReLU_Op,
)


def _numerical_grad(param: np.ndarray, f, eps=1e-5) -> np.ndarray:
    grad = np.zeros_like(param, dtype=np.float64)
    it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = float(param[idx])

        param[idx] = old + eps
        fp = float(f())
        param[idx] = old - eps
        fm = float(f())
        param[idx] = old

        grad[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


def _assert_allclose(testcase, got, expected, *, rtol=1e-4, atol=1e-5, name="grad"):
    testcase.assertTrue(
        np.allclose(got, expected, rtol=rtol, atol=atol),
        msg=f"{name} mismatch: max_abs={np.max(np.abs(got - expected))} max_rel={np.max(np.abs(got - expected) / (np.maximum(1e-12, np.abs(expected))))}",
    )


class AutoDiffGradCheckTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_linear_gradcheck_w_and_b(self):
        x = self.rng.normal(size=(3, 4)).astype(np.float64)
        w = self.rng.normal(size=(4, 5)).astype(np.float64)
        b = self.rng.normal(size=(1, 5)).astype(np.float64)

        x_node = Variable(x.copy())
        w_node = Variable(w.copy(), trainable=True)
        b_node = Variable(b.copy(), trainable=True)
        op = Linear(x_node, w_node, b_node)

        def f():
            op.forward(x_node.value, w_node.value, b_node.value)
            return np.sum(op.value)

        # analytic
        op.forward(x_node.value, w_node.value, b_node.value)
        x_node.clear_grad()
        w_node.clear_grad()
        b_node.clear_grad()
        op.grad = np.ones_like(op.value)
        op.backward()

        # numeric
        w_num = _numerical_grad(w_node.value, f)
        b_num = _numerical_grad(b_node.value, f)

        _assert_allclose(self, w_node.grad, w_num, name="Linear dW")
        _assert_allclose(self, b_node.grad, b_num, name="Linear db")

    def _conv2d_gradcheck_kernel(self, OpCls, *, alpha=None):
        x = self.rng.normal(size=(1, 2, 4, 4)).astype(np.float64)
        k = self.rng.normal(size=(3, 2, 3, 3)).astype(np.float64)
        b = self.rng.normal(size=(3,)).astype(np.float64)

        x_node = Variable(x.copy())
        k_node = Variable(k.copy(), trainable=True)
        b_node = Variable(b.copy(), trainable=True)

        if alpha is None:
            op = OpCls(x_node, k_node, stride=1, padding=1, bias=b_node)
        else:
            op = OpCls(x_node, k_node, stride=1, padding=1, bias=b_node, alpha=alpha)

        def f():
            op.forward(x_node.value, k_node.value, b_node.value)
            return np.sum(op.value)

        # analytic
        op.forward(x_node.value, k_node.value, b_node.value)
        x_node.clear_grad()
        k_node.clear_grad()
        b_node.clear_grad()
        op.grad = np.ones_like(op.value)
        op.backward()

        k_num = _numerical_grad(k_node.value, f)
        b_num = _numerical_grad(b_node.value, f)

        _assert_allclose(self, k_node.grad, k_num, name=f"{OpCls.__name__} dK")
        _assert_allclose(self, b_node.grad, b_num, name=f"{OpCls.__name__} db")

    def test_conv2d_gradcheck(self):
        self._conv2d_gradcheck_kernel(Conv2D_Op)

    def test_conv2d_relu_gradcheck(self):
        self._conv2d_gradcheck_kernel(Conv2D_ReLU_Op)

    def test_conv2d_leaky_relu_gradcheck(self):
        self._conv2d_gradcheck_kernel(Conv2D_LeakyReLU_Op, alpha=0.2)

    def _deconv_gradcheck_kernel(self, OpCls, *, alpha=None):
        x = self.rng.normal(size=(1, 2, 3, 3)).astype(np.float64)
        k = self.rng.normal(size=(2, 3, 3, 3)).astype(np.float64)
        b = self.rng.normal(size=(3,)).astype(np.float64)

        x_node = Variable(x.copy())
        k_node = Variable(k.copy(), trainable=True)
        b_node = Variable(b.copy(), trainable=True)

        if alpha is None:
            op = OpCls(x_node, k_node, stride=2, padding=1, output_padding=1, bias=b_node)
        else:
            op = OpCls(x_node, k_node, stride=2, padding=1, output_padding=1, bias=b_node, alpha=alpha)

        def f():
            op.forward(x_node.value, k_node.value, b_node.value)
            return np.sum(op.value)

        # analytic
        op.forward(x_node.value, k_node.value, b_node.value)
        x_node.clear_grad()
        k_node.clear_grad()
        b_node.clear_grad()
        op.grad = np.ones_like(op.value)
        op.backward()

        k_num = _numerical_grad(k_node.value, f)
        b_num = _numerical_grad(b_node.value, f)

        _assert_allclose(self, k_node.grad, k_num, name=f"{OpCls.__name__} dK")
        _assert_allclose(self, b_node.grad, b_num, name=f"{OpCls.__name__} db")

    def test_convtranspose_gradcheck(self):
        self._deconv_gradcheck_kernel(ConvTranspose2D_Op)

    def test_convtranspose_relu_gradcheck(self):
        self._deconv_gradcheck_kernel(ConvTranspose2D_ReLU_Op)

    def test_convtranspose_leaky_relu_gradcheck(self):
        self._deconv_gradcheck_kernel(ConvTranspose2D_LeakyReLU_Op, alpha=0.15)


if __name__ == "__main__":
    unittest.main()

