"""Frozen paths, independently requested parent gradients and reanalysis of a reused graph."""

import unittest
from unittest.mock import patch

import numpy as np

import MyFlows as ms
from MyFlows.core.device import asnumpy, cuda_available, get_device, set_device
from MyFlows.ops.basic import Linear
from MyFlows.ops.batchnorm import BatchNorm2d_Op
from MyFlows.ops.convolution import Conv2D_Op, Conv2D_ReLU_Op


class GradientPruningTest(unittest.TestCase):
    def setUp(self):
        self.previous_device = get_device()
        set_device("cpu")

    def tearDown(self):
        set_device(self.previous_device)

    def test_freeze_then_progressively_unfreeze_with_a_shared_branch(self):
        x = ms.Variable(np.array([[1., 2.]], np.float32), requires_grad=False)
        w = ms.Variable(np.eye(2, dtype=np.float32), trainable=True)
        b = ms.Variable(np.zeros(2, np.float32), trainable=True)
        feature = Linear(x, w, b)
        head = ms.Variable(np.array([[2.], [3.]], np.float32), trainable=True)
        target = ms.MatMul(ms.Add(feature, feature), head)
        graph = ms.Graph(target, optimize=True)
        graph.forward()
        graph.backward()
        expected = w.grad.copy()
        w.trainable = b.trainable = False
        with patch.object(feature, "backward", wraps=feature.backward) as backward:
            graph.forward()
            graph.backward()
            backward.assert_not_called()
            self.assertIsNone(w.grad)
            self.assertIsNone(b.grad)
            self.assertIsNone(feature.grad)
            np.testing.assert_array_equal(head.grad, [[2.], [4.]])
            w.trainable = True
            graph.forward()
            graph.backward()
            backward.assert_called_once()
            np.testing.assert_array_equal(w.grad, expected)
            self.assertIsNone(b.grad)

    def test_retaining_an_intermediate_stops_at_the_frozen_feature(self):
        x = ms.Variable(np.ones((1, 2), np.float32), requires_grad=False)
        w = ms.Variable(np.eye(2, dtype=np.float32), trainable=True)
        w.trainable = False
        feature = ms.MatMul(x, w)
        target = ms.Add(feature, feature)
        regular = ms.Graph(target)
        cam = ms.Graph(target)
        cam.retain_grad(feature)
        with patch.object(feature, "backward", wraps=feature.backward) as backward:
            cam.forward()
            cam.backward()
            np.testing.assert_array_equal(feature.grad, [[2., 2.]])
            backward.assert_not_called()
            regular.forward()
            regular.backward()
            self.assertIsNone(feature.grad)
            self.assertIsNone(w.grad)
            cam.forward()
            cam.backward()
            np.testing.assert_array_equal(feature.grad, [[2., 2.]])

    def _conv_masks(self, backend, groups=1):
        rng = np.random.default_rng(8)
        x = ms.Variable(rng.normal(size=(2, 3, 6, 5)).astype(np.float32), requires_grad=True)
        w = ms.Variable(rng.normal(size=(3, 3 // groups, 3, 3)).astype(np.float32), trainable=True)
        b = ms.Variable(rng.normal(size=3).astype(np.float32), trainable=True)
        op = Conv2D_ReLU_Op(x, w, bias=b, backend=backend, padding=1, groups=groups)
        graph = ms.Graph(op)
        graph.forward()
        graph.backward()
        expected = [asnumpy(p.grad).copy() for p in (x, w, b)]
        for mask in ((True, False, False), (False, True, False), (False, False, True),
                     (True, False, True), (False, False, False)):
            with self.subTest(backend=backend, groups=groups, mask=mask):
                x.requires_grad, w.trainable, b.trainable = mask
                graph.forward()
                graph.backward()
                for need, node, reference in zip(mask, (x, w, b), expected):
                    if need:
                        np.testing.assert_allclose(asnumpy(node.grad), reference, rtol=2e-6, atol=2e-6)
                    else:
                        self.assertIsNone(node.grad)
                if not any(mask):
                    self.assertIsNone(getattr(op, "_cuda_im2col_cols", None))
                    self.assertFalse(getattr(op, "cols", []))
                    self.assertIsNone(getattr(op, "_depthwise_input", None))

    def test_numpy_conv_and_depthwise_masks(self):
        self._conv_masks("numpy")
        self._conv_masks("numpy", groups=3)

    @unittest.skipUnless(cuda_available(), "requires CUDA and CuPy")
    def test_cuda_conv_masks_on_all_engines(self):
        set_device("cuda")
        for backend in ("cupy", "cuda_native_register", "cuda_native_cublas"):
            self._conv_masks(backend)
        self._conv_masks("cupy", groups=3)

    def test_batchnorm_frozen_affine_preserves_input_gradient(self):
        from MyFlows.core.device import xp

        x = ms.Variable(np.arange(24, dtype=np.float32).reshape(2, 3, 2, 2))
        gamma = ms.Variable(np.array([1., 2., 3.], np.float32), trainable=True)
        beta = ms.Variable(np.ones(3, np.float32), trainable=True)
        bn = BatchNorm2d_Op(x, gamma, beta, xp.zeros(3, np.float32), xp.ones(3, np.float32), training=False)
        graph = ms.Graph(bn)
        graph.forward()
        graph.backward()
        expected = x.grad.copy()
        gamma.trainable = beta.trainable = False
        graph.forward()
        graph.backward()
        np.testing.assert_array_equal(x.grad, expected)
        self.assertIsNone(gamma.grad)
        self.assertIsNone(beta.grad)
