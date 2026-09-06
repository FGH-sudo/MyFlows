import unittest

import numpy as np

from MyFlows.core.device import cuda_available, set_device, xp, asnumpy
from MyFlows.core.graph import Graph
from MyFlows.core.node import Variable, Node
from MyFlows.examples.stage1_cnn import build, assert_fp32
from MyFlows.layers.layer import Conv2D
from MyFlows.ops.activation import ReLU, LeakyReLU
from MyFlows.ops.basic import Add
from MyFlows.ops.convolution import Conv2D_Op, MaxPool2d_Op


class SumAll(Node):
    def forward(self, value):
        self.value = xp.sum(value)

    def backward(self):
        self.parents[0].grad += self.grad


class BackendValidationTest(unittest.TestCase):
    def test_cpu_cuda_request_and_unsupported_configuration_fail(self):
        set_device("cpu")
        x = Variable(np.ones((1, 1, 4, 4), np.float32))
        w = Variable(np.ones((1, 1, 3, 3), np.float32))
        for backend in ("cuda_c", "cuda_im2col", "cuda_im2col_gemm", "cupy"):
            with self.assertRaises(ValueError):
                Graph(Conv2D_Op(x, w, backend=backend)).forward()
        for kwargs in ({"groups": 2}, {"dilation": 2}, {"stride": (1.5, 1)}):
            with self.assertRaises((ValueError, TypeError)):
                Conv2D_Op(x, w, backend="cuda_c", **kwargs)
        with self.assertRaises(ValueError):
            Conv2D_Op(x, w, backend="typo")


@unittest.skipUnless(cuda_available(), "CUDA is required")
class CudaGraphIntegrationTest(unittest.TestCase):
    def setUp(self):
        set_device("cuda")

    def tearDown(self):
        set_device("cuda")
        xp.cuda.get_current_stream().synchronize()
        set_device("cpu")

    def test_shared_weight_and_two_branches_accumulate(self):
        arrays = [np.random.default_rng(i).normal(size=s).astype(np.float32)
                  for i, s in enumerate(((2, 2, 5, 7), (3, 2, 3, 3), (3,)))]
        snapshots = []
        for backend in ("cupy", "cuda_c", "cuda_im2col", "cuda_im2col_gemm"):
            x, w, b = [Variable(xp.asarray(a)) for a in arrays]
            left = Conv2D_Op(x, w, padding=1, bias=b, backend=backend)
            right = Conv2D_Op(x, w, padding=1, bias=b, backend=backend)
            graph = Graph(SumAll(Add(left, right)))
            graph.forward()
            graph.backward()
            snapshots.append([asnumpy(a).copy() for a in (left.value, x.grad, w.grad, b.grad)])
            self.assertEqual(left.actual_backend, backend)
        for snapshot in snapshots[1:]:
            for expected, actual in zip(snapshots[0], snapshot):
                np.testing.assert_allclose(expected, actual, atol=2e-3, rtol=2e-3)

    def test_context_refresh_after_batch_and_spatial_change(self):
        x = Variable(xp.ones((1, 1, 5, 7), xp.float32))
        w = Variable(xp.ones((2, 1, 3, 3), xp.float32))
        conv = Conv2D_Op(x, w, padding=1, backend="cuda_c")
        pool = MaxPool2d_Op(conv, 3, 1, backend="cuda_c")
        graph = Graph(SumAll(pool))
        for shape in ((1, 1, 5, 7), (3, 1, 8, 6), (2, 1, 4, 5)):
            x.value = xp.ones(shape, xp.float32)
            graph.forward()
            graph.backward()
            self.assertEqual(x.grad.shape, shape)
            self.assertEqual(pool._cuda_context.input_shape, conv.value.shape)
            self.assertEqual(pool.value.shape, (shape[0], 2, shape[2] - 2, shape[3] - 2))
            self.assertTrue(bool(xp.isfinite(x.grad).all()))

    def test_backward_rejects_device_change(self):
        x = Variable(xp.ones((1, 1, 4, 4), xp.float32))
        w = Variable(xp.ones((1, 1, 3, 3), xp.float32))
        op = Conv2D_Op(x, w, backend="cuda_c")
        Graph(op).forward()
        op.grad = xp.ones_like(op.value)
        set_device("cpu")
        with self.assertRaises(ValueError):
            op.backward()

    def test_fusion_keeps_explicit_backend(self):
        for optimize, fuse_layer, activation in ((False, True, ReLU), (True, False, ReLU), (True, False, LeakyReLU)):
            x = Variable(xp.ones((1, 1, 4, 4), xp.float32))
            conv = Conv2D(1, 2, backend="cuda_c", dtype=np.float32, activation=activation,
                          fuse_activation=fuse_layer)
            graph = Graph(SumAll(conv(x)), optimize=optimize)
            graph.forward()
            graph.backward()
            conv_nodes = [n for n in graph.nodes if isinstance(n, Conv2D_Op)]
            self.assertEqual(len(conv_nodes), 1)
            self.assertEqual(conv_nodes[0].actual_backend, "cuda_c")

    def test_fp32_training_same_initial_values_gradients_updates_and_loss(self):
        models = [build(backend) for backend in ("cupy", "cuda_c")]
        for a, b in zip(models[0]["params"], models[1]["params"]):
            np.testing.assert_array_equal(asnumpy(a.value), asnumpy(b.value))
        for step in range(100):
            for model in models:
                model["optimizer"].one_step()
                assert_fp32(model)
            if step < 3:
                for a, b in zip(models[0]["params"], models[1]["params"]):
                    np.testing.assert_allclose(asnumpy(a.grad), asnumpy(b.grad), atol=1e-4, rtol=1e-3)
            for model in models:
                model["optimizer"].update()
                assert_fp32(model)
            if step < 3:
                for a, b in zip(models[0]["params"], models[1]["params"]):
                    np.testing.assert_allclose(asnumpy(a.value), asnumpy(b.value), atol=1e-4, rtol=1e-3)
        losses = []
        for model in models:
            model["graph"].forward()
            assert_fp32(model)
            accuracy = np.mean(asnumpy(model["logits"].value).argmax(1) == asnumpy(model["y"].value).ravel())
            self.assertGreaterEqual(accuracy, 0.95)
            losses.append(float(model["loss"].value))
        np.testing.assert_allclose(*losses, atol=1e-4, rtol=1e-3)
