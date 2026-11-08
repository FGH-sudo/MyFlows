import tempfile
import unittest
from pathlib import Path

import numpy as np

import MyFlows as ms
from MyFlows.core.device import asnumpy, cuda_available, get_device, set_device, xp
from MyFlows.ops import precision
from MyFlows.ops.basic import Linear
from MyFlows.ops.convolution import Conv2D_Op
from MyFlows.train.loss_scaling import DynamicLossScaler


class PrecisionTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(set_device, get_device())

    def test_scaler_growth_backoff_and_state(self):
        set_device("cpu")
        scaler = DynamicLossScaler(8.0, 2, 32.0)
        grad = np.array([16.0], np.float32)
        self.assertTrue(scaler.unscale([grad]))
        self.assertEqual(grad.item(), 2.0)
        scaler.finish()
        restored = DynamicLossScaler.from_state(scaler.state_dict())
        restored.finish()
        self.assertEqual(restored.scale, 16.0)
        self.assertFalse(restored.unscale([np.array([float("inf")], np.float32)]))
        restored.finish()
        self.assertEqual((restored.scale, restored.skipped_steps), (8.0, 1))

    @unittest.skipUnless(cuda_available(), "cuBLAS precision modes require a CUDA device")
    def test_dense_and_convolution_use_packed_storage_and_fp32_results(self):
        set_device("cuda")
        rng = np.random.default_rng(94)
        for backend in precision.BACKENDS:
            with self.subTest(backend=backend):
                x = ms.Variable(rng.normal(size=(2, 3, 4)).astype(np.float32))
                w = ms.Variable(rng.normal(size=(4, 5)).astype(np.float32), trainable=True)
                b = ms.Variable(np.zeros(5, np.float32), trainable=True)
                out = Linear(x, w, b, backend=backend)
                graph = ms.Graph(out)
                graph.forward()
                graph.backward()
                self.assertEqual(out.value.dtype, np.float32)
                self.assertEqual(out._packed_x.itemsize, 4 if backend == "cuda_tf32" else 2)
                np.testing.assert_allclose(
                    asnumpy(out.value), asnumpy(x.value) @ asnumpy(w.value), atol=0.045, rtol=0.012
                )
                expected = np.ones((6, 5), np.float32) @ asnumpy(w.value).T
                np.testing.assert_allclose(asnumpy(x.grad).reshape(6, 4), expected, atol=0.03, rtol=0.01)
                image = ms.Variable(rng.normal(size=(2, 2, 5, 7)).astype(np.float32))
                weights = ms.Variable(rng.normal(size=(3, 2, 3, 3)).astype(np.float32), trainable=True)

                def conv(mode):
                    op = Conv2D_Op(image, weights, padding=1, backend=mode)
                    g = ms.Graph(op)
                    g.forward()
                    g.backward()
                    return op, [asnumpy(v).copy() for v in (op.value, image.grad, weights.grad)]

                _, reference = conv("cuda_native_cublas")
                op, actual = conv(backend)
                self.assertEqual(op.cols[0].itemsize, 4 if backend == "cuda_tf32" else 2)
                for old, new in zip(reference, actual):
                    np.testing.assert_allclose(new, old, atol=0.065, rtol=0.015)

    @unittest.skipUnless(cuda_available(), "actual FP16 gradient overflow requires CUDA")
    def test_overflow_skips_the_update_then_recovers_and_restores_checkpoint_state(self):
        set_device("cuda")
        x = ms.Variable(np.array([[2048.0]], np.float32), requires_grad=False)
        head = ms.Dense(1, 1, backend="cuda_fp16")
        head.W.value[:] = 1
        target = ms.Variable(np.zeros((1, 1), np.float32), requires_grad=False)
        graph = ms.Graph(ms.MSELoss(head(x), target))
        opt = ms.Adam(graph, loss_scaler=DynamicLossScaler(128.0, 2))
        opt.one_step()
        opt.update()
        self.assertEqual((opt.t, opt.loss_scaler.scale, opt.loss_scaler.skipped_steps), (0, 64.0, 1))
        self.assertEqual(float(head.W.value.item()), 1.0)
        x.value = xp.ones((1, 1), np.float32)
        opt.one_step()
        opt.update()
        self.assertEqual(opt.t, 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "checkpoint"
            ms.save_checkpoint([head], opt, 1, filepath=str(path))
            other = ms.Adam(graph, loss_scaler=DynamicLossScaler())
            ms.load_checkpoint([head], other, str(path))
            self.assertEqual(other.loss_scaler.state_dict(), opt.loss_scaler.state_dict())
