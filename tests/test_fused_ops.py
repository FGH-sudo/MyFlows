"""Fused BatchNorm / softmax cross-entropy / Adam / activation kernels and the default dtype."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

import MyFlows as ms
from MyFlows.core.device import cuda_available, set_device, xp
from MyFlows.core.dtype import get_default_dtype, set_default_dtype
from MyFlows.core.graph import Graph
from MyFlows.core.node import Variable
from MyFlows.ops import fused
from MyFlows.ops.activation import LeakyReLU, ReLU
from MyFlows.ops.batchnorm import BatchNorm2d_Op
from MyFlows.ops.loss import CrossEntropy, MSELoss
from MyFlows.train.opt import Adam

EPS32 = float(np.finfo(np.float32).eps)
DEVICES = ("cpu", "cuda") if cuda_available() else ("cpu",)


def _near(test, actual, expected, rel, name):
    actual = np.asarray(xp.asnumpy(actual) if hasattr(xp, "asnumpy") and not isinstance(actual, np.ndarray) else actual,
                        dtype=np.float64)
    scale = max(float(np.abs(expected).max()), 1e-30)
    error = float(np.abs(actual - expected).max())
    test.assertLessEqual(error, rel * scale, f"{name}: {error:.3e} > {rel:.1e} x {scale:.3e}")


def _host(value):
    return value if isinstance(value, np.ndarray) else value.get()


class _DeviceCase(unittest.TestCase):
    def tearDown(self):
        set_device("cpu")


class BatchNormTests(_DeviceCase):
    def _reference(self, x, dy, gamma, beta, eps, training, rm, rv):
        x, dy, gamma, beta = (a.astype(np.float64) for a in (x, dy, gamma, beta))
        axes = (0, 2, 3)
        mean = x.mean(axis=axes) if training else rm.astype(np.float64)
        var = x.var(axis=axes) if training else rv.astype(np.float64)
        inv = 1.0 / np.sqrt(var + eps)
        ch = lambda v: v.reshape(1, -1, 1, 1)
        xhat = (x - ch(mean)) * ch(inv)
        y = xhat * ch(gamma) + ch(beta)
        dbeta = dy.sum(axis=axes)
        dgamma = (dy * xhat).sum(axis=axes)
        m = x.shape[0] * x.shape[2] * x.shape[3]
        if training:
            dx = ch(gamma * inv) * (dy - ch(dbeta / m) - xhat * ch(dgamma / m))
        else:
            dx = dy * ch(gamma * inv)
        return y, dx, dgamma, dbeta, mean, var

    def _check(self, device, shape, offset=0.0, scale=1.0):
        set_device(device)
        rng = np.random.default_rng(1)
        x = (offset + scale * rng.standard_normal(shape)).astype(np.float32)
        dy = rng.standard_normal(shape).astype(np.float32)
        c = shape[1]
        gamma = rng.uniform(0.5, 1.5, c).astype(np.float32)
        beta = rng.standard_normal(c).astype(np.float32)
        for training in (True, False):
            rm = (offset + 0.1 * rng.standard_normal(c)).astype(np.float32)
            rv = rng.uniform(0.5, 2.0, c).astype(np.float32) * np.float32(scale ** 2)
            running_mean, running_var = xp.asarray(rm.copy()), xp.asarray(rv.copy())
            nodes = [Variable(xp.asarray(a)) for a in (x, gamma, beta)]
            op = BatchNorm2d_Op(*nodes, running_mean, running_var, momentum=0.1, eps=1e-5, training=training)
            op.forward(*(n.value for n in nodes))
            for n in nodes:
                n.clear_grad()
            op.grad = xp.asarray(dy)
            op.backward()
            y, dx, dgamma, dbeta, mean, var = self._reference(x, dy, gamma, beta, 1e-5, training, rm, rv)
            label = f"{device} training={training} offset={offset}"
            self.assertEqual(op.value.dtype, np.float32)
            _near(self, _host(op.value), y, 8 * EPS32, f"y {label}")
            _near(self, _host(nodes[0].grad), dx, 64 * EPS32, f"dx {label}")
            _near(self, _host(nodes[1].grad), dgamma, 4 * EPS32, f"dgamma {label}")
            _near(self, _host(nodes[2].grad), dbeta, 2 * EPS32, f"dbeta {label}")
            if training:
                _near(self, _host(running_mean), 0.9 * rm.astype(np.float64) + 0.1 * mean, 4 * EPS32, "running_mean")
                _near(self, _host(running_var), 0.9 * rv.astype(np.float64) + 0.1 * var, 4 * EPS32, "running_var")
            else:
                np.testing.assert_array_equal(_host(running_mean), rm)

    def test_matches_fp64_reference_on_all_devices(self):
        for device in DEVICES:
            for shape in ((4, 3, 5, 7), (2, 40, 9, 11), (1, 2, 1, 1)):
                with self.subTest(device=device, shape=shape):
                    self._check(device, shape)

    def test_statistics_survive_a_large_mean(self):
        # mean 1e3, std 1e-2: a float32 one-pass variance would lose every digit.
        for device in DEVICES:
            with self.subTest(device=device):
                self._check(device, (8, 4, 6, 6), offset=1000.0, scale=0.01)


class CrossEntropyTests(_DeviceCase):
    def _reference(self, logits, target):
        x = logits.astype(np.float64)
        logp = x - x.max(axis=1, keepdims=True)
        logp -= np.log(np.exp(logp).sum(axis=1, keepdims=True))
        onehot = np.eye(x.shape[1])[target] if target.ndim == 1 else target.astype(np.float64)
        return -(onehot * logp).sum(axis=1).mean(), np.exp(logp), onehot

    def test_loss_probs_and_gradient_match_fp64(self):
        rng = np.random.default_rng(2)
        for device in DEVICES:
            set_device(device)
            for scale in (1.0, 1e4):  # 1e4: log(p + 1e-10) used to clamp the loss near 23
                logits = (scale * rng.standard_normal((37, 10))).astype(np.float32)
                labels = rng.integers(0, 10, 37)
                for target in (labels, np.eye(10, dtype=np.float32)[labels]):
                    with self.subTest(device=device, scale=scale, onehot=target.ndim == 2):
                        x_node, y_node = Variable(xp.asarray(logits)), Variable(xp.asarray(target))
                        loss = CrossEntropy(x_node, y_node)
                        graph = Graph(loss)
                        graph.forward()
                        graph.backward()
                        ref_loss, ref_probs, onehot = self._reference(logits, target)
                        self.assertEqual(loss.value.dtype, np.float32)
                        self.assertLessEqual(abs(float(loss.value) - ref_loss), 2 * EPS32 * max(1.0, abs(ref_loss)))
                        _near(self, _host(loss.probs), ref_probs, 2 * EPS32, "probs")
                        _near(self, _host(x_node.grad), (ref_probs - onehot) / 37, 4 * EPS32, "grad")

    def test_gpu_invalid_label_gives_nan_instead_of_syncing(self):
        if "cuda" not in DEVICES:
            self.skipTest("CUDA is required")
        set_device("cuda")
        loss, _, _, _ = fused.softmax_cross_entropy(xp.zeros((2, 3), xp.float32), xp.asarray([0, 7]))
        self.assertTrue(np.isnan(float(loss)))
        set_device("cpu")
        with self.assertRaises(ValueError):
            fused.softmax_cross_entropy(np.zeros((2, 3), np.float32), np.asarray([0, 7]))


class AdamTests(_DeviceCase):
    def _steps(self, device, grads, lr=1e-3):
        set_device(device)
        node = Variable(xp.asarray(np.linspace(-1, 1, grads[0].size, dtype=np.float32).reshape(grads[0].shape)),
                        trainable=True)

        class G:
            nodes = (node,)

        opt = Adam(G(), learning_rate=lr)
        history = []
        for g in grads:
            opt.acc_gradient = {node: xp.asarray(g)}
            opt.acc_no = 1
            opt._update()
            history.append(_host(node.value).copy())
        return history

    def test_textbook_adam(self):
        rng = np.random.default_rng(3)
        grads = [rng.standard_normal((5, 7)).astype(np.float32) for _ in range(4)]
        history = self._steps("cpu", grads)
        theta = np.linspace(-1, 1, 35, dtype=np.float32).reshape(5, 7).astype(np.float64)
        m = np.zeros_like(theta)
        v = np.zeros_like(theta)
        for t, g in enumerate(grads, 1):
            g = g.astype(np.float64)
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g * g
            theta = theta - 1e-3 * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
            _near(self, history[t - 1], theta, 4 * EPS32, f"step {t}")
        # Standard Adam moves every coordinate by about lr on the first step.
        first = np.abs(history[0] - np.linspace(-1, 1, 35, dtype=np.float32).reshape(5, 7))
        np.testing.assert_allclose(first, 1e-3, rtol=1e-3)

    def test_cpu_and_gpu_steps_are_bitwise_identical(self):
        if "cuda" not in DEVICES:
            self.skipTest("CUDA is required")
        rng = np.random.default_rng(4)
        grads = [rng.standard_normal((64, 33)).astype(np.float32) * 10.0 ** rng.integers(-6, 2) for _ in range(5)]
        for cpu, gpu in zip(self._steps("cpu", grads), self._steps("cuda", grads)):
            np.testing.assert_array_equal(cpu, gpu)

    def test_gradient_averaging_uses_accumulation_count(self):
        set_device("cpu")
        node = Variable(np.zeros(3, np.float32), trainable=True)

        class G:
            nodes = (node,)

        averaged, summed = Adam(G(), learning_rate=0.1), Adam(G(), learning_rate=0.1)
        node.value = np.zeros(3, np.float32)
        averaged.acc_gradient, averaged.acc_no = {node: np.asarray([2.0, -4.0, 6.0], np.float32)}, 2
        averaged._update()
        first = node.value.copy()
        node.value = np.zeros(3, np.float32)
        summed.acc_gradient, summed.acc_no = {node: np.asarray([1.0, -2.0, 3.0], np.float32)}, 1
        summed._update()
        np.testing.assert_array_equal(first, node.value)


class ActivationAndNormTests(_DeviceCase):
    def test_relu_backward_keeps_dtype_and_masks_nan(self):
        for device in DEVICES:
            set_device(device)
            x = xp.asarray(np.array([-1.0, 0.0, 2.0, np.nan], np.float32))
            g = xp.asarray(np.array([1.0, 2.0, 3.0, 4.0], np.float32))
            for op_class, expected in ((ReLU, [0, 0, 3, 0]), (LeakyReLU, [0.01, 0.02, 3, 0.04])):
                node = Variable(x)
                op = op_class(node)
                op.forward(x)
                node.clear_grad()
                op.grad = g
                op.backward()
                self.assertEqual(node.grad.dtype, np.float32)
                np.testing.assert_allclose(_host(node.grad), expected, rtol=1e-6)

    def test_global_norm_accumulates_in_fp64(self):
        rng = np.random.default_rng(5)
        arrays = [rng.standard_normal(s).astype(np.float32) for s in ((100, 7), (3,), (2, 2, 2))]
        expected = np.sqrt(sum(float(np.sum(a.astype(np.float64) ** 2)) for a in arrays))
        for device in DEVICES:
            set_device(device)
            got = fused.global_norm([xp.asarray(a) for a in arrays])
            self.assertAlmostEqual(got, expected, delta=1e-12 * expected)

    def test_mse_loss_stays_on_device_in_prediction_dtype(self):
        for device in DEVICES:
            set_device(device)
            pred = Variable(xp.asarray(np.array([[1.0, 2.0]], np.float32)))
            target = Variable(xp.asarray(np.array([[0.5, 4.0]], np.float32)))
            loss = MSELoss(pred, target)
            graph = Graph(loss)
            graph.forward()
            graph.backward()
            self.assertEqual(loss.value.dtype, np.float32)
            self.assertAlmostEqual(float(loss.value), (0.25 + 4.0) / 2, places=6)
            np.testing.assert_allclose(_host(pred.grad), [[0.5, -2.0]], rtol=1e-6)


class LinearGradientTests(_DeviceCase):
    def test_bias_gradient_is_accumulated_in_fp64(self):
        from MyFlows.ops.basic import Linear

        rng = np.random.default_rng(6)
        # A large common offset makes a float32 running sum lose the small per-row differences.
        g = (1000.0 + rng.standard_normal((4097, 5))).astype(np.float32)
        for device in DEVICES:
            set_device(device)
            nodes = [Variable(xp.zeros((4097, 3), xp.float32)), Variable(xp.zeros((3, 5), xp.float32)),
                     Variable(xp.zeros((5,), xp.float32))]
            op = Linear(*nodes)
            op.forward(*(n.value for n in nodes))
            for n in nodes:
                n.clear_grad()
            op.grad = xp.asarray(g)
            op.backward()
            exact = g.astype(np.float64).sum(axis=0)
            np.testing.assert_array_equal(_host(nodes[2].grad), exact.astype(np.float32))

    def test_three_dimensional_input_and_frozen_input(self):
        from MyFlows.ops.basic import Linear

        rng = np.random.default_rng(7)
        x, w = rng.standard_normal((2, 4, 3)), rng.standard_normal((3, 5))
        g = rng.standard_normal((2, 4, 5))
        for device in DEVICES:
            set_device(device)
            for requires_grad in (True, False):
                nodes = [Variable(xp.asarray(x), requires_grad=requires_grad), Variable(xp.asarray(w), trainable=True),
                         Variable(xp.zeros((1, 5)), trainable=True)]
                op = Linear(*nodes)
                op.forward(*(n.value for n in nodes))
                for n in nodes:
                    n.clear_grad()
                op.grad = xp.asarray(g)
                op.backward()
                np.testing.assert_allclose(_host(nodes[1].grad), np.einsum("nti,nto->io", x, g), rtol=1e-12)
                np.testing.assert_allclose(_host(nodes[2].grad), g.sum(axis=(0, 1))[None], rtol=1e-12)
                expected_dx = g @ w.T if requires_grad else np.zeros_like(x)
                np.testing.assert_allclose(_host(nodes[0].grad), expected_dx, rtol=1e-12, atol=0)


class DefaultDtypeTests(unittest.TestCase):
    def tearDown(self):
        set_default_dtype(np.float32)
        set_device("cpu")

    def test_layers_default_to_float32_and_follow_the_setting(self):
        self.assertEqual(get_default_dtype(), np.float32)
        model = ms.ResNet18(in_channels=3, output_dim=2, stem="cifar", base_width=4)
        self.assertEqual({str(p.value.dtype) for p in model.params}, {"float32"})
        self.assertEqual({str(b.running_mean.dtype) for b in (model.stem_bn,)}, {"float32"})
        self.assertEqual(ms.Dense(3, 2).W.value.dtype, np.float32)
        previous = set_default_dtype(np.float64)
        self.assertEqual(previous, np.float32)
        self.assertEqual(ms.Conv2D(1, 2).kernel.value.dtype, np.float64)
        self.assertEqual(ms.BatchNorm2d(2).gamma.value.dtype, np.float64)
        self.assertEqual(ms.Dense(3, 2, dtype=np.float32).W.value.dtype, np.float32)
        with self.assertRaises(ValueError):
            set_default_dtype(np.float16)

    def test_checkpoint_restores_into_the_model_dtype(self):
        set_default_dtype(np.float64)
        old = ms.Dense(3, 2)
        set_default_dtype(np.float32)
        new = ms.Dense(3, 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "ckpt")
            ms.save_checkpoint([old], None, 0, filepath=path)
            ms.load_checkpoint([new], None, filepath=path)
        self.assertEqual(new.W.value.dtype, np.float32)
        np.testing.assert_allclose(new.W.value, old.W.value.astype(np.float32))


if __name__ == "__main__":
    unittest.main()
