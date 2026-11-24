# -*- coding: utf-8 -*-
"""Route B MyFlows components: Huber/WeightedControl losses, CosineWarmupScheduler, driving transforms, ONNX export."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import MyFlows as ms
from MyFlows.core.device import asnumpy, cuda_available, set_device
from MyFlows.core.dtype import set_default_dtype
from MyFlows.layers.mobilenet import AlexNet, MobileNetV2


class HuberAndWeightedControlLossTest(unittest.TestCase):
    def tearDown(self) -> None:
        set_device("cpu")
        set_default_dtype(np.float64)

    def test_huber_loss_values_and_finite_diff_grad(self) -> None:
        set_device("cpu")
        set_default_dtype(np.float64)
        pred_arr = np.array([[0.10, 0.80], [-0.90, 0.25], [0.00, -0.35]], dtype=np.float64)
        true_arr = np.array([[0.00, 0.20], [-0.10, 0.20], [0.15, -0.30]], dtype=np.float64)
        delta = 0.2
        y_pred = ms.Variable(pred_arr.copy(), trainable=True, name="pred")
        y_true = ms.Variable(true_arr.copy(), trainable=False, name="true")
        loss = ms.HuberLoss(y_pred, y_true, delta=delta)
        graph = ms.Graph(loss)
        graph.forward()

        diff = pred_arr - true_arr
        abs_diff = np.abs(diff)
        expected_elems = np.where(abs_diff <= delta, 0.5 * diff * diff, delta * (abs_diff - 0.5 * delta))
        self.assertAlmostEqual(float(loss.value), float(np.mean(expected_elems)), places=12)

        loss.grad = np.ones_like(loss.value)
        graph.backward()
        analytical = y_pred.grad.copy()

        eps = 1e-6
        numerical = np.zeros_like(pred_arr)
        for i in range(pred_arr.shape[0]):
            for j in range(pred_arr.shape[1]):
                p_plus = pred_arr.copy()
                p_plus[i, j] += eps
                p_minus = pred_arr.copy()
                p_minus[i, j] -= eps
                d_p = p_plus - true_arr
                d_m = p_minus - true_arr
                l_p = np.mean(np.where(np.abs(d_p) <= delta, 0.5 * d_p * d_p, delta * (np.abs(d_p) - 0.5 * delta)))
                l_m = np.mean(np.where(np.abs(d_m) <= delta, 0.5 * d_m * d_m, delta * (np.abs(d_m) - 0.5 * delta)))
                numerical[i, j] = (l_p - l_m) / (2.0 * eps)
        np.testing.assert_allclose(analytical, numerical, rtol=1e-6, atol=1e-7)

    def test_weighted_control_loss_matches_mse_at_defaults(self) -> None:
        set_device("cpu")
        set_default_dtype(np.float64)
        rng = np.random.default_rng(7)
        pred_arr = rng.normal(0.0, 0.4, size=(8, 2))
        true_arr = rng.normal(0.0, 0.4, size=(8, 2))

        p_mse = ms.Variable(pred_arr.copy(), trainable=True)
        t_mse = ms.Variable(true_arr.copy(), trainable=False)
        l_mse = ms.MSELoss(p_mse, t_mse)
        g_mse = ms.Graph(l_mse)
        g_mse.forward()
        l_mse.grad = 1.0
        g_mse.backward()

        p_w = ms.Variable(pred_arr.copy(), trainable=True)
        t_w = ms.Variable(true_arr.copy(), trainable=False)
        l_w = ms.WeightedControlLoss(p_w, t_w, base="mse", angle_weight=1.0, throttle_weight=1.0, turn_weight=0.0)
        g_w = ms.Graph(l_w)
        g_w.forward()
        l_w.grad = 1.0
        g_w.backward()

        self.assertAlmostEqual(float(l_w.value), float(l_mse.value), places=12)
        np.testing.assert_allclose(p_w.grad, p_mse.grad, rtol=1e-12, atol=1e-12)

    def test_weighted_control_loss_finite_diff_grad_mse_and_huber(self) -> None:
        set_device("cpu")
        set_default_dtype(np.float64)
        pred_arr = np.array(
            [[0.05, 0.40], [-0.75, 0.55], [0.42, 0.10], [-0.15, -0.35], [0.90, 0.50]],
            dtype=np.float64,
        )
        true_arr = np.array(
            [[0.00, 0.50], [-0.40, 0.50], [0.15, 0.25], [-0.10, 0.40], [0.35, 0.50]],
            dtype=np.float64,
        )
        for base in ("mse", "huber"):
            with self.subTest(base=base):
                y_pred = ms.Variable(pred_arr.copy(), trainable=True)
                y_true = ms.Variable(true_arr.copy(), trainable=False)
                loss = ms.WeightedControlLoss(
                    y_pred, y_true, base=base, delta=0.2, angle_weight=1.0, throttle_weight=0.1, turn_weight=3.0
                )
                graph = ms.Graph(loss)
                graph.forward()
                loss.grad = 1.0
                graph.backward()
                analytical = y_pred.grad.copy()

                def eval_fn(arr: np.ndarray) -> float:
                    vp = ms.Variable(arr.copy(), trainable=False)
                    vt = ms.Variable(true_arr.copy(), trainable=False)
                    ln = ms.WeightedControlLoss(
                        vp, vt, base=base, delta=0.2, angle_weight=1.0, throttle_weight=0.1, turn_weight=3.0
                    )
                    ms.Graph(ln).forward()
                    return float(ln.value)

                eps = 1e-6
                numerical = np.zeros_like(pred_arr)
                for i in range(pred_arr.shape[0]):
                    for j in range(pred_arr.shape[1]):
                        p_p = pred_arr.copy()
                        p_p[i, j] += eps
                        p_m = pred_arr.copy()
                        p_m[i, j] -= eps
                        numerical[i, j] = (eval_fn(p_p) - eval_fn(p_m)) / (2.0 * eps)
                np.testing.assert_allclose(analytical, numerical, rtol=1e-6, atol=1e-7)

    def test_loss_validation_errors(self) -> None:
        v = ms.Variable(np.zeros((2, 2), dtype=np.float32))
        t = ms.Variable(np.zeros((2, 2), dtype=np.float32))
        with self.assertRaises(ValueError):
            ms.HuberLoss(v, t, delta=0.0)
        with self.assertRaises(ValueError):
            ms.WeightedControlLoss(v, t, base="l1")
        with self.assertRaises(ValueError):
            ms.WeightedControlLoss(v, t, angle_weight=0.0, throttle_weight=0.0)
        with self.assertRaises(ValueError):
            ms.WeightedControlLoss(v, t, turn_weight=-0.1)
        bad_v = ms.Variable(np.zeros((2, 3), dtype=np.float32))
        bad_t = ms.Variable(np.zeros((2, 3), dtype=np.float32))
        with self.assertRaises(ValueError):
            ms.Graph(ms.WeightedControlLoss(bad_v, bad_t)).forward()

    @unittest.skipUnless(cuda_available(), "CUDA unavailable")
    def test_losses_cuda_parity(self) -> None:
        pred_arr = np.array([[0.1, 0.7], [-0.8, 0.2], [0.4, -0.3]], dtype=np.float32)
        true_arr = np.array([[0.0, 0.5], [-0.2, 0.5], [0.1, 0.1]], dtype=np.float32)
        for factory in (
            lambda p, t: ms.HuberLoss(p, t, delta=0.2),
            lambda p, t: ms.WeightedControlLoss(p, t, base="huber", delta=0.2, angle_weight=1.0, throttle_weight=0.1, turn_weight=3.0),
            lambda p, t: ms.WeightedControlLoss(p, t, base="mse", angle_weight=1.0, throttle_weight=0.2, turn_weight=2.0),
        ):
            set_device("cpu")
            set_default_dtype(np.float32)
            pc = ms.Variable(pred_arr.copy(), trainable=True)
            tc = ms.Variable(true_arr.copy(), trainable=False)
            lc = factory(pc, tc)
            gc = ms.Graph(lc)
            gc.forward()
            lc.grad = np.ones_like(lc.value)
            gc.backward()
            cpu_val, cpu_grad = float(asnumpy(lc.value)), np.asarray(asnumpy(pc.grad))

            set_device("cuda")
            set_default_dtype(np.float32)
            pg = ms.Variable(pred_arr.copy(), trainable=True)
            tg = ms.Variable(true_arr.copy(), trainable=False)
            lg = factory(pg, tg)
            gg = ms.Graph(lg)
            gg.forward()
            lg.grad = lg.value * 0 + 1.0
            gg.backward()
            gpu_val, gpu_grad = float(asnumpy(lg.value)), np.asarray(asnumpy(pg.grad))
            self.assertAlmostEqual(gpu_val, cpu_val, places=5)
            np.testing.assert_allclose(gpu_grad, cpu_grad, rtol=1e-5, atol=1e-5)


class CosineWarmupSchedulerTest(unittest.TestCase):
    def test_warmup_and_cosine_curve(self) -> None:
        sched = ms.CosineWarmupScheduler(1e-3, 12, warmup_steps=4, min_lr=1e-5)
        lrs = [sched.lr_at(s) for s in range(12)]
        self.assertAlmostEqual(lrs[0], 2.5e-4, places=12)
        self.assertAlmostEqual(lrs[3], 1e-3, places=12)
        self.assertTrue(all(lrs[i] < lrs[i + 1] for i in range(3)))
        self.assertTrue(all(lrs[i] >= lrs[i + 1] for i in range(3, 11)))
        self.assertAlmostEqual(lrs[11], 1e-5, places=12)
        self.assertAlmostEqual(sched.lr_at(20), 1e-5, places=12)

        no_warm = ms.CosineWarmupScheduler(1e-3, 5, warmup_steps=0, min_lr=1e-4)
        self.assertAlmostEqual(no_warm.lr_at(0), 1e-3, places=12)
        self.assertAlmostEqual(no_warm.lr_at(4), 1e-4, places=12)

        single = ms.CosineWarmupScheduler(1e-3, 1, warmup_steps=0, min_lr=1e-4)
        self.assertAlmostEqual(single.lr_at(0), 1e-3, places=12)

    def test_state_dict_roundtrip_and_step(self) -> None:
        sched = ms.CosineWarmupScheduler(2e-3, 10, warmup_steps=2, min_lr=5e-5)
        restored = ms.CosineWarmupScheduler.from_state_dict(sched.state_dict())
        for step in range(12):
            self.assertEqual(sched.lr_at(step), restored.lr_at(step))

        x = ms.Variable(np.ones((2, 2), dtype=np.float64), trainable=True)
        g = ms.Graph(x)
        opt = ms.Adam(g, learning_rate=2e-3)
        lr = restored.step(opt, 5)
        self.assertEqual(opt.learning_rate, lr)
        self.assertTrue(5e-5 < lr < 2e-3)


class DrivingTransformsTest(unittest.TestCase):
    def test_horizontal_shift_steering_sign_and_throttle_preserved(self) -> None:
        img = np.zeros((32, 64, 3), dtype=np.float32)
        img[:, 30:34, :] = 1.0  # center white stripe

        t_right = ms.RandomHorizontalShift(max_shift_frac=0.1, steer_per_frac=1.2, max_vertical_frac=0.0, seed=0)
        t_right._rng.uniform = lambda a, b: 0.05 if math.isclose(b, 0.1) else 0.0
        out_r, lab_r = t_right(img, np.array([0.1, 0.5], dtype=np.float32))
        self.assertAlmostEqual(float(lab_r[0]), 0.16, places=5)
        self.assertAlmostEqual(float(lab_r[1]), 0.5, places=6)
        self.assertGreater(float(out_r[:, 35:, :].sum()), float(img[:, 35:, :].sum()))

        t_left = ms.RandomHorizontalShift(max_shift_frac=0.1, steer_per_frac=1.2, max_vertical_frac=0.0, seed=0)
        t_left._rng.uniform = lambda a, b: -0.05 if math.isclose(b, 0.1) else 0.0
        _out_l, lab_l = t_left(img, np.array([-0.98, 0.4], dtype=np.float32))
        self.assertAlmostEqual(float(lab_l[0]), -1.0, places=6)  # clipped at -1.0
        self.assertAlmostEqual(float(lab_l[1]), 0.4, places=6)

    def test_random_shadow_darkens_without_changing_label(self) -> None:
        img = np.full((32, 64, 3), 0.8, dtype=np.float32)
        label = np.array([0.25, 0.5], dtype=np.float32)
        shadow = ms.RandomShadow(prob=1.0, min_alpha=0.5, max_alpha=0.5, seed=3)
        out, out_label = shadow(img, label.copy())
        np.testing.assert_allclose(out_label, label)
        self.assertTrue((out >= 0.0).all() and (out <= 1.0).all())
        self.assertLess(float(out.min()), 0.45)
        self.assertAlmostEqual(float(out.max()), 0.8, places=5)

    def test_driving_transforms_rng_replay(self) -> None:
        img = np.random.default_rng(1).random((32, 64, 3), dtype=np.float32)
        lab = np.array([0.2, 0.5], dtype=np.float32)
        tr = ms.driving_train_transforms(seed=17)
        for _ in range(3):
            tr(img, lab.copy())
        saved = [t._rng.getstate() for t in tr.transforms]
        expected = [tr(img, lab.copy()) for _ in range(4)]
        for t, state in zip(tr.transforms, saved):
            t._rng.setstate(state)
        replayed = [tr(img, lab.copy()) for _ in range(4)]
        for (e_img, e_lab), (r_img, r_lab) in zip(expected, replayed):
            np.testing.assert_array_equal(e_img, r_img)
            np.testing.assert_array_equal(e_lab, r_lab)


class OnnxMetadataAndMobileNetExportTest(unittest.TestCase):
    def tearDown(self) -> None:
        set_device("cpu")
        set_default_dtype(np.float64)

    def test_export_onnx_writes_metadata_props(self) -> None:
        import onnxruntime as ort

        set_device("cpu")
        set_default_dtype(np.float32)
        x = ms.Variable(np.array([[1.0, 2.0, 3.0]], dtype=np.float32), name="x", requires_grad=False)
        dense = ms.Dense(3, 2, name="dense")
        graph = ms.Graph(dense(x))
        graph.forward()
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "tiny.onnx"
            ms.export_onnx(graph, str(onnx_path), input_nodes=[x], output_names=["out"],
                           metadata={"myflows.roi_crop": "60,16"})
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            self.assertEqual(sess.get_modelmeta().custom_metadata_map["myflows.roi_crop"], "60,16")

    def test_mobilenet_v2_and_alexnet_onnx_export_parity(self) -> None:
        import onnxruntime as ort

        rng = np.random.default_rng(123)
        sample = rng.uniform(0.0, 1.0, size=(1, 3, 64, 64)).astype(np.float32)
        for cls, fname in ((MobileNetV2, "mobilenet_v2.onnx"), (AlexNet, "alexnet.onnx")):
            with self.subTest(arch=cls.__name__), tempfile.TemporaryDirectory() as tmp:
                set_device("cpu")
                set_default_dtype(np.float32)
                x = ms.Variable(sample.copy(), name="X", requires_grad=False)
                model = cls(output_dim=2, dropout=0.0, input_norm="imagenet")
                out = model(x)
                model.eval()
                graph = ms.Graph(out, optimize=True)
                graph.forward()
                expected = np.asarray(asnumpy(graph.target_node.value), dtype=np.float32)

                onnx_path = Path(tmp) / fname
                ms.export_onnx(
                    graph,
                    str(onnx_path),
                    input_nodes=[x],
                    output_names=["prediction"],
                    metadata={"myflows.roi_crop": "12,4"},
                )
                sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
                self.assertEqual(sess.get_modelmeta().custom_metadata_map["myflows.roi_crop"], "12,4")
                ort_out = np.asarray(sess.run(None, {sess.get_inputs()[0].name: sample})[0], dtype=np.float32)
                np.testing.assert_allclose(ort_out, expected, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
