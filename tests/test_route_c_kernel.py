# -*- coding: utf-8 -*-
"""Route C Batch 1 tests: Module/Parameter/Buffer, dynamic-batch ONNX, capability guard, CUDA depthwise."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from MyFlows import (
    Buffer,
    Conv2D,
    Dense,
    Graph,
    Module,
    Parameter,
    ResNet18,
    Variable,
    export_onnx,
    load_checkpoint,
    save_checkpoint,
)
from MyFlows.utils.checkpoint import named_state
from MyFlows.core.device import asnumpy, cuda_available, get_device, set_device, xp
from MyFlows.core.dtype import get_default_dtype, set_default_dtype
from MyFlows.layers.mobilenet import MobileNetV2
from MyFlows.layers.resnet import BatchNorm2d, GlobalAvgPool2d
from MyFlows.ops.activation import ReLU
from MyFlows.ops.basic import Add
from MyFlows.ops.convolution import Conv2D_Op, depthwise_conv2d_backward, depthwise_conv2d_forward
from MyFlows.ops.cuda_native import autotune, native
from MyFlows.ops.fused import depthwise_conv2d_backward_cuda, depthwise_conv2d_forward_cuda
from MyFlows.ops.loss import MSELoss
from MyFlows.utils.initializers import make_initializer


def _naive_depthwise_forward(x: np.ndarray, kernel: np.ndarray, stride, padding, dilation) -> np.ndarray:
    x64 = x.astype(np.float64)
    k64 = kernel.astype(np.float64)
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    n, c, h, w = x64.shape
    co, k_in, kh, kw = k64.shape
    assert co == c and k_in == 1
    eff_h = (kh - 1) * dh + 1
    eff_w = (kw - 1) * dw + 1
    x_pad = np.pad(x64, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant")
    oh = (h + 2 * ph - eff_h) // sh + 1
    ow = (w + 2 * pw - eff_w) // sw + 1
    out = np.zeros((n, c, oh, ow), dtype=np.float64)
    for ch in range(c):
        for r in range(oh):
            hs = r * sh
            for col in range(ow):
                ws = col * sw
                win = x_pad[:, ch, hs : hs + eff_h : dh, ws : ws + eff_w : dw]
                out[:, ch, r, col] = np.sum(win * k64[ch, 0], axis=(1, 2))
    return out


def _naive_depthwise_backward(
    x: np.ndarray, kernel: np.ndarray, grad_y: np.ndarray, stride, padding, dilation
) -> tuple[np.ndarray, np.ndarray]:
    x64 = x.astype(np.float64)
    k64 = kernel.astype(np.float64)
    dy64 = grad_y.astype(np.float64)
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    _, c, h, w = x64.shape
    _, _, kh, kw = k64.shape
    oh, ow = dy64.shape[2:]
    eff_h = (kh - 1) * dh + 1
    eff_w = (kw - 1) * dw + 1
    x_pad = np.pad(x64, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant")
    dx_pad = np.zeros_like(x_pad)
    dk = np.zeros_like(k64)
    for ch in range(c):
        for r in range(oh):
            hs = r * sh
            for col in range(ow):
                ws = col * sw
                dy_slice = dy64[:, ch, r, col][:, None, None]
                dx_pad[:, ch, hs : hs + eff_h : dh, ws : ws + eff_w : dw] += dy_slice * k64[ch, 0]
                win = x_pad[:, ch, hs : hs + eff_h : dh, ws : ws + eff_w : dw]
                dk[ch, 0] += np.sum(win * dy_slice, axis=0)
    dx = dx_pad[:, :, ph : ph + h, pw : pw + w]
    return dx, dk


class _SubBlock(Module):
    def __init__(self):
        super().__init__(name="sub")
        rng = np.random.default_rng(3)
        self.conv = Conv2D(4, 4, kernel_size=3, padding=1, backend="auto", name="conv")
        self.bn = BatchNorm2d(4, name="bn")
        self.weight = Parameter(rng.normal(size=(4,)).astype(np.float32), trainable=True, name="w")
        self.frozen = Parameter(np.full((1, 4, 1, 1), 0.75, dtype=np.float32), trainable=False, name="frozen")
        self.register_buffer("scale", np.array([1.0, 1.5, 2.0, 2.5], dtype=np.float32))
        self.register_buffer("scratch", np.zeros((4,), dtype=np.float32), persistent=False)

    def forward(self, x):
        out = ReLU(self.bn(self.conv(x)))
        return Add(out, self.frozen)


class _RootModule(Module):
    def __init__(self):
        super().__init__(name="root")
        shared = _SubBlock()
        self.shared_a = shared
        self.shared_b = shared
        shared.parent_ref = self  # intentional reference cycle to test traversal cycle safety
        self.pool = GlobalAvgPool2d(name="gap")
        self.head = Dense(4, 2, activation=None, name="head")

    def forward(self, x):
        out = self.shared_a(x)
        pooled = self.pool(out)
        return self.head(pooled)


class ModuleParameterBufferTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(set_device, get_device())
        self.addCleanup(set_default_dtype, get_default_dtype())
        set_device("cpu")
        set_default_dtype(np.float32)

    def test_auto_registration_cycles_eval_state_dict_and_checkpoint_roundtrip(self):
        root = _RootModule()

        self.assertTrue(root.shared_a.frozen._is_parameter)
        self.assertFalse(root.shared_a.frozen.trainable)

        named_p = dict(root.named_parameters())
        self.assertIn("shared_a.weight", named_p)
        self.assertIn("shared_a.frozen", named_p)
        self.assertIn("shared_a.conv.conv_kernel", named_p)
        self.assertIn("head.head_W", named_p)
        # shared_b aliases shared_a; deduplication must avoid duplicate names/objects
        self.assertFalse(any(k.startswith("shared_b.") for k in named_p))
        self.assertEqual(len(list(root.parameters())), len(named_p))
        self.assertGreaterEqual(len(root.params), len(named_p))
        self.assertGreaterEqual(len(root.sub_layers), 3)

        named_b = dict(root.named_buffers())
        self.assertIn("shared_a.scale", named_b)
        self.assertIn("shared_a.scratch", named_b)
        self.assertIn("shared_a.bn.running_mean", named_b)
        self.assertIn("shared_a.bn.running_var", named_b)

        modules = dict(root.named_modules())
        self.assertIn("", modules)
        self.assertIn("shared_a", modules)
        self.assertIn("shared_a.bn", modules)
        self.assertEqual(len(list(root.children())), 3)

        root.eval()
        self.assertFalse(root.training)
        self.assertFalse(root.shared_a.training)
        self.assertFalse(root.shared_a.bn.training)
        root.train()
        self.assertTrue(root.training)
        self.assertTrue(root.shared_a.bn.training)

        sd = root.state_dict()
        self.assertIn("shared_a.scale", sd)
        self.assertIn("shared_a.frozen", sd)
        self.assertIn("shared_a.bn.running_mean", sd)
        self.assertNotIn("shared_a.scratch", sd)

        orig_scale = sd["shared_a.scale"].copy()
        orig_frozen = sd["shared_a.frozen"].copy()
        orig_rm = root.shared_a.bn.running_mean.copy()
        scale_buf = root.shared_a._buffers["scale"]
        scale_arr_id = id(scale_buf.data)
        rm_arr_id = id(root.shared_a.bn.running_mean)

        root.shared_a.frozen.value[...] = 0.0
        scale_buf.data[...] = 9.0
        root.shared_a.bn.running_mean[...] = -3.0

        root.load_state_dict(sd, strict=True)
        np.testing.assert_array_equal(root.shared_a.frozen.value, orig_frozen)
        np.testing.assert_array_equal(scale_buf.data, orig_scale)
        np.testing.assert_array_equal(root.shared_a.bn.running_mean, orig_rm)
        self.assertEqual(id(scale_buf.data), scale_arr_id)
        self.assertEqual(id(root.shared_a.bn.running_mean), rm_arr_id)

        with self.assertRaises(ValueError):
            root.load_state_dict({k: v for k, v in sd.items() if k != "shared_a.scale"}, strict=True)
        with self.assertRaises(ValueError):
            root.load_state_dict({**sd, "extra.unknown": np.zeros((1,), dtype=np.float32)}, strict=True)
        bad_shape = dict(sd)
        bad_shape["shared_a.scale"] = np.zeros((2,), dtype=np.float32)
        with self.assertRaises(ValueError):
            root.load_state_dict(bad_shape, strict=True)

        p_names, b_names = named_state([root])
        self.assertTrue(any(k.endswith("_scale") for k in b_names))
        self.assertFalse(any(k.endswith("_scratch") for k in b_names))
        self.assertTrue(any(k.endswith("_frozen") for k in p_names))

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "module_ckpt"
            save_checkpoint([root], epoch=4, loss=0.125, filepath=ckpt)
            root.shared_a.frozen.value[...] = 5.0
            scale_buf.data[...] = -1.0
            root.shared_a.bn.running_mean[...] = 7.0
            epoch, loss = load_checkpoint([root], filepath=ckpt)
            self.assertEqual((epoch, loss), (4, 0.125))
            np.testing.assert_array_equal(root.shared_a.frozen.value, orig_frozen)
            np.testing.assert_array_equal(scale_buf.data, orig_scale)
            np.testing.assert_array_equal(root.shared_a.bn.running_mean, orig_rm)

    def test_frozen_parameter_exports_as_initializer_not_graph_input(self):
        if not (importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime")):
            self.skipTest("onnx and onnxruntime required")
        import onnxruntime as ort

        net = _RootModule()
        net.eval()
        rng = np.random.default_rng(17)
        x = Variable(rng.normal(size=(1, 4, 8, 8)).astype(np.float32), name="input", requires_grad=False)
        graph = Graph(net(x), optimize=False)
        graph.forward()

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "root_dyn.onnx"
            export_onnx(graph, onnx_path, input_nodes=[x], output_names=["out"], dynamic_batch=True)
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            inputs = sess.get_inputs()
            self.assertEqual(len(inputs), 1)
            self.assertEqual(inputs[0].shape[0], "batch")
            in_name = inputs[0].name
            for batch in (1, 3, 7):
                sample = rng.normal(size=(batch, 4, 8, 8)).astype(np.float32)
                x.value = sample
                graph.forward()
                (got,) = sess.run(None, {in_name: sample})
                np.testing.assert_allclose(got, graph.target_node.value, rtol=1e-4, atol=1e-4)


class DynamicBatchOnnxExportTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(set_device, get_device())
        self.addCleanup(set_default_dtype, get_default_dtype())
        set_device("cpu")
        set_default_dtype(np.float32)
        if not (importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime")):
            self.skipTest("onnx and onnxruntime required")

    def _assert_dynamic_parity(self, model, shape_hw: tuple[int, int], *, metadata=None, batches=(1, 3, 7)):
        import onnx
        import onnxruntime as ort

        model.eval()
        h, w = shape_hw
        rng = np.random.default_rng(29)
        x = Variable(rng.uniform(0.0, 1.0, size=(1, 3, h, w)).astype(np.float32), name="image", requires_grad=False)
        graph = Graph(model(x), optimize=True)
        graph.forward()

        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "dyn.onnx"
            export_onnx(
                graph,
                onnx_path,
                input_nodes=[x],
                output_names=["control"],
                metadata=metadata,
                dynamic_batch=True,
            )
            proto = onnx.load(str(onnx_path))
            if metadata:
                props = {entry.key: entry.value for entry in proto.metadata_props}
                for k, v in metadata.items():
                    self.assertEqual(props.get(k), v)

            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            inp = sess.get_inputs()[0]
            out_meta = sess.get_outputs()[0]
            self.assertEqual(inp.shape[0], "batch")
            self.assertEqual(out_meta.shape[0], "batch")

            for batch in batches:
                sample = rng.uniform(0.0, 1.0, size=(batch, 3, h, w)).astype(np.float32)
                x.value = sample
                graph.forward()
                (got,) = sess.run(None, {inp.name: sample})
                self.assertEqual(got.shape, (batch, 2))
                np.testing.assert_allclose(got, graph.target_node.value, rtol=1e-4, atol=1e-4)

    def test_resnet18_dynamic_batch_matches_myflows_and_preserves_metadata(self):
        model = ResNet18(
            in_channels=3,
            output_dim=2,
            stem="cifar",
            base_width=4,
            input_norm="imagenet",
            initializer=make_initializer("kaiming_normal", seed=5),
        )
        self._assert_dynamic_parity(model, (32, 48), metadata={"myflows.roi_crop": "16,8"})

    def test_mobilenet_v2_dynamic_batch_matches_myflows(self):
        model = MobileNetV2(
            output_dim=2,
            dropout=0.0,
            input_norm="imagenet",
            initializer=make_initializer("kaiming_normal", seed=7),
        )
        self._assert_dynamic_parity(model, (48, 64), metadata={"myflows.roi_crop": "12,4"})


class DeviceCapabilitiesAndAutotuneGuardTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(native.device_capabilities.cache_clear)
        native.device_capabilities.cache_clear()

    def test_device_capabilities_reports_expected_flags_for_mocked_architectures(self):
        def _fake_cupy(cc_major, cc_minor, name):
            attrs = {
                "ComputeCapabilityMajor": cc_major,
                "ComputeCapabilityMinor": cc_minor,
                "MultiProcessorCount": 40,
                "MaxSharedMemoryPerBlock": 49152,
            }

            class _FakeDevice:
                def __init__(self, dev_id):
                    self.id = dev_id
                    self.attributes = attrs
                    self.pci_bus_id = "0000:01:00.0"

            return SimpleNamespace(
                cuda=SimpleNamespace(
                    Device=_FakeDevice,
                    runtime=SimpleNamespace(
                        getDevice=lambda: 0,
                        getDeviceProperties=lambda dev: {
                            "name": name,
                            "major": cc_major,
                            "minor": cc_minor,
                            "multiProcessorCount": 40,
                            "sharedMemPerBlock": 49152,
                        },
                    ),
                )
            )

        with patch.object(native, "_lazy_cupy", return_value=_fake_cupy(7, 5, b"Turing-T4")):
            native.device_capabilities.cache_clear()
            cap = native.device_capabilities(0)
            self.assertEqual(cap["cc"], (7, 5))
            self.assertFalse(cap["supports_cp_async"])
            self.assertEqual(cap["name"], "Turing-T4")

        with patch.object(native, "_lazy_cupy", return_value=_fake_cupy(9, 0, "Hopper-H100")):
            native.device_capabilities.cache_clear()
            cap = native.device_capabilities(0)
            self.assertEqual(cap["cc"], (9, 0))
            self.assertTrue(cap["supports_cp_async"])

    @unittest.skipUnless(cuda_available(), "CUDA required for real device capabilities and autotune")
    def test_real_device_capabilities_and_turing_cc_skips_cp_async_tiles(self):
        self.addCleanup(set_device, get_device())
        set_device("cuda")
        cap = native.device_capabilities()
        self.assertEqual(
            set(cap.keys()),
            {"device", "name", "cc", "sms", "shared_mem_per_block", "supports_cp_async"},
        )
        self.assertEqual(len(cap["cc"]), 2)
        self.assertEqual(cap["supports_cp_async"], tuple(cap["cc"]) >= (8, 0))

        cp_async_tiles = {tile for tile in native.tile_catalog() if "/s" in tile}
        self.assertTrue(cp_async_tiles)
        result = autotune._calibrate("NN", (32, 32, 32), {"cc": (7, 5)})
        by_tile = {row["tile"]: row for row in result["candidates"]}
        for tile in cp_async_tiles:
            self.assertEqual(by_tile[tile]["skip"], "cp.async requires compute capability >= 8.0")
        self.assertNotIn("/s", result["selected"]["tile"])


class DepthwiseConv2dCudaKernelTest(unittest.TestCase):
    CASES = (
        ((2, 4, 7, 9), (4, 1, 3, 3), (1, 1), (1, 1), (1, 1)),
        ((2, 6, 8, 7), (6, 1, 5, 3), (2, 2), (2, 1), (1, 1)),
        ((1, 3, 9, 10), (3, 1, 3, 3), (1, 2), (0, 0), (2, 2)),
        ((2, 5, 6, 6), (5, 1, 3, 3), (1, 1), (2, 2), (2, 2)),
    )

    def setUp(self):
        if not cuda_available():
            self.skipTest("CUDA is required")
        self.addCleanup(set_device, get_device())
        self.addCleanup(set_default_dtype, get_default_dtype())
        set_device("cuda")
        set_default_dtype(np.float32)
        self.rng = np.random.default_rng(101)

    def test_depthwise_forward_and_backward_match_naive_and_are_deterministic(self):
        for x_shape, k_shape, stride, padding, dilation in self.CASES:
            with self.subTest(x=x_shape, k=k_shape, stride=stride, padding=padding, dilation=dilation):
                x = self.rng.normal(size=x_shape).astype(np.float32)
                k = self.rng.normal(size=k_shape).astype(np.float32)
                ref_y = _naive_depthwise_forward(x, k, stride, padding, dilation)
                dy = self.rng.normal(size=ref_y.shape).astype(np.float32)
                ref_dx, ref_dk = _naive_depthwise_backward(x, k, dy, stride, padding, dilation)

                # Exercise non-contiguous strided GPU input without explicit pre-copy.
                n, c, h, w = x_shape
                host_wide = np.empty((n, c, h, w * 2), dtype=np.float32)
                host_wide[:, :, :, ::2] = x
                host_wide[:, :, :, 1::2] = -999.0
                x_noncontig = xp.asarray(host_wide)[:, :, :, ::2]
                self.assertFalse(x_noncontig.flags.c_contiguous)

                def _run():
                    x_node = Variable(x_noncontig)
                    k_node = Variable(xp.asarray(k.copy()), trainable=True)
                    op = Conv2D_Op(
                        x_node, k_node, stride=stride, padding=padding, dilation=dilation, groups=c
                    )
                    op.forward(x_node.value, k_node.value)
                    x_node.clear_grad()
                    k_node.clear_grad()
                    op.grad = xp.asarray(dy)
                    op.backward()
                    return asnumpy(op.value), asnumpy(x_node.grad), asnumpy(k_node.grad)

                y1, dx1, dk1 = _run()
                y2, dx2, dk2 = _run()
                np.testing.assert_allclose(y1, ref_y, rtol=1e-4, atol=1e-4)
                np.testing.assert_allclose(dx1, ref_dx, rtol=1e-4, atol=1e-4)
                np.testing.assert_allclose(dk1, ref_dk, rtol=1e-4, atol=1e-4)
                self.assertEqual(y1.tobytes(), y2.tobytes())
                self.assertEqual(dx1.tobytes(), dx2.tobytes())
                self.assertEqual(dk1.tobytes(), dk2.tobytes())

    def test_depthwise_cuda_grad_flags_and_legacy_xpad_backward(self):
        x_shape, k_shape, stride, padding, dilation = ((2, 4, 7, 7), (4, 1, 3, 3), (2, 1), (1, 2), (2, 1))
        x = self.rng.normal(size=x_shape).astype(np.float32)
        k = self.rng.normal(size=k_shape).astype(np.float32)
        ref_y = _naive_depthwise_forward(x, k, stride, padding, dilation)
        dy = self.rng.normal(size=ref_y.shape).astype(np.float32)
        ref_dx, ref_dk = _naive_depthwise_backward(x, k, dy, stride, padding, dilation)

        dx_arr = xp.asarray(x)
        dk_arr = xp.asarray(k)
        ddy_arr = xp.asarray(dy)
        got_y = depthwise_conv2d_forward_cuda(dx_arr, dk_arr, stride, padding, dilation)
        np.testing.assert_allclose(asnumpy(got_y), ref_y, rtol=1e-4, atol=1e-4)

        gk_none, gx_only = depthwise_conv2d_backward_cuda(
            dx_arr, dk_arr, ddy_arr, x_shape, stride, padding, dilation,
            need_input_grad=True, need_weight_grad=False,
        )
        self.assertIsNone(gk_none)
        np.testing.assert_allclose(asnumpy(gx_only), ref_dx, rtol=1e-4, atol=1e-4)

        gk_only, gx_none = depthwise_conv2d_backward_cuda(
            dx_arr, dk_arr, ddy_arr, x_shape, stride, padding, dilation,
            need_input_grad=False, need_weight_grad=True,
        )
        self.assertIsNone(gx_none)
        np.testing.assert_allclose(asnumpy(gk_only), ref_dk, rtol=1e-4, atol=1e-4)

        # Legacy caller compatibility: x_pad of padded spatial dimensions passed into depthwise_conv2d_backward.
        ph, pw = padding
        x_pad = xp.pad(dx_arr, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant")
        gk_legacy, gx_legacy = depthwise_conv2d_backward(
            x_pad, dk_arr, ddy_arr, x_shape, stride, padding, dilation,
            need_input_grad=True, need_weight_grad=True,
        )
        np.testing.assert_allclose(asnumpy(gk_legacy), ref_dk, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(asnumpy(gx_legacy), ref_dx, rtol=1e-4, atol=1e-4)

        # Unpadded forward/backward via public wrapper also matches.
        y_pub, saved_x = depthwise_conv2d_forward(dx_arr, dk_arr, stride, padding, dilation)
        self.assertEqual(saved_x.shape, x_shape)
        gk_pub, gx_pub = depthwise_conv2d_backward(
            saved_x, dk_arr, ddy_arr, x_shape, stride, padding, dilation,
            need_input_grad=True, need_weight_grad=True,
        )
        np.testing.assert_allclose(asnumpy(y_pub), ref_y, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(asnumpy(gk_pub), ref_dk, rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(asnumpy(gx_pub), ref_dx, rtol=1e-4, atol=1e-4)

    def test_mobilenet_v2_forward_and_backward_cuda_matches_cpu_reference(self):
        rng = np.random.default_rng(77)
        x_np = rng.uniform(0.0, 1.0, size=(2, 3, 64, 64)).astype(np.float32)
        target_np = rng.normal(size=(2, 2)).astype(np.float32)

        def _run_on(device: str, state: dict | None = None):
            set_device(device)
            model = MobileNetV2(
                output_dim=2,
                dropout=0.0,
                input_norm="imagenet",
                initializer=make_initializer("kaiming_normal", seed=13),
            )
            if state is not None:
                for idx, p in enumerate(model.params):
                    p.value[...] = xp.asarray(state["params"][idx])
                bns = [m for m in model.sub_layers if hasattr(m, "running_mean")]
                for idx, bn in enumerate(bns):
                    bn.running_mean[...] = xp.asarray(state["rm"][idx])
                    bn.running_var[...] = xp.asarray(state["rv"][idx])
            else:
                # Shift BN beta slightly away from ReLU6's zero kink so 1e-7 rounding cannot flip discrete subgradients.
                for p in model.params:
                    if p.name == "beta":
                        p.value[...] = p.value + xp.asarray(0.25, dtype=p.value.dtype)
            model.train()
            x_var = Variable(xp.asarray(x_np), name="x", requires_grad=False)
            t_var = Variable(xp.asarray(target_np), name="y", requires_grad=False)
            pred = model(x_var)
            loss = MSELoss(pred, t_var)
            graph = Graph(loss, optimize=False)
            graph.forward()
            for p in model.params:
                p.clear_grad()
            graph.backward()
            snapshot = {
                "params": [asnumpy(p.value).copy() for p in model.params],
                "rm": [asnumpy(m.running_mean).copy() for m in model.sub_layers if hasattr(m, "running_mean")],
                "rv": [asnumpy(m.running_var).copy() for m in model.sub_layers if hasattr(m, "running_var")],
            }
            grads = [asnumpy(p.grad).copy() for p in model.params]
            return asnumpy(pred.value).copy(), float(asnumpy(loss.value)), grads, snapshot

        cpu_out, cpu_loss, cpu_grads, snapshot = _run_on("cpu")
        cuda_out, cuda_loss, cuda_grads, _ = _run_on("cuda", snapshot)

        np.testing.assert_allclose(cuda_out, cpu_out, rtol=1e-4, atol=1e-4)
        self.assertAlmostEqual(cuda_loss, cpu_loss, places=4)
        self.assertEqual(len(cuda_grads), len(cpu_grads))
        for idx, (cg, rg) in enumerate(zip(cuda_grads, cpu_grads)):
            np.testing.assert_allclose(cg, rg, rtol=1e-3, atol=1e-3, err_msg=f"grad mismatch at param {idx}")


if __name__ == "__main__":
    unittest.main()
