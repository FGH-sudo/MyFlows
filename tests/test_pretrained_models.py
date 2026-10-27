# -*- coding: utf-8 -*-
"""E2：ResNet18(compat='torchvision')、MobileNetV2、AlexNet 与 torchvision 结构的输出一致性，权重映射报告，
分阶段冻结（含 BN buffer），以及新增算子（maxpool padding、深度卷积、ReLU6、自适应池化、无偏 running_var）
与 torch 的前向/反向比对。

参考模型是按 torchvision 定义写的 torch.nn 模块（本环境未装 torchvision）；它们以 strict=True 载入官方
state_dict，名称与形状因此与官方完全一致。官方权重已转换（runs/pretrained/converted/*.npz）时，
另用真实权重和真实图片比对 1000 类输出。容差按 semester_spec §4.6：FP32 atol=1e-4、rtol=1e-3。
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_PARENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_PARENT))

import numpy as np

from MyFlows.core.device import cuda_available, set_device, xp
from MyFlows.core.graph import Graph
from MyFlows.core.node import Variable
from MyFlows.ops.activation import ReLU6
from MyFlows.ops.convolution import Conv2D_Op, MaxPool2d_Op
from MyFlows.ops.loss import CrossEntropy
from MyFlows.ops.pooling import AdaptiveAvgPool2d_Op, IMAGENET_MEAN, IMAGENET_STD
from MyFlows.layers.resnet import BatchNorm2d, ResNet18
from MyFlows.train.opt import Adam
from MyFlows.utils import pretrained

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover
    torch = None

ATOL, RTOL = 1e-4, 1e-3
CONVERTED = PROJECT_PARENT / "runs" / "pretrained" / "converted"
DEVICES = ("cpu", "cuda") if cuda_available() else ("cpu",)


def _host(value):
    return value if isinstance(value, np.ndarray) else value.get()


if torch is not None:
    from MyFlows.tests.torch_reference import TORCH_MODELS


def _randomize_bn(module, seed):
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for m in module.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.copy_(torch.rand(m.weight.shape, generator=gen) + 0.5)
                m.bias.copy_(torch.randn(m.bias.shape, generator=gen) * 0.1)
                m.running_mean.copy_(torch.randn(m.running_mean.shape, generator=gen) * 0.1)
                m.running_var.copy_(torch.rand(m.running_var.shape, generator=gen) + 0.5)


def _save_state(state, directory, name="weights"):
    path = Path(directory) / f"{name}.npz"
    np.savez(path, **{k: v.detach().cpu().numpy() for k, v in state.items()})
    return path


def _forward(model, x):
    x_var = Variable(xp.asarray(x), name="X", requires_grad=False)
    out = model(x_var)
    Graph(out).forward()
    return _host(out.value)


def _assert_close(test, actual, expected, label):
    error = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    bound = ATOL + RTOL * np.abs(expected.astype(np.float64))
    worst = float((error - bound).max())
    test.assertLessEqual(worst, 0.0, f"{label}: max |diff| {error.max():.3e} exceeds atol={ATOL} rtol={RTOL}")


def _donkey_images(count, size=224):
    """真实图片（Donkey tub 的前几张），按 ImageNet 方式 resize；没有数据时返回 None。"""
    images = sorted((PROJECT_PARENT / "mycar" / "data").glob("**/images/*.jpg"))[:count]
    if len(images) < count:
        return None
    from apps.common.image_preprocess import imread_nchw
    return np.concatenate([imread_nchw(p, (size, size), np.float32) for p in images])


def _normalize(x):
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 3, 1, 1)
    return (x - mean) / std


class _DeviceCase(unittest.TestCase):
    def tearDown(self):
        set_device("cpu")


@unittest.skipIf(torch is None, "torch is not installed")
class OpsMatchTorchTest(_DeviceCase):
    """新增算子的前向与反向和 torch 一致（CPU 与 GPU）。"""

    def _check(self, device, mf_fn, torch_fn, x, params=()):
        set_device(device)
        x_var = Variable(xp.asarray(x), name="x")
        p_vars = [Variable(xp.asarray(p), name=f"p{i}", trainable=True) for i, p in enumerate(params)]
        out = mf_fn(x_var, *p_vars)
        graph = Graph(out)
        graph.forward()
        dy = np.random.default_rng(3).standard_normal(out.value.shape).astype(np.float32)
        for node in graph.nodes:
            node.clear_grad()
        out.grad = xp.asarray(dy)
        for node in reversed(graph.sorted_nodes):
            node.backward()

        xt = torch.tensor(x, requires_grad=True)
        pt = [torch.tensor(p, requires_grad=True) for p in params]
        yt = torch_fn(xt, *pt)
        yt.backward(torch.tensor(dy))
        _assert_close(self, _host(out.value), yt.detach().numpy(), f"{device} forward")
        _assert_close(self, _host(x_var.grad), xt.grad.numpy(), f"{device} dx")
        for var, ref in zip(p_vars, pt):
            _assert_close(self, _host(var.grad), ref.grad.numpy(), f"{device} dparam")

    def test_maxpool_padding(self):
        x = np.random.default_rng(0).standard_normal((2, 3, 11, 9)).astype(np.float32)
        for device in DEVICES:
            for backend in ("auto",) + (("cuda_native_register",) if device == "cuda" else ()):
                with self.subTest(device=device, backend=backend):
                    self._check(device, lambda v: MaxPool2d_Op(v, 3, 2, backend=backend, padding=1),
                                lambda t: F.max_pool2d(t, 3, 2, 1), x)

    def test_depthwise_convolution(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal((2, 6, 9, 10)).astype(np.float32)
        kernel = rng.standard_normal((6, 1, 3, 3)).astype(np.float32)
        bias = rng.standard_normal(6).astype(np.float32)
        for device in DEVICES:
            for stride in (1, 2):
                with self.subTest(device=device, stride=stride):
                    self._check(device, lambda v, k, b: Conv2D_Op(v, k, stride=stride, padding=1, groups=6, bias=b),
                                lambda t, k, b: F.conv2d(t, k, b, stride=stride, padding=1, groups=6),
                                x, (kernel, bias))

    def test_relu6(self):
        x = (np.random.default_rng(2).standard_normal((3, 4, 5, 5)) * 5).astype(np.float32)
        for device in DEVICES:
            with self.subTest(device=device):
                self._check(device, ReLU6, F.relu6, x)

    def test_adaptive_avg_pool(self):
        for shape, size in (((2, 3, 13, 13), (6, 6)), ((2, 3, 2, 4), (6, 6)), ((1, 2, 7, 5), (3, 2)), ((1, 2, 6, 6), (6, 6))):
            x = np.random.default_rng(4).standard_normal(shape).astype(np.float32)
            for device in DEVICES:
                with self.subTest(device=device, shape=shape, size=size):
                    self._check(device, lambda v: AdaptiveAvgPool2d_Op(v, size), lambda t: F.adaptive_avg_pool2d(t, size), x)

    def test_unbiased_running_var_matches_torch(self):
        rng = np.random.default_rng(5)
        x = (rng.standard_normal((4, 3, 5, 6)) * 2 + 1).astype(np.float32)
        ref = nn.BatchNorm2d(3)
        ref.train()
        ref(torch.tensor(x))
        for device in DEVICES:
            with self.subTest(device=device):
                set_device(device)
                bn = BatchNorm2d(3, unbiased_running_var=True)
                Graph(bn(Variable(xp.asarray(x), name="x"))).forward()
                _assert_close(self, _host(bn.running_var), ref.running_var.numpy(), "running_var")
                _assert_close(self, _host(bn.running_mean), ref.running_mean.numpy(), "running_mean")
                legacy = BatchNorm2d(3)
                Graph(legacy(Variable(xp.asarray(x), name="x"))).forward()
                biased = 0.9 + 0.1 * x.astype(np.float64).var(axis=(0, 2, 3))
                _assert_close(self, _host(legacy.running_var), biased.astype(np.float32), "legacy running_var")


@unittest.skipIf(torch is None, "torch is not installed")
class ArchitectureMatchesTorchvisionTest(_DeviceCase):
    """随机权重（含随机 BN buffer）：名称、形状完全一致，eval 输出在容差内。"""

    def _compare(self, arch, size, batch=2, seed=0):
        torch.manual_seed(seed)
        reference = TORCH_MODELS[arch]()
        _randomize_bn(reference, seed)
        reference.eval()
        x = np.random.default_rng(seed).standard_normal((batch, 3, size, size)).astype(np.float32)
        with torch.no_grad():
            expected = reference(torch.tensor(x)).numpy()
        with tempfile.TemporaryDirectory() as tmp:
            path = _save_state(reference.state_dict(), tmp)
            model = pretrained.build_model(arch, 1000)
            report = pretrained.load_torchvision_weights(model, path)
        model.eval()
        self.assertEqual((report["missing"], report["unexpected"], report["shape_mismatch"]), ([], [], []))
        self.assertEqual(report["loaded"], report["expected"])
        self.assertEqual(report["success_rate"], 1.0)
        _assert_close(self, _forward(model, x), expected, arch)
        return report

    def test_resnet18(self):
        report = self._compare("resnet18", 64)
        self.assertEqual(report["loaded"], 102)  # 20 卷积 + 20 个 BN × 4 + fc；torch 另存的 20 个 num_batches_tracked 按策略忽略
        self.assertEqual(report["ignored"]["keys"], 20)

    def test_mobilenet_v2(self):
        report = self._compare("mobilenet_v2", 64)
        self.assertEqual(report["ignored"]["keys"], 52)

    def test_alexnet(self):
        # 64 输入时最后特征图为 1x1，自适应池化把它展开到 6x6；224 输入时恰为 6x6
        self._compare("alexnet", 64)

    def test_resnet18_old_structure_is_unchanged(self):
        x = Variable(np.zeros((1, 3, 224, 224), dtype=np.float32), name="X", requires_grad=False)
        old, new = ResNet18(output_dim=2, stem="imagenet"), ResNet18(output_dim=1000, compat="torchvision")
        Graph(old(x)).forward()
        Graph(new(x)).forward()
        self.assertEqual(old._last_feature_nodes["layer1"].value.shape[2:], (55, 55))
        self.assertEqual(new._last_feature_nodes["layer1"].value.shape[2:], (56, 56))
        self.assertIsNotNone(old.stem_conv.b)
        self.assertIsNone(new.stem_conv.b)
        self.assertEqual(len(old.params) - len(new.params), 20)  # 20 个卷积偏置
        with self.assertRaises(ValueError):
            old.torchvision_tensors()


@unittest.skipIf(torch is None, "torch is not installed")
class OfficialWeightsTest(_DeviceCase):
    """官方 ImageNet 权重：真实图片、相同预处理，1000 类输出与来源模型一致。"""

    def _official(self, arch):
        path = CONVERTED / f"{arch}.npz"
        if not path.exists():
            self.skipTest(f"{path} not converted (tools/convert_torchvision_weights.py)")
        return path

    def _compare(self, arch):
        path = self._official(arch)
        images = _donkey_images(2)
        x = images if images is not None else np.random.default_rng(0).random((2, 3, 224, 224), dtype=np.float32)
        reference = TORCH_MODELS[arch]()
        with np.load(path) as data:
            reference.load_state_dict({k: torch.from_numpy(data[k]) for k in data.files}, strict=True)
        reference.eval()
        with torch.no_grad():
            expected = reference(torch.tensor(_normalize(x))).numpy()
        for device in DEVICES:
            with self.subTest(device=device):
                # GPU 上用训练默认的原生卷积引擎（深度卷积仍走 CuPy）
                set_device(device)
                model = pretrained.build_model(arch, 1000, input_norm="imagenet",
                                               conv_backend="cuda_native_register" if device == "cuda" else "auto")
                report = pretrained.load_torchvision_weights(model, path)
                model.eval()
                actual = _forward(model, x)
                _assert_close(self, actual, expected, f"{arch} on {device}")
                self.assertTrue(np.array_equal(actual.argmax(axis=1), expected.argmax(axis=1)))
        manifest = pretrained.read_manifest(path)
        self.assertEqual(report["source"]["source_sha256"], manifest["source_sha256"])
        self.assertTrue(manifest["source_sha256"].startswith(manifest["source_file"].rsplit("-", 1)[1][:8]))
        self.assertEqual(report["success_rate"], 1.0)

    def test_resnet18(self):
        self._compare("resnet18")

    def test_mobilenet_v2(self):
        self._compare("mobilenet_v2")

    def test_alexnet(self):
        self._compare("alexnet")


class LoadReportTest(unittest.TestCase):
    def _weights(self, arch, tmp, drop=None, reshape=None):
        model = pretrained.build_model(arch, 1000)
        arrays = {}
        for key, target, kind in model.torchvision_tensors():
            value = _host(getattr(target[0], target[1]) if isinstance(target, tuple) else target.value)
            arrays[key] = np.ascontiguousarray(value.T) if kind == "param_t" else value
        arrays = {k: v + 1.0 for k, v in arrays.items()}
        arrays["bn1.num_batches_tracked"] = np.asarray(7, dtype=np.int64)
        if drop:
            arrays.pop(drop)
        if reshape:
            arrays[reshape] = arrays[reshape][:1]
        path = Path(tmp) / "w.npz"
        np.savez(path, **arrays)
        return path

    def test_replacing_the_head_reports_it_and_keeps_its_initialisation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._weights("resnet18", tmp)
            model = pretrained.build_model("resnet18", 2)
            head_before = _host(model.fc_W.value).copy()
            report = pretrained.load_torchvision_weights(model, path, replace_head=True)
        self.assertEqual([r["source"] for r in report["replaced_head"]], ["fc.weight", "fc.bias"])
        self.assertEqual(report["replaced_head"][0]["target_shape"], [512, 2])
        self.assertEqual(report["success_rate"], 1.0)
        self.assertEqual(report["ignored"]["keys"], 1)
        self.assertTrue(np.array_equal(_host(model.fc_W.value), head_before))
        self.assertTrue(any(m["transform"] for m in report["mapping"]) is False)  # 头被替换，没有转置项

    def test_strict_mode_rejects_missing_and_misshaped_weights_without_touching_the_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            for kwargs, field in (({"drop": "layer2.0.conv1.weight"}, "missing"),
                                  ({"reshape": "layer3.1.bn2.running_var"}, "shape_mismatch")):
                path = self._weights("resnet18", tmp, **kwargs)
                model = pretrained.build_model("resnet18", 1000)
                before = _host(model.stem_conv.kernel.value).copy()
                with self.assertRaises(pretrained.PretrainedLoadError) as caught:
                    pretrained.load_torchvision_weights(model, path)
                self.assertEqual(len(caught.exception.report[field]), 1)
                self.assertTrue(np.array_equal(_host(model.stem_conv.kernel.value), before))

    def test_unexpected_keys_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._weights("alexnet", tmp)
            with np.load(path) as data:
                arrays = dict(data)
            arrays["features.99.weight"] = np.zeros(3, dtype=np.float32)
            np.savez(path, **arrays)
            report = pretrained.load_torchvision_weights(pretrained.build_model("alexnet", 1000), path, strict=False)
        self.assertEqual(report["unexpected"], ["features.99.weight"])


class FreezeTest(_DeviceCase):
    """冻结：参数不更新、BN 用 running stats 且不更新、外层 train() 不覆盖；解冻后恢复。"""

    def _train_steps(self, model, x, labels, steps=2):
        x_var = Variable(xp.asarray(x), name="X", requires_grad=False)
        y_var = Variable(xp.asarray(labels), name="y", requires_grad=False)
        loss = CrossEntropy(model(x_var), y_var)
        graph = Graph(loss)
        optimizer = Adam(graph, learning_rate=1e-2)
        for _ in range(steps):
            optimizer.one_step()
            optimizer.update()
        return graph

    def _check(self, arch, size, count):
        model = pretrained.build_model(arch, 3, dropout=0.0)
        summary = pretrained.freeze_stages(model, count)
        self.assertEqual(summary["freeze_stages"], count)
        self.assertEqual(summary["frozen"], [name for name, _ in model.stages()[:count]])
        model.train()
        bns = [layer for layer in pretrained._stage_layers(model.stages()[0][1]) if isinstance(layer, BatchNorm2d)]
        for bn in bns:
            self.assertFalse(bn.training, "model.train() must not switch a frozen BN back to training")
        before = pretrained.frozen_fingerprint(model)
        trainable_before = [_host(p.value).copy() for p in model.params if p.trainable]
        rng = np.random.default_rng(0)
        x = rng.random((2, 3, size, size), dtype=np.float32)
        self._train_steps(model, x, np.array([0, 2]))
        self.assertEqual(pretrained.frozen_fingerprint(model), before)
        trainable_after = [_host(p.value) for p in model.params if p.trainable]
        self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(trainable_before, trainable_after)))
        return model, bns

    def test_resnet18_two_stages(self):
        model, bns = self._check("resnet18", 32, 2)
        pretrained.unfreeze_all(model)
        model.train()
        self.assertTrue(all(bn.training for bn in bns))
        self.assertTrue(all(p.trainable for p in model.params))
        self.assertEqual(pretrained.frozen_fingerprint(model)["tensors"], 0)

    def test_mobilenet_v2_all_stages_trains_only_the_head(self):
        model, _ = self._check("mobilenet_v2", 32, 5)
        self.assertEqual({id(p) for p in model.params if p.trainable}, {id(model.fc_W), id(model.fc_b)})

    def test_alexnet_three_convs(self):
        model, _ = self._check("alexnet", 64, 3)
        self.assertEqual(sum(1 for c in model.convs if c.kernel.trainable), 2)

    def test_out_of_range_is_rejected(self):
        with self.assertRaises(ValueError):
            pretrained.freeze_stages(pretrained.build_model("alexnet", 2), 6)


if __name__ == "__main__":
    unittest.main()
