"""ImageNet 预训练权重的载入、映射报告与分阶段冻结。

权重来源是 PyTorch 官方 torchvision 权重（.pth），由 ``tools/convert_torchvision_weights.py`` 在隔离的
解释器里用 ``torch.load(weights_only=True)`` 读出，存成以 torchvision 名称为键的 .npz，旁边的 .json
manifest 记录来源文件、校验和与 torch 版本。本模块只读 .npz，不依赖 torch。

模型需提供 ``torchvision_tensors()``、``head_keys`` 与 ``stages()``（见 ResNet18(compat='torchvision')、
MobileNetV2、AlexNet）。
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..core.device import asnumpy, xp
from ..layers.resnet import BatchNorm2d
from .checkpoint import collect_layers

# 官方权重：文件名中的 8 位十六进制是文件 sha256 的前缀
TORCHVISION_WEIGHTS = {
    "resnet18": {"file": "resnet18-f37072fd.pth", "url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
                 "weights": "ResNet18_Weights.IMAGENET1K_V1"},
    "mobilenet_v2": {"file": "mobilenet_v2-b0353104.pth",
                     "url": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
                     "weights": "MobileNet_V2_Weights.IMAGENET1K_V1"},
    "alexnet": {"file": "alexnet-owt-7be5be79.pth", "url": "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
                "weights": "AlexNet_Weights.IMAGENET1K_V1"},
}
ARCHS = tuple(TORCHVISION_WEIGHTS)
IGNORED_SUFFIX = ".num_batches_tracked"
NUM_BATCHES_TRACKED_POLICY = ("ignored: MyFlows BN uses a fixed momentum (0.1), for which torch does not read "
                              "num_batches_tracked either")


class PretrainedLoadError(ValueError):
    """严格模式下主干参数缺失、多余或形状不符。"""

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def build_model(arch, output_dim=1000, *, dropout=None, initializer=None, conv_backend="auto", dropout_seed=None,
                input_norm=None, name=None):
    """按名称构造与 torchvision 结构一致的模型。dropout=None 时用 torchvision 默认值。"""
    from ..layers.mobilenet import AlexNet, MobileNetV2
    from ..layers.resnet import ResNet18

    common = dict(initializer=initializer, conv_backend=conv_backend, dropout_seed=dropout_seed, input_norm=input_norm,
                  name=name)
    if arch == "resnet18":
        return ResNet18(in_channels=3, output_dim=output_dim, stem="imagenet", base_width=64,
                        dropout=0.0 if dropout is None else dropout, compat="torchvision", **common)
    if arch == "mobilenet_v2":
        return MobileNetV2(output_dim=output_dim, dropout=0.2 if dropout is None else dropout, **common)
    if arch == "alexnet":
        return AlexNet(output_dim=output_dim, dropout=0.5 if dropout is None else dropout, **common)
    raise ValueError(f"unknown arch {arch!r}; expected one of {ARCHS}")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(npz_path):
    manifest = Path(npz_path).with_suffix(".json")
    return json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else None


def _target_name(target):
    if isinstance(target, tuple):
        layer, attr = target
        return f"{type(layer).__name__}.{attr}"
    return getattr(target, "name", None) or type(target).__name__


def _target_shape(target):
    if isinstance(target, tuple):
        layer, attr = target
        return tuple(getattr(layer, attr).shape)
    return tuple(target.value.shape)


def _assign(target, array):
    if isinstance(target, tuple):
        layer, attr = target
        buffer = getattr(layer, attr)  # BN 算子持有同一个数组，只能原地写
        buffer[...] = xp.asarray(array, dtype=buffer.dtype)
    else:
        target.value = xp.asarray(array, dtype=target.value.dtype)


def load_torchvision_weights(model, npz_path, *, replace_head=False, strict=True):
    """按 torchvision 名称载入权重并返回映射报告。

    replace_head=True 时跳过任务头（``model.head_keys``），任务头保持模型自己的初始化，报告里列为 replaced_head；
    否则任务头也必须完全匹配（例如保留 1000 类头与来源模型比对输出）。strict=True 时主干有任何缺失、
    多余或形状不符都抛 PretrainedLoadError，参数不做任何修改。
    """
    npz_path = Path(npz_path)
    manifest = read_manifest(npz_path)
    npz_sha256 = file_sha256(npz_path)
    if manifest is not None and manifest.get("npz_sha256") not in (None, npz_sha256):
        raise PretrainedLoadError(f"{npz_path} does not match its manifest checksum", {"npz_sha256": npz_sha256})
    with np.load(npz_path, allow_pickle=False) as data:
        source = {key: data[key] for key in data.files}

    head = set(getattr(model, "head_keys", ()))
    mapping, missing, mismatched, replaced, pending = [], [], [], [], []
    mapped_keys = set()
    for key, target, kind in model.torchvision_tensors():
        mapped_keys.add(key)
        expected = _target_shape(target)
        if replace_head and key in head:
            replaced.append({"source": key, "target": _target_name(target), "source_shape":
                             list(source[key].shape) if key in source else None, "target_shape": list(expected)})
            continue
        if key not in source:
            missing.append(key)
            continue
        array = source[key]
        if kind == "param_t":
            array = np.ascontiguousarray(array.T)
        if tuple(array.shape) != expected:
            mismatched.append({"source": key, "source_shape": list(array.shape), "target_shape": list(expected)})
            continue
        pending.append((target, array))
        mapping.append({"source": key, "target": _target_name(target), "shape": list(expected),
                        "transform": "transpose [out,in] -> [in,out]" if kind == "param_t" else None,
                        "kind": "buffer" if kind == "buffer" else "param"})

    ignored = sorted(key for key in source if key.endswith(IGNORED_SUFFIX))
    unexpected = sorted(key for key in source if key not in mapped_keys and not key.endswith(IGNORED_SUFFIX))
    expected_count = len(mapped_keys) - len(replaced)
    report = {
        "arch": getattr(model, "arch", type(model).__name__),
        "npz": str(npz_path),
        "npz_sha256": npz_sha256,
        "source": None if manifest is None else {k: manifest.get(k) for k in (
            "source_file", "source_url", "source_sha256", "weights", "torch_version")},
        "expected": expected_count,
        "loaded": len(pending),
        "success_rate": len(pending) / expected_count if expected_count else 0.0,
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatch": mismatched,
        "replaced_head": replaced,
        "ignored": {"keys": len(ignored), "policy": NUM_BATCHES_TRACKED_POLICY} if ignored else {"keys": 0},
        "mapping": mapping,
    }
    if strict and (missing or unexpected or mismatched):
        raise PretrainedLoadError(
            f"pretrained weights do not match {report['arch']}: {len(missing)} missing, {len(unexpected)} unexpected, "
            f"{len(mismatched)} shape mismatches", report)
    for target, array in pending:
        _assign(target, array)
    return report


def summarize_load_report(report):
    """训练结果里保存的精简版：不含逐项 mapping。"""
    keep = ("arch", "npz", "npz_sha256", "source", "expected", "loaded", "success_rate", "missing", "unexpected",
            "shape_mismatch", "replaced_head", "ignored")
    return {key: report.get(key) for key in keep} | {"mapped": len(report.get("mapping", []))}


def _stage_layers(layers):
    return collect_layers(list(layers))


def freeze_stages(model, count):
    """冻结前 count 个阶段、解冻其余阶段（可重复调用）。

    冻结的阶段：参数 trainable=False（优化器跳过它们），BN 同时固定 running statistics 并保持 eval，
    外层 model.train() 不会把它们切回训练模式。解冻的 BN 在下一次 model.train() 时恢复训练模式。
    """
    stages = model.stages()
    count = int(count)
    if not 0 <= count <= len(stages):
        raise ValueError(f"freeze_stages must be in [0, {len(stages)}] for {getattr(model, 'arch', model)}")
    for index, (_, layers) in enumerate(stages):
        frozen = index < count
        for layer in _stage_layers(layers):
            if isinstance(layer, BatchNorm2d):
                layer.freeze(frozen)
            for param in getattr(layer, "params", []):
                param.trainable = not frozen
    return freeze_summary(model)


def unfreeze_all(model):
    return freeze_stages(model, 0)


def freeze_summary(model):
    stages = model.stages()
    frozen_names, frozen_tensors, frozen_values = [], 0, 0
    for name, layers in stages:
        params = {id(p): p for layer in _stage_layers(layers) for p in getattr(layer, "params", [])}
        if params and all(not p.trainable for p in params.values()):
            frozen_names.append(name)
        frozen = [p for p in params.values() if not p.trainable]
        frozen_tensors += len(frozen)
        frozen_values += sum(int(p.value.size) for p in frozen)
    trainable_values = sum(int(p.value.size) for p in model.params if p.trainable)
    return {"stages": [name for name, _ in stages], "freeze_stages": len(frozen_names), "frozen": frozen_names,
            "frozen_tensors": frozen_tensors, "frozen_values": frozen_values, "trainable_values": trainable_values}


def frozen_tensors(model):
    """冻结阶段里的参数与 BN buffer：[(标签, 数组)]，按阶段顺序。"""
    items = []
    for name, layers in model.stages():
        for layer_index, layer in enumerate(_stage_layers(layers)):
            if isinstance(layer, BatchNorm2d) and layer.frozen:
                items += [(f"{name}/{layer_index}/{attr}", getattr(layer, attr))
                          for attr in ("running_mean", "running_var")]
            for param_index, param in enumerate(getattr(layer, "params", [])):
                if not param.trainable:
                    items.append((f"{name}/{layer_index}/{param_index}:{param.name}", param.value))
    seen, unique = set(), []
    for label, array in items:
        if id(array) not in seen:
            seen.add(id(array))
            unique.append((label, array))
    return unique


def frozen_fingerprint(model):
    """冻结张量的 sha256（逐字节）：训练前后相同即冻结参数与 buffer 完全没变。"""
    digest = hashlib.sha256()
    tensors = frozen_tensors(model)
    for label, array in tensors:
        digest.update(label.encode("utf-8"))
        digest.update(np.ascontiguousarray(asnumpy(array)).tobytes())
    return {"tensors": len(tensors), "sha256": digest.hexdigest()}
