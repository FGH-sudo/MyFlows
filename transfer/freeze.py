"""Stage and name-based parameter freezing, BN statistics and frozen tensor fingerprints."""

from __future__ import annotations

import fnmatch
import hashlib

import numpy as np

from ..core.device import asnumpy
from ..core.node import Variable
from ..layers.layer import Layer
from ..layers.resnet import BatchNorm2d
from ..utils.checkpoint import collect_layers

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


def named_parameters(model):
    """Official torchvision keys or stable attribute paths, deduplicated by variable identity."""
    try:
        official = model.torchvision_tensors()
    except (AttributeError, ValueError):
        official = ()
    result = {name: node for name, node, kind in official if kind != "buffer"}
    seen = {id(node) for node in result.values()}
    visited = set()
    def walk(layer, path):
        if id(layer) in visited:
            return
        visited.add(id(layer))
        for attr, value in vars(layer).items():
            if attr.startswith("_") or attr in ("params", "sub_layers"):
                continue
            prefix = f"{path}.{attr}" if path else attr
            if isinstance(value, Variable) and id(value) not in seen:
                result[prefix] = value
                seen.add(id(value))
            elif isinstance(value, Layer):
                walk(value, prefix)
            elif isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, Layer):
                        walk(item, f"{prefix}.{i}")
        for i, child in enumerate(getattr(layer, "sub_layers", ())):
            walk(child, f"{path + '.' if path else ''}layers.{i}")
        for i, node in enumerate(getattr(layer, "params", ())):
            if id(node) not in seen:
                result[f"{path + '.' if path else ''}params.{i}"] = node
                seen.add(id(node))
    walk(model, "")
    members = set(model.params)
    return {name: p for name, p in result.items() if p in members}


class ParameterFreezer:
    MODES = ("both", "params", "bn")

    def __init__(self, model):
        self.model = model
        self.level = 0
        self.names = named_parameters(model)
        self.batchnorms = [layer for layer in collect_layers([model]) if isinstance(layer, BatchNorm2d)]

    @staticmethod
    def _statistics(layer, frozen):
        changed = layer.frozen != frozen
        layer.frozen = frozen
        if frozen or changed:
            layer.train(not frozen)

    def stages(self, count, *, mode="both", bn_unfreeze="follow"):
        stages = self.model.stages()
        if mode not in self.MODES or bn_unfreeze not in ("follow", "keep-frozen") or not 0 <= count <= len(stages):
            raise ValueError("invalid freeze mode, BN policy or stage count")
        for i, (_, layers) in enumerate(stages):
            frozen = i < count
            for layer in _stage_layers(layers):
                if mode in ("both", "params"):
                    for param in getattr(layer, "params", ()):
                        param.trainable = not frozen
                if mode in ("both", "bn") and isinstance(layer, BatchNorm2d):
                    self._statistics(layer, frozen or (bn_unfreeze == "keep-frozen" and layer.frozen))
        self.level = count
        return freeze_summary(self.model)

    def by_name(self, patterns, *, frozen=True, mode="both"):
        if mode not in self.MODES:
            raise ValueError(f"freeze mode must be one of {self.MODES}")
        patterns = (patterns,) if isinstance(patterns, str) else tuple(patterns)
        matched = {k: v for k, v in self.names.items() if any(fnmatch.fnmatchcase(k, p) for p in patterns)}
        if not matched:
            raise ValueError(f"no parameters match {patterns}")
        if mode in ("params", "both"):
            for param in matched.values():
                param.trainable = not frozen
        if mode in ("bn", "both"):
            selected = set(matched.values())
            for bn in self.batchnorms:
                if selected.intersection(bn.params):
                    self._statistics(bn, frozen)
        return sorted(matched)


def parse_schedule(value):
    """Map zero-based epoch indices to a number of frozen stages."""
    if value is None or value == "":
        return {}
    try:
        entries = [(int(k), int(v)) for k, v in (part.split(":") for part in value.split(","))]
    except (TypeError, ValueError) as exc:
        raise ValueError("unfreeze schedule must have the form 0:4,2:3,4:2") from exc
    if len(dict(entries)) != len(entries) or any(k < 0 or v < 0 for k, v in entries):
        raise ValueError("unfreeze schedule needs unique nonnegative epochs and stage counts")
    return dict(sorted(entries))
