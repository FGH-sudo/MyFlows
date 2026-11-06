"""Validate named tensors and shapes before applying pretrained weights."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..core.device import xp

IGNORED_SUFFIX = ".num_batches_tracked"
NUM_BATCHES_TRACKED_POLICY = ("ignored: MyFlows BN uses a fixed momentum (0.1), for which torch does not read "
                              "num_batches_tracked either")


class PretrainedLoadError(ValueError):
    """严格模式下主干参数缺失、多余或形状不符。"""

    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


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


class CheckpointLoader:
    def load_converted(self, model, path, *, replace_head=True, strict=True):
        return load_torchvision_weights(model, path, replace_head=replace_head, strict=strict)

    def load_checkpoint(self, model, path, *, optimizer=None, allow_dtype_conversion=False):
        from ..utils.checkpoint import load_checkpoint, named_state, read_checkpoint_metadata, resolve_checkpoint_paths

        metadata = read_checkpoint_metadata(path)
        if metadata is None:
            raise FileNotFoundError(path)
        _, npz = resolve_checkpoint_paths(path)
        params, buffers = named_state(model)
        live = {k: v.value for k, v in params.items()}
        live.update({k: getattr(layer, attr) for k, (layer, attr) in buffers.items()})
        entries = metadata["model"].get("params", []) + metadata["model"].get("buffers", [])
        names = [entry["key"] for entry in entries]
        report = {"source": "MyFlows", "npz_sha256": file_sha256(npz), "missing": sorted(set(live) - set(names)),
                  "unexpected": sorted(set(names) - set(live)), "shape_mismatch": [], "dtype_conversions": [],
                  "loaded": 0, "expected": len(live)}
        if len(set(names)) != len(names):
            raise PretrainedLoadError("duplicate tensor names in checkpoint", report)
        with np.load(npz, allow_pickle=False) as arrays:
            for name in set(live).intersection(names):
                if name not in arrays.files:
                    report["missing"].append(name)
                    continue
                value = arrays[name]
                if value.shape != live[name].shape:
                    report["shape_mismatch"].append({"name": name, "source": list(value.shape), "target": list(live[name].shape)})
                if value.dtype != live[name].dtype:
                    report["dtype_conversions"].append({"name": name, "source": str(value.dtype), "target": str(live[name].dtype)})
        if report["missing"] or report["unexpected"] or report["shape_mismatch"] or (report["dtype_conversions"] and not allow_dtype_conversion):
            raise PretrainedLoadError("checkpoint tensors do not match the model", report)
        # The native checkpoint API restores arrays in checkpoint order. Verify that order too.
        if names != list(live):
            raise PretrainedLoadError("checkpoint tensor order differs from the model", report)
        epoch, score = load_checkpoint([model], optimizer, str(path))
        return {**report, "loaded": len(live), "epoch": epoch, "score": score}
