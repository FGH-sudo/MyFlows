import json
import re
from pathlib import Path

import numpy as np

from ..core.device import asnumpy, xp


_CHECKPOINT_VERSION = 1


def _safe_name(name):
    text = "unnamed" if name is None else str(name)
    return re.sub(r"[^0-9a-zA-Z_]+", "_", text).strip("_") or "unnamed"


def _json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return None


def _as_roots(layers):
    if hasattr(layers, "params") or hasattr(layers, "sub_layers"):
        return [layers]
    return list(layers)


def collect_layers(layers):
    roots = _as_roots(layers)
    collected = []
    seen = set()

    def visit(layer):
        if id(layer) in seen:
            return
        seen.add(id(layer))
        collected.append(layer)
        for child in getattr(layer, "sub_layers", []):
            visit(child)

    for root in roots:
        visit(root)
    return collected


def resolve_checkpoint_paths(filepath):
    path = Path(filepath)
    if path.suffix == ".json":
        return path, path.with_suffix(".npz")
    if path.suffix == ".npz":
        return path.with_suffix(".json"), path
    return path.with_suffix(".json"), path.with_suffix(".npz")


def _collect_model_arrays(layers):
    layer_list = collect_layers(layers)
    arrays = {}
    params = []
    buffers = []
    param_key_by_id = {}

    seen_params = set()
    for layer_index, layer in enumerate(layer_list):
        layer_name = getattr(layer, "name", None)
        layer_class = layer.__class__.__name__
        for param_index, param in enumerate(getattr(layer, "params", [])):
            if id(param) in seen_params:
                continue
            seen_params.add(id(param))
            key = f"param_{len(params):04d}_{_safe_name(getattr(param, 'name', None))}"
            value = asnumpy(param.value)
            arrays[key] = value
            param_key_by_id[id(param)] = key
            params.append({
                "key": key,
                "name": getattr(param, "name", None),
                "layer_index": layer_index,
                "layer_name": layer_name,
                "layer_class": layer_class,
                "param_index": param_index,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "trainable": bool(getattr(param, "trainable", False)),
            })

        for attr in ("running_mean", "running_var"):
            if not hasattr(layer, attr):
                continue
            value = getattr(layer, attr)
            if not hasattr(value, "shape"):
                continue
            key = f"buffer_{len(buffers):04d}_{_safe_name(layer_name)}_{attr}"
            arrays[key] = asnumpy(value)
            buffers.append({
                "key": key,
                "name": attr,
                "layer_index": layer_index,
                "layer_name": layer_name,
                "layer_class": layer_class,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            })

    layers_meta = [
        {
            "index": index,
            "name": getattr(layer, "name", None),
            "class": layer.__class__.__name__,
        }
        for index, layer in enumerate(layer_list)
    ]
    return layer_list, arrays, params, buffers, layers_meta, param_key_by_id


def _collect_optimizer_arrays(optimizer, param_key_by_id):
    arrays = {}
    state_meta = {}
    if optimizer is None:
        return arrays, {"class": None, "scalars": {}, "states": state_meta}

    for state_name in ("v", "m", "s", "acc_gradient"):
        state = getattr(optimizer, state_name, None)
        if not isinstance(state, dict):
            continue
        entries = []
        for node, value in state.items():
            param_key = param_key_by_id.get(id(node))
            if param_key is None or value is None:
                continue
            array_key = f"optimizer_{state_name}_{param_key}"
            arrays[array_key] = asnumpy(value)
            entries.append({"param_key": param_key, "array_key": array_key})
        state_meta[state_name] = entries

    scalars = {}
    for attr in (
        "learning_rate",
        "momentum",
        "beta",
        "beta_1",
        "beta_2",
        "eps",
        "t",
        "acc_no",
    ):
        if not hasattr(optimizer, attr):
            continue
        value = _json_scalar(getattr(optimizer, attr))
        if value is not None:
            scalars[attr] = value

    return arrays, {
        "class": optimizer.__class__.__name__,
        "scalars": scalars,
        "states": state_meta,
    }


def save_checkpoint(layers, optimizer=None, epoch=0, acc=0.0, filepath="checkpoint"):
    """以 JSON + NPZ 格式保存模型权重、BN buffer 与优化器状态。"""
    json_path, npz_path = resolve_checkpoint_paths(filepath)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    _, model_arrays, params, buffers, layers_meta, param_key_by_id = _collect_model_arrays(layers)
    opt_arrays, optimizer_meta = _collect_optimizer_arrays(optimizer, param_key_by_id)

    arrays = {}
    arrays.update(model_arrays)
    arrays.update(opt_arrays)

    metadata = {
        "format": "MyFlows checkpoint",
        "version": _CHECKPOINT_VERSION,
        "array_file": npz_path.name,
        "epoch": int(epoch),
        "best_acc": None if acc is None else float(acc),
        "model": {
            "layers": layers_meta,
            "params": params,
            "buffers": buffers,
        },
        "optimizer": optimizer_meta,
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    np.savez_compressed(npz_path, **arrays)
    return str(json_path), str(npz_path)


def load_checkpoint(layers, optimizer=None, filepath="checkpoint"):
    """加载 JSON + NPZ checkpoint，并返回 ``(epoch, best_acc)``。"""
    json_path, npz_path = resolve_checkpoint_paths(filepath)
    if not json_path.exists() or not npz_path.exists():
        print("未发现 Checkpoint，从零开始训练。")
        return -1, 0.0

    with json_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    if metadata.get("version") != _CHECKPOINT_VERSION:
        raise ValueError(f"不支持的 checkpoint 版本: {metadata.get('version')}")

    _, _, live_params, live_buffers, _, _ = _collect_model_arrays(layers)
    saved_params = metadata["model"].get("params", [])
    saved_buffers = metadata["model"].get("buffers", [])
    if len(saved_params) != len(live_params):
        raise ValueError(f"参数数量不匹配: checkpoint={len(saved_params)}, model={len(live_params)}")

    saved_key_to_var = {}
    with np.load(npz_path, allow_pickle=False) as data:
        for saved, live in zip(saved_params, live_params):
            value = np.asarray(data[saved["key"]])
            expected_shape = tuple(live["shape"])
            if value.shape != expected_shape:
                raise ValueError(
                    f"参数形状不匹配 {saved['key']}: checkpoint={value.shape}, model={expected_shape}"
                )
            live_var = _param_by_live_record(layers, live)
            live_var.value = value.copy()
            saved_key_to_var[saved["key"]] = live_var

        if len(saved_buffers) == len(live_buffers):
            live_layer_list = collect_layers(layers)
            for saved, live in zip(saved_buffers, live_buffers):
                value = np.asarray(data[saved["key"]])
                layer = live_layer_list[live["layer_index"]]
                current = getattr(layer, live["name"])
                if current.shape != value.shape:
                    raise ValueError(
                        f"buffer 形状不匹配 {saved['key']}: checkpoint={value.shape}, model={current.shape}"
                    )
                arr = xp.asarray(value, dtype=current.dtype)
                current[...] = arr

        _restore_optimizer(optimizer, metadata.get("optimizer", {}), data, saved_key_to_var)

    epoch = int(metadata.get("epoch", -1))
    best_acc = metadata.get("best_acc", 0.0)
    if optimizer is not None:
        print(f"--- 已成功加载断点，准备从 Epoch {epoch + 1} 继续训练 ---")
    return epoch, 0.0 if best_acc is None else float(best_acc)


def _param_by_live_record(layers, live_record):
    layer = collect_layers(layers)[live_record["layer_index"]]
    candidates = getattr(layer, "params", [])
    param_index = live_record["param_index"]
    if param_index >= len(candidates):
        raise ValueError(f"模型参数索引不存在: {param_index}")
    return candidates[param_index]


def _restore_optimizer(optimizer, optimizer_meta, data, saved_key_to_var):
    if optimizer is None:
        return

    for attr, value in optimizer_meta.get("scalars", {}).items():
        if hasattr(optimizer, attr):
            setattr(optimizer, attr, value)

    for state_name, entries in optimizer_meta.get("states", {}).items():
        if not hasattr(optimizer, state_name):
            continue
        state = {}
        for entry in entries:
            var = saved_key_to_var.get(entry["param_key"])
            if var is not None:
                state[var] = xp.asarray(data[entry["array_key"]]).copy()
        setattr(optimizer, state_name, state)
