import json
import os
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


def named_state(layers):
    """Parameters and BN running statistics of ``layers`` under their checkpoint keys.

    Returns ``({key: parameter node}, {key: (layer, attribute)})`` in checkpoint order, so every
    process that builds the same model gets the same names (distributed training uses them).
    """
    params, buffers = {}, {}
    seen = set()
    for layer in collect_layers(layers):
        for param in getattr(layer, "params", []):
            if id(param) in seen:
                continue
            seen.add(id(param))
            params[f"param_{len(params):04d}_{_safe_name(getattr(param, 'name', None))}"] = param
        for attr in ("running_mean", "running_var"):
            if hasattr(getattr(layer, attr, None), "shape"):
                buffers[f"buffer_{len(buffers):04d}_{_safe_name(getattr(layer, 'name', None))}_{attr}"] = (layer, attr)
    return params, buffers


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


def _resolve_score(acc=None, loss=None, score=None, score_name=None):
    if loss is not None:
        return "loss", float(loss), "best_loss"
    if score is not None:
        name = "score" if score_name is None else str(score_name)
        key = "best_loss" if name == "loss" else None
        if name in ("acc", "accuracy"):
            name = "accuracy"
            key = "best_acc"
        return name, float(score), key
    if acc is not None:
        return "accuracy", float(acc), "best_acc"
    return None, None, None


def save_checkpoint(layers, optimizer=None, epoch=0, acc=None, filepath="checkpoint", *, loss=None, score=None,
                    score_name=None, training=None, compress=True):
    """以 JSON + NPZ 格式保存模型权重、BN buffer 与优化器状态。

    ``training`` 是可选的 JSON 字典（如训练进度），原样写入元数据，由 ``read_checkpoint_metadata`` 读回。
    ``compress=False`` 写不压缩的 NPZ（频繁保存的 TrainingState 用：浮点权重几乎压不小，压缩只是费时）。
    两个文件先写临时文件再替换：保存中途被结束时，原有断点仍然完整。
    """
    json_path, npz_path = resolve_checkpoint_paths(filepath)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    _, model_arrays, params, buffers, layers_meta, param_key_by_id = _collect_model_arrays(layers)
    opt_arrays, optimizer_meta = _collect_optimizer_arrays(optimizer, param_key_by_id)

    arrays = {}
    arrays.update(model_arrays)
    arrays.update(opt_arrays)
    metric_name, metric_value, legacy_metric_key = _resolve_score(
        acc=acc,
        loss=loss,
        score=score,
        score_name=score_name,
    )

    metadata = {
        "format": "MyFlows checkpoint",
        "version": _CHECKPOINT_VERSION,
        "array_file": npz_path.name,
        "epoch": int(epoch),
        "model": {
            "layers": layers_meta,
            "params": params,
            "buffers": buffers,
        },
        "optimizer": optimizer_meta,
    }
    if metric_name is not None:
        metadata["score_name"] = metric_name
        metadata["score_value"] = metric_value
        if legacy_metric_key is not None:
            metadata[legacy_metric_key] = metric_value
    if training is not None:
        metadata["training"] = training

    # 数组先落盘、元数据后落盘：替换到一半时，旧元数据仍指向同名、同形状的数组
    npz_tmp = npz_path.with_name(npz_path.name + ".tmp")
    with npz_tmp.open("wb") as f:
        (np.savez_compressed if compress else np.savez)(f, **arrays)
    os.replace(npz_tmp, npz_path)
    json_tmp = json_path.with_name(json_path.name + ".tmp")
    with json_tmp.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    os.replace(json_tmp, json_path)
    return str(json_path), str(npz_path)


def read_checkpoint_metadata(filepath):
    """Return the JSON metadata of a checkpoint without loading its arrays, or None when it is missing."""
    json_path, npz_path = resolve_checkpoint_paths(filepath)
    if not json_path.exists() or not npz_path.exists():
        return None
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_checkpoint_score(metadata):
    if "score_value" in metadata:
        return metadata.get("score_value")
    if "best_loss" in metadata:
        return metadata.get("best_loss")
    return metadata.get("best_acc", 0.0)


def load_checkpoint(layers, optimizer=None, filepath="checkpoint"):
    """Load JSON+NPZ checkpoint and return ``(epoch, score_value)``."""
    json_path, npz_path = resolve_checkpoint_paths(filepath)
    if not json_path.exists() or not npz_path.exists():
        if optimizer is None or json_path.exists() or npz_path.exists():
            raise FileNotFoundError(f"Checkpoint 不存在或不完整: {json_path} + {npz_path}")
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
    if len(saved_buffers) != len(live_buffers):
        raise ValueError(f"buffer 数量不匹配: checkpoint={len(saved_buffers)}, model={len(live_buffers)}")

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
            # 按模型参数当前的 dtype 恢复：旧的 float64 断点载入 float32 模型时不会把模型变回 float64
            live_dtype = getattr(live_var.value, "dtype", None)
            live_var.value = value.astype(live_dtype) if live_dtype is not None else value.copy()
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
    score_value = _read_checkpoint_score(metadata)
    if optimizer is not None:
        print(f"--- 已成功加载断点，准备从 Epoch {epoch + 1} 继续训练 ---")
    return epoch, 0.0 if score_value is None else float(score_value)


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
                state[var] = xp.asarray(data[entry["array_key"]], dtype=var.value.dtype).copy()
        setattr(optimizer, state_name, state)


# ---- TrainingState: transactional, checksummed snapshots for exact resume and rollback -----------------

TRAINING_STATE_VERSION = 1
_STATE_DIR = re.compile(r"^step-(\d{9})$")
_STATE_FILES = ("checkpoint.json", "checkpoint.npz")


class TrainingStateError(RuntimeError):
    """A TrainingState is missing, incomplete or does not match its manifest."""


def _sha256(path):
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _copy_checkpoint(source, target):
    import shutil

    source_json, source_npz = resolve_checkpoint_paths(source)
    json_path, npz_path = resolve_checkpoint_paths(target)
    metadata = json.loads(source_json.read_text(encoding="utf-8"))
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = npz_path.with_name(npz_path.name + ".tmp")
    shutil.copyfile(source_npz, tmp)
    os.replace(tmp, npz_path)
    metadata["array_file"] = npz_path.name
    _replace_json(json_path, metadata)


def save_training_state(directory, layers, optimizer, step, state, *, keep=3, best_filepath=None):
    """Write one TrainingState under ``directory/step-<step>/`` as a transaction.

    The weights, BN buffers and optimizer state go through :func:`save_checkpoint` (``state`` becomes its
    ``training`` metadata: RNG states, data cursor, early-stopping state, config fingerprint, ...). Everything
    is written into a staging directory first; ``manifest.json`` with the SHA-256 of every file is written
    last, the directory is renamed into place, and only then ``latest.json`` points at it. A crash at any
    point leaves either the previous states untouched or a staging directory that readers ignore.
    The newest ``keep`` states are kept (older ones are deleted) plus the step-0 state, the initial weights
    every rollback can fall back to; returns the state directory. ``best_filepath`` snapshots the historical
    best checkpoint as well, so a later improvement cannot change the best model of an older state.
    """
    import shutil
    import time
    import uuid

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = f"step-{int(step):09d}"
    staging = directory / f".staging-{name}-{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    try:
        epoch = int(state.get("epoch", 0)) if isinstance(state, dict) else 0
        save_checkpoint(layers, optimizer, epoch, filepath=str(staging / "checkpoint"), compress=False,
                        training={**state, "state_version": TRAINING_STATE_VERSION, "step": int(step)})
        names = list(_STATE_FILES)
        if best_filepath is not None:
            _copy_checkpoint(best_filepath, staging / "best")
            names.extend(("best.json", "best.npz"))
        files = {f: {"sha256": _sha256(staging / f), "bytes": (staging / f).stat().st_size} for f in names}
        _replace_json(staging / "manifest.json", {"format": "MyFlows TrainingState", "version": TRAINING_STATE_VERSION,
                                                  "step": int(step), "created_unix_s": time.time(), "files": files})
        final = directory / name
        if final.exists():
            shutil.rmtree(final)
        os.replace(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    _replace_json(directory / "latest.json", {"state": name, "step": int(step)})
    states = [s for s in list_training_states(directory, verify=False) if s["step"] != 0]
    for old in states[:-keep] if keep and len(states) > keep else []:
        shutil.rmtree(old["path"], ignore_errors=True)
    return final


def verify_training_state(path):
    """Check a TrainingState against its manifest; returns the manifest or raises TrainingStateError."""
    path = Path(path)
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        raise TrainingStateError(f"{path}: 没有 manifest（未完成的写入）")
    try:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, ValueError) as exc:
        raise TrainingStateError(f"{path}: 无法读取 manifest: {exc}") from exc
    if not isinstance(manifest, dict) or manifest.get("format") != "MyFlows TrainingState" \
            or manifest.get("version") != TRAINING_STATE_VERSION:
        raise TrainingStateError(f"{path}: 不支持的 manifest 格式或版本")
    files = manifest.get("files")
    allowed = {*_STATE_FILES, "best.json", "best.npz"}
    if not isinstance(files, dict) or not set(_STATE_FILES).issubset(files) or set(files) - allowed \
            or (("best.json" in files) != ("best.npz" in files)):
        raise TrainingStateError(f"{path}: manifest 文件清单不完整或无效")
    for name, expected in files.items():
        file = path / name
        if not isinstance(expected, dict) or not isinstance(expected.get("sha256"), str):
            raise TrainingStateError(f"{path}: {name} 的校验记录无效")
        if not file.is_file():
            raise TrainingStateError(f"{path}: 缺少 {name}")
        if _sha256(file) != expected.get("sha256"):
            raise TrainingStateError(f"{path}: {name} 的校验和与 manifest 不符")
    return manifest


def list_training_states(directory, *, verify=True):
    """Complete TrainingStates under ``directory``, oldest first: ``[{path, step, state}]``.

    ``state`` is the training metadata (read from ``checkpoint.json``); with ``verify`` the checksums are
    checked and broken states are skipped.
    """
    directory = Path(directory)
    out = []
    if not directory.is_dir():
        return out
    for entry in sorted(directory.iterdir()):
        m = _STATE_DIR.match(entry.name)
        if not m or not entry.is_dir() or not (entry / "manifest.json").exists():
            continue
        try:
            if verify:
                verify_training_state(entry)
            meta = read_checkpoint_metadata(entry / "checkpoint")
        except (TrainingStateError, OSError, ValueError):
            continue
        if meta is not None:
            out.append({"path": entry, "step": int(m.group(1)), "state": meta.get("training") or {}})
    return sorted(out, key=lambda s: s["step"])


def load_training_state(path, layers, optimizer, *, best_filepath=None):
    """Verify, then restore weights, buffers and optimizer state; returns the training metadata."""
    path = Path(path)
    manifest = verify_training_state(path)
    meta = read_checkpoint_metadata(path / "checkpoint") or {}
    state = meta.get("training") or {}
    if state.get("state_version") != TRAINING_STATE_VERSION:
        raise TrainingStateError(f"{path}: 不支持的 TrainingState 版本 {state.get('state_version')}")
    if best_filepath is not None and state.get("best_loss") is not None:
        if not {"best.json", "best.npz"}.issubset(manifest["files"]):
            raise TrainingStateError(f"{path}: TrainingState 未保存历史最佳模型，无法恢复最佳权重")
    load_checkpoint(layers, optimizer, filepath=str(path / "checkpoint"))
    if best_filepath is not None and state.get("best_loss") is not None:
        _copy_checkpoint(path / "best", best_filepath)
    return state
