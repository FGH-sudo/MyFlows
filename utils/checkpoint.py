import json
import math
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ..core.device import asnumpy, xp


_CHECKPOINT_VERSION = 1
_OPTIMIZER_STATES = ("v", "m", "s", "acc_gradient")
_OPTIMIZER_SCALARS = ("learning_rate", "momentum", "beta", "beta_1", "beta_2", "eps", "t", "acc_no")


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
    if hasattr(layers, "params") or hasattr(layers, "sub_layers") or hasattr(layers, "_modules"):
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


def _iter_layer_buffers(layer):
    seen = set()
    for attr in ("running_mean", "running_var"):
        value = getattr(layer, attr, None)
        if value is not None and hasattr(value, "shape"):
            seen.add(attr)
            yield attr, value
    for name, buf in getattr(layer, "_buffers", {}).items():
        if buf is None or not getattr(buf, "persistent", True) or name in seen:
            continue
        arr = getattr(buf, "data", buf)
        if arr is not None and hasattr(arr, "shape"):
            seen.add(name)
            yield name, arr


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
        for attr, _arr in _iter_layer_buffers(layer):
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

        for attr, value in _iter_layer_buffers(layer):
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

    for state_name in _OPTIMIZER_STATES:
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
    for attr in _OPTIMIZER_SCALARS:
        if not hasattr(optimizer, attr):
            continue
        value = _json_scalar(getattr(optimizer, attr))
        if value is not None:
            scalars[attr] = value

    return arrays, {
        "class": optimizer.__class__.__name__,
        "scalars": scalars,
        "loss_scaler": optimizer.loss_scaler.state_dict() if getattr(optimizer, "loss_scaler", None) else None,
        "states": state_meta,
        "param_groups": optimizer.groups.serialize(param_key_by_id, optimizer.learning_rate) if hasattr(optimizer, "groups") else [],
        "parameter_steps": {param_key_by_id[id(node)]: int(step) for node, step in getattr(optimizer, "parameter_steps", {}).items()
                            if id(node) in param_key_by_id},
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
    JSON 与 NPZ 分别通过临时文件替换。需要成组发布时使用 ``save_training_state``。
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

    # Publish the arrays before the metadata that references them.
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

    epoch = int(metadata.get("epoch", -1))
    score_value = _read_checkpoint_score(metadata)
    score_value = 0.0 if score_value is None else float(score_value)
    if not math.isfinite(score_value):
        raise ValueError("checkpoint score must be finite")
    saved_key_to_var = {saved["key"]: _param_by_live_record(layers, live)
                        for saved, live in zip(saved_params, live_params)}
    model_keys = [s["key"] for s in (*saved_params, *saved_buffers)]
    if len(set(model_keys)) != len(model_keys):
        raise ValueError("duplicate checkpoint model keys")
    opt_meta = metadata.get("optimizer", {})
    if optimizer is not None:
        from ..train.param_groups import ParameterGroups

        if opt_meta.get("class") not in (None, optimizer.__class__.__name__):
            raise ValueError("checkpoint optimizer class does not match the live optimizer")
        ParameterGroups.restore(opt_meta.get("param_groups", []), saved_key_to_var)
        if opt_meta.get("loss_scaler") is not None:
            from ..train.loss_scaling import DynamicLossScaler
            DynamicLossScaler.from_state(opt_meta["loss_scaler"])
        for key, step in opt_meta.get("parameter_steps", {}).items():
            if key not in saved_key_to_var or type(step) is not int or step < 0:
                raise ValueError("invalid optimizer per-parameter step")
        for attr, value in opt_meta.get("scalars", {}).items():
            if attr not in _OPTIMIZER_SCALARS or not hasattr(optimizer, attr):
                raise ValueError(f"invalid optimizer scalar: {attr}")
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid optimizer scalar: {attr}")
            if attr in ("t", "acc_no") and type(value) is not int:
                raise ValueError(f"optimizer {attr} must be a nonnegative integer")
            if attr in ("momentum", "beta", "beta_1", "beta_2") and value >= 1:
                raise ValueError(f"optimizer {attr} must be less than one")

    # Validate every referenced array before replacing any live weight or optimizer state.
    # Host staging avoids an extra full model allocation on the GPU during rollbacks.
    live_layer_list = collect_layers(layers)
    arrays = {}
    with np.load(npz_path, allow_pickle=False) as archive:
        def stage(key, shape, dtype):
            if key in arrays:
                raise ValueError(f"duplicate checkpoint array reference: {key}")
            if key not in archive:
                raise ValueError(f"checkpoint array missing: {key}")
            value = np.asarray(archive[key])
            if value.shape != tuple(shape) or value.dtype.kind not in "biuf":
                raise ValueError(f"invalid checkpoint shape or dtype: {key}")
            with np.errstate(over="ignore", invalid="ignore"):
                value = value.astype(dtype, copy=False)
            if not np.isfinite(value).all():
                raise ValueError(f"non-finite checkpoint array: {key}")
            arrays[key] = value

        for saved, live in zip(saved_params, live_params):
            var = saved_key_to_var[saved["key"]]
            stage(saved["key"], live["shape"], var.value.dtype)
        for saved, live in zip(saved_buffers, live_buffers):
            current = getattr(live_layer_list[live["layer_index"]], live["name"])
            if hasattr(current, "data") and not hasattr(current, "__array__"):
                current = current.data
            stage(saved["key"], current.shape, current.dtype)
        if optimizer is not None:
            state_keys = {}
            for state_name, entries in opt_meta.get("states", {}).items():
                if state_name not in _OPTIMIZER_STATES or not hasattr(optimizer, state_name):
                    raise ValueError(f"invalid optimizer state: {state_name}")
                seen = set()
                for entry in entries:
                    key = entry["param_key"]
                    if key not in saved_key_to_var or key in seen:
                        raise ValueError(f"unknown or duplicate optimizer parameter: {key}")
                    seen.add(key)
                    var = saved_key_to_var[key]
                    stage(entry["array_key"], var.value.shape, var.value.dtype)
                state_keys[state_name] = seen
            if optimizer.__class__.__name__ == "Adam" and state_keys.get("v", set()) != state_keys.get("s", set()):
                raise ValueError("Adam first and second moments cover different parameters")

    for key, var in saved_key_to_var.items():
        var.value = arrays[key]
    for saved, live in zip(saved_buffers, live_buffers):
        current = getattr(live_layer_list[live["layer_index"]], live["name"])
        if hasattr(current, "data") and not hasattr(current, "__array__"):
            current = current.data
        current[...] = xp.asarray(arrays[saved["key"]], dtype=current.dtype)
    _restore_optimizer(optimizer, opt_meta, arrays, saved_key_to_var)

    if optimizer is not None:
        print(f"--- 已成功加载断点，准备从 Epoch {epoch + 1} 继续训练 ---")
    return epoch, score_value


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
    if optimizer_meta.get("loss_scaler") is not None:
        from ..train.loss_scaling import DynamicLossScaler
        optimizer.loss_scaler = DynamicLossScaler.from_state(optimizer_meta["loss_scaler"])

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
    if hasattr(optimizer, "groups"):
        from ..train.param_groups import ParameterGroups

        optimizer.groups = ParameterGroups.restore(optimizer_meta.get("param_groups", []), saved_key_to_var)
    if hasattr(optimizer, "parameter_steps"):
        optimizer.parameter_steps = {saved_key_to_var[key]: int(step) for key, step in optimizer_meta.get("parameter_steps", {}).items()
                                     if key in saved_key_to_var}


# ---- TrainingState: transactional, checksummed snapshots for exact resume and rollback -----------------

TRAINING_STATE_VERSION = 1
_STATE_DIR = re.compile(r"step-(\d{9,})(?:-(\d+))?")
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


def _state_directories(directory):
    if not directory.is_dir():
        return []
    entries = []
    for path in directory.iterdir():
        match = _STATE_DIR.fullmatch(path.name)
        if match and path.is_dir():
            step, revision = match.groups()
            entries.append((int(step), int(revision or 0), path))
    return sorted(entries, reverse=True)


@contextmanager
def _state_publication_lock(directory, timeout_s=30.0):
    deadline = time.monotonic() + float(timeout_s)
    with (directory / ".publish.lock").open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            while True:
                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"timed out waiting for {directory / '.publish.lock'}") from None
                    time.sleep(0.05)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def save_training_state(directory, layers, optimizer, step, state, *, keep=3, best_filepath=None):
    """Publish an immutable TrainingState under ``directory/step-<step>/`` (optionally with a revision suffix).

    The weights, BN buffers and optimizer state go through :func:`save_checkpoint` (``state`` becomes its
    ``training`` metadata: RNG states, data cursor, early-stopping state, config fingerprint, ...). Everything
    is written into a staging directory first; ``manifest.json`` with the SHA-256 of every file is written
    last, the directory is renamed into place, and only then ``latest.json`` points at it. A crash at any
    point leaves previous states available. Repeating a step publishes a new revision alongside its predecessors.
    The newest ``keep`` steps, the published step and step 0 are retained. Returns the state directory.
    ``best_filepath`` snapshots the historical
    best checkpoint as well, so a later improvement cannot change the best model of an older state.
    """
    import shutil
    import uuid

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    step = int(step)
    if step < 0:
        raise ValueError("TrainingState step must be nonnegative")
    name = f"step-{step:09d}"
    staging = directory / f".staging-{name}-{uuid.uuid4().hex[:8]}"
    staging.mkdir()
    try:
        epoch = int(state.get("epoch", 0)) if isinstance(state, dict) else 0
        save_checkpoint(layers, optimizer, epoch, filepath=str(staging / "checkpoint"), compress=False,
                        training={**state, "state_version": TRAINING_STATE_VERSION, "step": step})
        names = list(_STATE_FILES)
        if best_filepath is not None:
            _copy_checkpoint(best_filepath, staging / "best")
            names.extend(("best.json", "best.npz"))
        files = {f: {"sha256": _sha256(staging / f), "bytes": (staging / f).stat().st_size} for f in names}
        _replace_json(staging / "manifest.json", {"format": "MyFlows TrainingState", "version": TRAINING_STATE_VERSION,
                                                  "step": step, "created_unix_s": time.time(), "files": files})
        with _state_publication_lock(directory):
            entries = _state_directories(directory)
            revision = 1 + max((version for saved_step, version, _ in entries if saved_step == step), default=-1)
            final = directory / (f"{name}-{revision}" if revision else name)
            os.replace(staging, final)
            _replace_json(directory / "latest.json", {"state": final.name, "step": step})
            current = _state_directories(directory)
            if keep:
                eligible = sorted({s for s, _, p in current if s != 0 and s <= step and (p / "manifest.json").is_file()})
                retained = {0, step, *eligible[-keep:]}
            else:
                retained = None
            kept_revisions = {}
            for old_step, _, path in current:
                if not (path / "manifest.json").is_file():
                    continue
                seen = kept_revisions.get(old_step, 0)
                if (retained is None or old_step in retained) and seen < 2:
                    kept_revisions[old_step] = seen + 1
                else:
                    shutil.rmtree(path, ignore_errors=True)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
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
    """Newest complete TrainingState per step, oldest step first: ``[{path, step, state}]``.

    ``state`` is the training metadata (read from ``checkpoint.json``); with ``verify`` the checksums are
    checked and broken states are skipped.
    """
    out = {}
    for step, _, entry in _state_directories(Path(directory)):
        if step in out or not (entry / "manifest.json").is_file():
            continue
        try:
            if verify:
                verify_training_state(entry)
            meta = read_checkpoint_metadata(entry / "checkpoint")
        except (TrainingStateError, OSError, ValueError):
            continue
        if meta is not None:
            out[step] = {"path": entry, "step": step, "state": meta.get("training") or {}}
    return list(reversed(out.values()))


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
