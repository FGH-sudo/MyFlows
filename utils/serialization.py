import json
import re
from pathlib import Path

import numpy as np

from ..core.device import asnumpy, xp
from ..core.node import Variable


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


def _collect_layers(layers):
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


def _resolve_checkpoint_paths(filepath):
    path = Path(filepath)
    if path.suffix == ".json":
        return path, path.with_suffix(".npz")
    if path.suffix == ".npz":
        return path.with_suffix(".json"), path
    return path.with_suffix(".json"), path.with_suffix(".npz")


def _collect_model_arrays(layers):
    layer_list = _collect_layers(layers)
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
    """以 JSON + NPZ 格式保存模型权重、BN buffer 与优化器状态。

    JSON 保存结构化元信息，NPZ 保存 NumPy 数组。传入 ``checkpoint`` 会生成
    ``checkpoint.json`` 与 ``checkpoint.npz``；传入 ``*.json`` 或 ``*.npz`` 时会
    自动推导另一侧文件名。
    """
    json_path, npz_path = _resolve_checkpoint_paths(filepath)
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
    json_path, npz_path = _resolve_checkpoint_paths(filepath)
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
            live_layer_list = _collect_layers(layers)
            for saved, live in zip(saved_buffers, live_buffers):
                value = np.asarray(data[saved["key"]])
                layer = live_layer_list[live["layer_index"]]
                current = getattr(layer, live["name"])
                if current.shape != value.shape:
                    raise ValueError(
                        f"buffer 形状不匹配 {saved['key']}: checkpoint={value.shape}, model={current.shape}"
                    )
                # checkpoint 为 NumPy；运行期 buffer 可能在 CuPy 上，须先转到当前设备
                arr = xp.asarray(value, dtype=current.dtype)
                current[...] = arr

        _restore_optimizer(optimizer, metadata.get("optimizer", {}), data, saved_key_to_var)

    epoch = int(metadata.get("epoch", -1))
    best_acc = metadata.get("best_acc", 0.0)
    if optimizer is not None:
        print(f"--- 已成功加载断点，准备从 Epoch {epoch + 1} 继续训练 ---")
    return epoch, 0.0 if best_acc is None else float(best_acc)


def _param_by_live_record(layers, live_record):
    layer = _collect_layers(layers)[live_record["layer_index"]]
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


def export_onnx(graph, filepath, input_nodes=None, output_names=None, opset=13):
    """将已构建的 ``Graph`` 导出为 ONNX。

    该功能依赖可选包 ``onnx``。导出前会确保图已经前向执行过一次，以便写入
    shape 信息；当前支持常见推断算子：Variable、Linear、MatMul、Add、
    ReLU/LeakyReLU/Sigmoid/Tanh/Softmax、Conv/ConvTranspose、MaxPool、
    Flatten、BatchNorm2d、GlobalAvgPool2d。
    """
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:
        raise ImportError("导出 ONNX 需要先安装可选依赖: pip install onnx") from exc

    if any(getattr(node, "value", None) is None for node in graph.sorted_nodes):
        graph.forward()

    input_nodes = set(input_nodes or [])
    tensor_names = {}
    nodes = []
    initializers = []
    inputs = []
    outputs = []
    value_infos = []

    def tensor_name(node):
        if node not in tensor_names:
            tensor_names[node] = f"{_safe_name(getattr(node, 'name', node.__class__.__name__))}_{len(tensor_names)}"
        return tensor_names[node]

    def _onnx_numpy(value):
        return np.asarray(asnumpy(value))

    def tensor_dtype(value):
        value = _onnx_numpy(value)
        if value.dtype == np.float32:
            return TensorProto.FLOAT
        if value.dtype == np.float64:
            return TensorProto.DOUBLE
        if np.issubdtype(value.dtype, np.integer):
            return TensorProto.INT64
        return TensorProto.FLOAT

    def value_info(name, value):
        value = _onnx_numpy(value)
        if np.issubdtype(value.dtype, np.floating):
            value = value.astype(np.float32, copy=False)
        return helper.make_tensor_value_info(name, tensor_dtype(value), list(value.shape))

    def add_initializer(name, value):
        arr = _onnx_numpy(value)
        if np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32, copy=False)
        initializers.append(numpy_helper.from_array(arr, name=name))

    for node in graph.sorted_nodes:
        name = tensor_name(node)
        if isinstance(node, Variable):
            if node in input_nodes or not getattr(node, "trainable", False):
                inputs.append(value_info(name, node.value))
            else:
                add_initializer(name, node.value)
            continue

        op_type = node.__class__.__name__
        parent_names = [tensor_name(parent) for parent in node.parents]
        out_name = name
        attrs = {}

        if op_type == "Linear":
            nodes.append(helper.make_node("Gemm", parent_names, [out_name], name=name))
        elif op_type == "MatMul":
            nodes.append(helper.make_node("MatMul", parent_names, [out_name], name=name))
        elif op_type == "Add":
            nodes.append(helper.make_node("Add", parent_names, [out_name], name=name))
        elif op_type == "ReLU":
            nodes.append(helper.make_node("Relu", parent_names, [out_name], name=name))
        elif op_type == "LeakyReLU":
            attrs["alpha"] = float(getattr(node, "alpha", 0.01))
            nodes.append(helper.make_node("LeakyRelu", parent_names, [out_name], name=name, **attrs))
        elif op_type == "Logistic":
            nodes.append(helper.make_node("Sigmoid", parent_names, [out_name], name=name))
        elif op_type == "Tanh":
            nodes.append(helper.make_node("Tanh", parent_names, [out_name], name=name))
        elif op_type == "Softmax":
            nodes.append(helper.make_node("Softmax", parent_names, [out_name], name=name, axis=-1))
        elif op_type in ("Conv2D_Op", "Conv2D_ReLU_Op", "Conv2D_LeakyReLU_Op"):
            attrs = {
                "strides": list(node.stride),
                "pads": [node.padding[0], node.padding[1], node.padding[0], node.padding[1]],
                "dilations": list(node.dilation),
                "group": int(node.groups),
            }
            conv_out = out_name if op_type == "Conv2D_Op" else f"{out_name}_conv"
            nodes.append(helper.make_node("Conv", parent_names, [conv_out], name=f"{name}_conv", **attrs))
            _append_fused_activation(helper, nodes, node, conv_out, out_name, name)
        elif op_type in ("ConvTranspose2D_Op", "ConvTranspose2D_ReLU_Op", "ConvTranspose2D_LeakyReLU_Op"):
            attrs = {
                "strides": list(node.stride),
                "pads": [node.padding[0], node.padding[1], node.padding[0], node.padding[1]],
                "dilations": list(node.dilation),
                "output_padding": list(node.output_padding),
                "group": int(node.groups),
            }
            conv_out = out_name if op_type == "ConvTranspose2D_Op" else f"{out_name}_convtranspose"
            nodes.append(helper.make_node("ConvTranspose", parent_names, [conv_out], name=f"{name}_convtranspose", **attrs))
            _append_fused_activation(helper, nodes, node, conv_out, out_name, name)
        elif op_type == "MaxPool2d_Op":
            nodes.append(helper.make_node(
                "MaxPool",
                parent_names,
                [out_name],
                name=name,
                kernel_shape=list(node.kernel_size),
                strides=list(node.stride),
            ))
        elif op_type == "Flatten_Op":
            nodes.append(helper.make_node("Flatten", parent_names, [out_name], name=name, axis=1))
        elif op_type == "BatchNorm2d_Op":
            mean_name = f"{name}_running_mean"
            var_name = f"{name}_running_var"
            add_initializer(mean_name, node.running_mean)
            add_initializer(var_name, node.running_var)
            nodes.append(helper.make_node(
                "BatchNormalization",
                [parent_names[0], parent_names[1], parent_names[2], mean_name, var_name],
                [out_name],
                name=name,
                epsilon=float(node.eps),
                momentum=float(1.0 - node.momentum),
            ))
        elif op_type == "GlobalAvgPool2d_Op":
            pooled = f"{out_name}_pooled"
            nodes.append(helper.make_node("GlobalAveragePool", parent_names, [pooled], name=f"{name}_gap"))
            nodes.append(helper.make_node("Flatten", [pooled], [out_name], name=f"{name}_flatten", axis=1))
        else:
            raise NotImplementedError(f"暂不支持导出 ONNX 算子: {op_type}")

        if getattr(node, "value", None) is not None:
            value_infos.append(value_info(out_name, node.value))

    target_name = tensor_name(graph.target_node)
    final_output_names = output_names or [target_name]
    if output_names:
        for index, output_name in enumerate(final_output_names):
            nodes.append(helper.make_node(
                "Identity",
                [target_name],
                [output_name],
                name=f"output_identity_{index}",
            ))
    for output_name in final_output_names:
        outputs.append(value_info(output_name, graph.target_node.value))

    model_graph = helper.make_graph(
        nodes,
        "MyFlowsGraph",
        inputs,
        outputs,
        initializer=initializers,
        value_info=value_infos,
    )
    model = helper.make_model(model_graph, opset_imports=[helper.make_operatorsetid("", int(opset))])
    onnx.checker.check_model(model)
    onnx.save(model, filepath)
    return str(filepath)


def _append_fused_activation(helper, nodes, node, conv_out, out_name, base_name):
    op_type = node.__class__.__name__
    if op_type.endswith("_ReLU_Op"):
        nodes.append(helper.make_node("Relu", [conv_out], [out_name], name=f"{base_name}_relu"))
    elif op_type.endswith("_LeakyReLU_Op"):
        nodes.append(helper.make_node(
            "LeakyRelu",
            [conv_out],
            [out_name],
            name=f"{base_name}_leaky_relu",
            alpha=float(getattr(node, "alpha", 0.01)),
        ))
