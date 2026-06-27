import re

import numpy as np

from ..core.device import asnumpy
from ..core.node import Variable


def _safe_name(name):
    text = "unnamed" if name is None else str(name)
    return re.sub(r"[^0-9a-zA-Z_]+", "_", text).strip("_") or "unnamed"


def export_onnx(graph, filepath, input_nodes=None, output_names=None, opset=13):
    """将已构建的 ``Graph`` 导出为 ONNX。"""
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
