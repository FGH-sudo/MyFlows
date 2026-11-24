"""
计算图构建阶段的可选优化（不改变动态图语义：仍按前向即时构图）。

当前支持：
- 常量折叠：将「两个 constant Variable 的 Add」替换为单个常量 Variable，减少前向/反向节点数。
- 统计信息：便于观察折叠次数。

说明：
- Conv2D + bias 的融合在 layers / Conv2D_Op 内完成，不经过本模块。
- 更激进的「算子融合」（如 Conv+ReLU）需要新增融合算子与模式匹配，可在此模块扩展。
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .device import xp
from .node import Node, Variable


def topological_order(target_node: Node) -> list[Node]:
    """Visit parents before consumers, preserving parent order without recursive calls."""
    order = []
    complete = set()
    active = {target_node}
    stack = [(target_node, iter(target_node.parents))]
    while stack:
        node, parents = stack[-1]
        parent = next(parents, None)
        if parent is None:
            stack.pop()
            active.remove(node)
            complete.add(node)
            order.append(node)
        elif parent not in complete:
            if parent in active:
                raise ValueError(f"cycle in computation graph at {parent.name!r}")
            active.add(parent)
            stack.append((parent, iter(parent.parents)))
    return order


def _add_values(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """与 ``ops.basic.Add.forward`` 一致的广播加法，用于折叠时常量求值。"""
    if a.ndim == 4 and b.ndim == 1:
        b_broadcast_shape = (1, b.shape[0], 1, 1)
        return a + b.reshape(b_broadcast_shape)
    return a + b


def replace_node(old: Node, new: Node) -> None:
    """将图中所有引用 ``old`` 的父链接替换为 ``new``，并维护 ``children``。"""
    for p in list(old.parents):
        if old in p.children:
            p.children.remove(old)
    for child in list(old.children):
        for i, p in enumerate(child.parents):
            if p is old:
                child.parents[i] = new
        if child not in new.children:
            new.children.append(child)
    old.children.clear()


def fold_constant_adds(target_node: Node) -> tuple[Node, int]:
    """
    将 ``Add(常量 Variable, 常量 Variable)`` 折叠为单个 ``Variable(constant=True)``。

    仅处理两父节点均为带 ``constant=True`` 的 ``Variable`` 的 ``Add``；其它图结构不变。
    按拓扑顺序替换，后续 Add 可以直接使用前面的折叠结果。

    若根节点被替换，返回新的根节点引用，避免调用方仍持有已脱离图的旧节点。
    返回 ``(current_root, fold_count)``。
    """
    from ..ops.basic import Add

    current_root = target_node
    total_folded = 0
    for node in topological_order(target_node):
        if not isinstance(node, Add) or len(node.parents) != 2:
            continue
        a, b = node.parents
        if not isinstance(a, Variable) or not isinstance(b, Variable):
            continue
        if not a.constant or not b.constant:
            continue

        val = _add_values(xp.asarray(a.value), xp.asarray(b.value))
        new_v = Variable(val, trainable=False, name=f"{node.name}_folded", constant=True, requires_grad=False)
        if node is current_root:
            current_root = new_v
        replace_node(node, new_v)
        total_folded += 1
    return current_root, total_folded


def _replace_parent(node: Node, index: int, new_parent: Node) -> None:
    old = node.parents[index]
    node.parents[index] = new_parent
    if node in getattr(old, "children", []):
        old.children.remove(node)
    if node not in new_parent.children:
        new_parent.children.append(node)


def _fold_bn_weights(conv_op, bn_op) -> None:
    """将 eval 态 BN 参数折入 Conv 的拷贝 kernel/bias，不修改原模型权重。"""
    from .device import xp
    from ..ops.convolution import Conv2D_Op

    if not isinstance(conv_op, Conv2D_Op):
        raise TypeError("fold_bn_into_conv 仅支持 Conv2D_Op")
    kernel_var = conv_op.parents[1]
    gamma_var, beta_var = bn_op.parents[1], bn_op.parents[2]
    W = xp.asarray(kernel_var.value)
    gamma = xp.asarray(gamma_var.value)
    beta = xp.asarray(beta_var.value)
    mean = xp.asarray(bn_op.running_mean)
    var = xp.asarray(bn_op.running_var)
    eps = float(bn_op.eps)
    scale = gamma / xp.sqrt(var + eps)
    C_out = int(W.shape[0])
    if conv_op.bias is not None:
        bias_var = conv_op.parents[2]
        b = xp.asarray(bias_var.value)
    else:
        bias_var = None
        b = xp.zeros((C_out,), dtype=W.dtype)
    W_new = W * scale.reshape(C_out, 1, 1, 1)
    b_new = (b - mean) * scale + beta
    new_kernel = Variable(
        W_new.copy() if hasattr(W_new, "copy") else W_new,
        trainable=bool(getattr(kernel_var, "trainable", True)),
        name=f"{getattr(kernel_var, 'name', 'kernel')}_bn_folded",
    )
    _replace_parent(conv_op, 1, new_kernel)
    if bias_var is not None:
        new_bias = Variable(
            b_new.copy() if hasattr(b_new, "copy") else b_new,
            trainable=bool(getattr(bias_var, "trainable", True)),
            name=f"{getattr(bias_var, 'name', 'bias')}_bn_folded",
        )
        _replace_parent(conv_op, 2, new_bias)
        conv_op.bias = new_bias
    else:
        new_bias = Variable(b_new, trainable=True, name=f"{getattr(conv_op, 'name', 'conv')}_folded_bias")
        conv_op.parents.append(new_bias)
        conv_op.bias = new_bias
        new_bias.children.append(conv_op)


def _detach_if_orphan(node: Node) -> None:
    if node.children:
        return
    for p in list(node.parents):
        while node in getattr(p, "children", []):
            p.children.remove(node)


def fold_bn_into_conv(target_node: Node) -> tuple[Node, int]:
    """
    推理态：将紧邻的 ``Conv2D_Op -> BatchNorm2d_Op(training=False)`` 折叠为单个 Conv。

    把 BN 参数折入 Conv 权重的拷贝，不污染原模型 Variable；再用 ``replace_node`` 跳过 BN。
    """
    from ..ops.batchnorm import BatchNorm2d_Op
    from ..ops.convolution import Conv2D_Op

    folded = 0
    root = target_node
    nodes = topological_order(root)
    uses = Counter(p for n in nodes for p in n.parents)
    for node in nodes:
        if not isinstance(node, BatchNorm2d_Op):
            continue
        if bool(getattr(node, "training", True)):
            continue
        if not node.parents:
            continue
        conv = node.parents[0]
        # An activation cannot be moved across BN, and other consumers must keep
        # seeing the original convolution output.
        if type(conv) is not Conv2D_Op or uses[conv] != 1 or any(child is not node for child in conv.children):
            continue
        if len(conv.parents) < 2:
            continue
        _fold_bn_weights(conv, node)
        if node is root:
            root = conv
        replace_node(node, conv)
        folded += 1
    return root, folded


def apply_graph_optimizations(
    target_node: Node,
    *,
    mode: str = "train",
    fold_constants: bool = True,
    fuse_linear: bool = True,
    fuse_activations: bool = True,
    fold_bn: bool | None = None,
) -> tuple[Node, dict]:
    """
    对以 ``target_node`` 为输出的子图应用构建期优化。

    ``mode``: ``"train"`` 或 ``"inference"``。推理模式下默认启用 BN 折叠。

    返回 ``(root_node, stats)``。``root_node`` 可能与传入的 ``target_node`` 不同
    （例如根被常量折叠替换时）。
    """
    if fold_bn is None:
        fold_bn = str(mode).lower() == "inference"

    stats: dict = {
        "constant_folds": 0,
        "linear_fusions": 0,
        "conv_act_fusions": 0,
        "bn_folds": 0,
    }
    root = target_node
    if fold_constants:
        root, stats["constant_folds"] = fold_constant_adds(root)
    # Fold eval Conv->BN before Conv+Act fusion so Conv->BN->ReLU becomes Conv2D_ReLU_Op.
    if fold_bn:
        root, stats["bn_folds"] = fold_bn_into_conv(root)
    if fuse_linear:
        root, stats["linear_fusions"] = fuse_linear_ops(root)
    if fuse_activations:
        root, stats["conv_act_fusions"] = fuse_conv_activation_ops(root)
    return root, stats


def batchnorm_activation_groups(nodes):
    """Match private intermediates. Shared logical nodes and aliases remain separate graph operations."""
    from collections import Counter
    from ..ops.activation import ReLU
    from ..ops.basic import Add
    from ..ops.batchnorm import BatchNorm2d_Op, BatchNormActivation

    uses = Counter(p for n in nodes for p in n.parents)
    stats = Counter(id(n.running_mean) for n in nodes if type(n) is BatchNorm2d_Op)
    groups = {}
    for output in nodes:
        if type(output) is not ReLU:
            continue
        parent = output.parents[0]
        add = parent if type(parent) is Add else None
        if add is not None and uses[add] != 1:
            continue
        for bn in add.parents if add is not None else (parent,):
            if (type(bn) is not BatchNorm2d_Op or uses[bn] != 1 or stats[id(bn.running_mean)] != 1
                    or any(uses[p] != 1 for p in bn.parents[1:])):
                continue
            residual = None if add is None else next((p for p in add.parents if p is not bn), None)
            if residual in bn.parents:
                continue
            groups[output] = BatchNormActivation(bn, add, output, residual)
            break
    return groups


def fuse_linear_ops(target_node: Node) -> tuple[Node, int]:
    """将 Add(MatMul(x, W), b) 或 Add(b, MatMul(x, W)) 融合为 Linear(x, W, b)。"""
    from ..ops.basic import Add, MatMul, Linear

    fused = 0
    root = target_node
    nodes = topological_order(root)
    uses = Counter(p for n in nodes for p in n.parents)
    for node in nodes:
        if not isinstance(node, Add) or len(node.parents) != 2:
            continue
        left, right = node.parents

        mm = None
        bias = None
        if isinstance(left, MatMul):
            mm, bias = left, right
        elif isinstance(right, MatMul):
            mm, bias = right, left
        else:
            continue

        if len(mm.parents) != 2 or uses[mm] != 1 or any(child is not node for child in mm.children):
            continue
        x, w = mm.parents

        lin = Linear(x, w, bias, name=f"{node.name}_fused", bias_first=mm is right)
        if node is root:
            root = lin
        replace_node(node, lin)
        _detach_if_orphan(mm)
        fused += 1

    return root, fused


def fuse_conv_activation_ops(target_node: Node) -> tuple[Node, int]:
    """将 {ReLU, LeakyReLU}(Conv/Deconv) 融合为对应融合算子。"""
    from ..ops.activation import ReLU, LeakyReLU
    from ..ops.convolution import (
        Conv2D_Op,
        Conv2D_ReLU_Op,
        Conv2D_LeakyReLU_Op,
        ConvTranspose2D_Op,
        ConvTranspose2D_ReLU_Op,
        ConvTranspose2D_LeakyReLU_Op,
    )

    fused = 0
    root = target_node
    nodes = topological_order(root)
    uses = Counter(p for n in nodes for p in n.parents)
    for node in nodes:
        if len(node.parents) != 1:
            continue
        parent = node.parents[0]
        is_relu = isinstance(node, ReLU)
        is_leaky = isinstance(node, LeakyReLU)
        if not (is_relu or is_leaky):
            continue
        if uses[parent] != 1 or any(child is not node for child in parent.children):
            continue

        # Conv2D(+bias) + activation
        if isinstance(parent, Conv2D_Op) and not isinstance(parent, (Conv2D_ReLU_Op, Conv2D_LeakyReLU_Op)):
            x = parent.parents[0]
            kernel = parent.parents[1]
            bias = parent.parents[2] if len(parent.parents) == 3 else None
            if is_relu:
                fused_node = Conv2D_ReLU_Op(
                    x,
                    kernel,
                    recompute_im2col=parent.recompute_im2col,
                    stride=parent.stride,
                    padding=parent.padding,
                    groups=parent.groups,
                    dilation=parent.dilation,
                    bias=bias,
                    backend=parent.backend,
                )
            else:
                fused_node = Conv2D_LeakyReLU_Op(
                    x,
                    kernel,
                    recompute_im2col=parent.recompute_im2col,
                    stride=parent.stride,
                    padding=parent.padding,
                    groups=parent.groups,
                    dilation=parent.dilation,
                    bias=bias,
                    backend=parent.backend,
                    alpha=getattr(node, "alpha", 0.01),
                )

        # ConvTranspose2D(+bias) + activation
        elif isinstance(parent, ConvTranspose2D_Op) and not isinstance(parent, (ConvTranspose2D_ReLU_Op, ConvTranspose2D_LeakyReLU_Op)):
            x = parent.parents[0]
            kernel = parent.parents[1]
            bias = parent.parents[2] if len(parent.parents) == 3 else None
            if is_relu:
                fused_node = ConvTranspose2D_ReLU_Op(
                    x,
                    kernel,
                    stride=parent.stride,
                    padding=parent.padding,
                    output_padding=parent.output_padding,
                    groups=parent.groups,
                    dilation=parent.dilation,
                    bias=bias,
                )
            else:
                fused_node = ConvTranspose2D_LeakyReLU_Op(
                    x,
                    kernel,
                    stride=parent.stride,
                    padding=parent.padding,
                    output_padding=parent.output_padding,
                    groups=parent.groups,
                    dilation=parent.dilation,
                    bias=bias,
                    alpha=getattr(node, "alpha", 0.01),
                )
        else:
            continue

        if node is root:
            root = fused_node
        replace_node(node, fused_node)
        _detach_if_orphan(parent)
        fused += 1

    return root, fused
