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

import numpy as np

from .node import Node, Variable


def _topo_sort(target_node: Node):
    visited, order = set(), []

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        for p in getattr(node, "parents", []):
            dfs(p)
        order.append(node)

    dfs(target_node)
    return order


def _add_values(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """与 ``ops.basic.Add.forward`` 一致的广播加法，用于折叠时常量求值。"""
    if a.ndim == 4 and b.ndim == 1:
        b_broadcast_shape = (1, b.shape[0], 1, 1)
        return a + b.reshape(b_broadcast_shape)
    a_broadcast_shape = (1,) * (max(a.ndim, b.ndim) - a.ndim) + a.shape
    b_broadcast_shape = (1,) * (max(a.ndim, b.ndim) - b.ndim) + b.shape
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
    多轮直到无变化，以便折叠链式 Add。

    若根节点被替换，返回新的根节点引用，避免调用方仍持有已脱离图的旧节点。
    返回 ``(current_root, fold_count)``。
    """
    from ..ops.basic import Add

    current_root = target_node
    total_folded = 0
    while True:
        folded = 0
        sorted_nodes = _topo_sort(current_root)
        for node in list(sorted_nodes):
            if not isinstance(node, Add):
                continue
            if len(node.parents) != 2:
                continue
            a, b = node.parents
            if not isinstance(a, Variable) or not isinstance(b, Variable):
                continue
            if not getattr(a, "constant", False) or not getattr(b, "constant", False):
                continue

            val = _add_values(np.asarray(a.value), np.asarray(b.value))
            new_v = Variable(
                val,
                trainable=False,
                name=f"{node.name}_folded",
                constant=True,
            )
            if node is current_root:
                current_root = new_v
            replace_node(node, new_v)
            folded += 1
        total_folded += folded
        if folded == 0:
            break
    return current_root, total_folded


def _fold_bn_weights(conv_op, bn_op) -> None:
    """将 eval 态 BN 参数折入 Conv 的 kernel/bias Variable（就地修改）。"""
    from ..core.device import xp
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
        b = xp.zeros((C_out,), dtype=W.dtype)
    W_new = W * scale.reshape(C_out, 1, 1, 1)
    b_new = (b - mean) * scale + beta
    kernel_var.value = W_new
    if conv_op.bias is not None:
        bias_var.value = b_new
    else:
        from .node import Variable

        new_bias = Variable(b_new, trainable=True, name=f"{getattr(conv_op, 'name', 'conv')}_folded_bias")
        conv_op.parents.append(new_bias)
        conv_op.bias = new_bias


def fold_bn_into_conv(target_node: Node) -> tuple[Node, int]:
    """
    推理态：将紧邻的 ``Conv2D_Op -> BatchNorm2d_Op(training=False)`` 折叠为单个 Conv。

    修改 Conv 的 weight/bias Variable，并用 ``replace_node`` 跳过 BN。
    """
    from ..ops.batchnorm import BatchNorm2d_Op
    from ..ops.convolution import Conv2D_Op

    folded = 0
    root = target_node
    for node in list(_topo_sort(root)):
        if not isinstance(node, BatchNorm2d_Op):
            continue
        if bool(getattr(node, "training", True)):
            continue
        if not node.parents:
            continue
        conv = node.parents[0]
        if not isinstance(conv, Conv2D_Op):
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
    if fuse_linear:
        root, stats["linear_fusions"] = fuse_linear_ops(root)
    if fuse_activations:
        root, stats["conv_act_fusions"] = fuse_conv_activation_ops(root)
    if fold_bn:
        root, stats["bn_folds"] = fold_bn_into_conv(root)
    return root, stats


def fuse_linear_ops(target_node: Node) -> tuple[Node, int]:
    """将 Add(MatMul(x, W), b) 或 Add(b, MatMul(x, W)) 融合为 Linear(x, W, b)。"""
    from ..ops.basic import Add, MatMul, Linear

    fused = 0
    root = target_node
    sorted_nodes = _topo_sort(root)
    for node in list(sorted_nodes):
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

        if len(mm.parents) != 2:
            continue
        x, w = mm.parents

        # 构造融合节点并替换 Add（mm 可能成为孤儿节点，允许其自然被剪枝）
        lin = Linear(x, w, bias, name=f"{node.name}_fused")
        if node is root:
            root = lin
        replace_node(node, lin)
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
    sorted_nodes = _topo_sort(root)

    for node in list(sorted_nodes):
        if len(node.parents) != 1:
            continue
        parent = node.parents[0]
        is_relu = isinstance(node, ReLU)
        is_leaky = isinstance(node, LeakyReLU)
        if not (is_relu or is_leaky):
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
                    stride=parent.stride,
                    padding=parent.padding,
                    groups=parent.groups,
                    dilation=parent.dilation,
                    bias=bias,
                )
            else:
                fused_node = Conv2D_LeakyReLU_Op(
                    x,
                    kernel,
                    stride=parent.stride,
                    padding=parent.padding,
                    groups=parent.groups,
                    dilation=parent.dilation,
                    bias=bias,
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
        fused += 1

    return root, fused
