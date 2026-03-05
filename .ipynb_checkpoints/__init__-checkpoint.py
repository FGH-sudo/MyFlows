# -*- coding: utf-8 -*-
"""MyFlows - 简单的神经网络框架，用于实现ADALINE模型"""

# 导入core模块
import importlib

# 确保core模块可以被正确导入和访问
try:
    from . import core
    # 将core模块中的Variable类直接导入到ms命名空间
    from .core.core import Variable
    # 更新core模块，使其包含Variable类
    core.Variable = Variable
except ImportError as e:
    print(f"导入core模块时出错: {e}")

try:
    from . import ops
    # 确保ops模块可以通过ms.ops直接访问
    # 导入ops包中的所有类
    from .ops.ops import Add, MatMul, Step
    from .ops.loss import loss
    # 将这些类添加到ops模块中
    ops.Add = Add
    ops.MatMul = MatMul
    ops.Step = Step
    ops.loss = loss
except ImportError as e:
    print(f"导入ops模块时出错: {e}")

# 确保这些模块可以通过ms直接访问
__all__ = ['core', 'ops']