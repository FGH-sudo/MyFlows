# -*- coding: utf-8 -*-
"""MyFlows - 简单的神经网络框架，用于实现ADALINE模型"""

# 导入core模块
import importlib

# --- 核心组件 (core) ---
# 导入 Node 和 Variable，并暴露在顶层命名空间 (ms.Node)
from .core.node import Node, Variable
# 导入 Graph，并暴露在顶层命名空间 (ms.Graph)
from .core.graph import Graph

# --- 连接层 (layers) ---
# 导入连接网络层
from .layers.layer import Layer, Dense, Conv2D, MaxPool2d, Flatten, save_checkpoint, load_checkpoint

# --- 操作符 (ops) ---
# 导入基本操作 
from .ops.basic import Add, MatMul
# 导入损失函数 
from .ops.loss import PerceptionLoss, LogLoss, CrossEntropy
# 导入激活函数
from .ops.activation import Logistic, Softmax, Tanh, ReLU, LeakyReLU
# 导入卷积、池化、展平算子
from .ops.convolution import Conv2D_Op, MaxPool2d_Op, Flatten_Op

# --- 训练和优化 (train) ---
# 导入优化器 (ms.Optimizer, ms.MBGD, ms.Momentum, ms.AdaGrad, ms.RMSProp, ms.Adam)
from .train.opt import Optimizer, MBGD, Momentum, AdaGrad, RMSProp, Adam

# --- 工具和数据 (utils) ---
# 导入数据生成器 (ms.DataGenerator)
from .utils.data import DataGenerator, StandardScaler, augment_data, gradient_check, plot_computational_graph


# 定义当你使用 `from MyFlows import *` 时，会暴露哪些名字
# 这也作为所有暴露给用户的 API 列表
__all__ = [
    # Core
    'Node', 'Variable', 'Graph',
    # Layers
    'Layer', 'Dense', 'Conv2D', 'MaxPool2d', 'Flatten', 'save_checkpoint', 'load_checkpoint',
    # Ops
    'Add', 'MatMul', 'PerceptionLoss', 'LogLoss', 'CrossEntropy', 'Logistic', 'Softmax', 'Tanh', 'ReLU', 'LeakyReLU', 'Conv2D_Op', 'MaxPool2d_Op', 'Flatten_Op', 
    # Train
    'Optimizer', 'MBGD', 'Momentum', 'AdaGrad', 'RMSProp', 'Adam',
    # Utils
    'DataGenerator', 'StandardScaler', 'augment_data', 'gradient_check', 'plot_computational_graph',
]