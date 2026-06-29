# -*- coding: utf-8 -*-
"""MyFlows - 简单的神经网络框架，用于实现ADALINE模型"""

# 导入core模块
import importlib

# --- 核心组件 (core) ---
# 导入 Node 和 Variable，并暴露在顶层命名空间 (ms.Node)
from .core.node import Node, Variable
# 导入 Tensor，并暴露在顶层命名空间 (ms.Tensor)
from .core.tensor import Tensor
# 导入 Graph，并暴露在顶层命名空间 (ms.Graph)
from .core.graph import Graph
from .core.graph_opt import apply_graph_optimizations
from .core.device import cuda_available, configure_cuda_dll_path, get_device, set_device, to_device, use_cuda

# --- 连接层 (layers) ---
# 导入连接网络层
from .layers.layer import (
    Layer,
    Dense,
    Conv2D,
    GroupedConv2D,
    DepthwiseConv2D,
    DepthwiseSeparableConv2D,
    DilatedConv2D,
    ConvTranspose2D,
    Dropout,
    MaxPool2d,
    Flatten,
)
from .layers.resnet import (
    BatchNorm2d,
    GlobalAvgPool2d,
    BasicBlock,
    ResNet18,
)
from .layers.vgg import VGG11, vgg_fc_input_dim
from .utils.serialization import save_checkpoint, load_checkpoint, export_onnx

# --- 操作符 (ops) ---
# 导入基本操作 
from .ops.basic import Add, MatMul
# 导入损失函数 
from .ops.loss import PerceptionLoss, LogLoss, CrossEntropy, MSELoss
# 导入激活函数
from .ops.activation import Logistic, Softmax, Tanh, ReLU, LeakyReLU
# 导入卷积、池化、展平算子
from .ops.convolution import (
    Conv2D_Op,
    ConvTranspose2D_Op,
    MaxPool2d_Op,
    Flatten_Op,
    effective_kernel_size,
    im2col,
    col2im,
)
from .ops.batchnorm import BatchNorm2d_Op, GlobalAvgPool2d_Op
from .ops.dropout import Dropout_Op

# --- 训练和优化 (train) ---
# 导入优化器 (ms.Optimizer, ms.MBGD, ms.Momentum, ms.AdaGrad, ms.RMSProp, ms.Adam)
from .train.opt import Optimizer, MBGD, Momentum, AdaGrad, RMSProp, Adam
from .train.regularization import RegularizationConfig, apply_regularization

# --- 工具 (utils) ---
from .utils.metrics import (
    DonkeyRegressionEvaluator,
    accuracy,
    angle_sign_accuracy,
    classification_report,
    confusion_matrix,
    mae,
    mse,
    multivariate_regression_metrics,
    precision_recall_f1,
    regression_metrics,
    rmse,
    r2_score,
)
from .utils.viz import TrainingHistory, plot_metric_series, plot_training_curves
from .utils.tensorboard_logger import TensorBoardLogger, has_tensorboard_events
from .utils.quantize import quantize_onnx_dynamic, quantize_onnx_static
from .utils.initializers import make_initializer
from .utils.model_inspector import format_inspection_report, format_model_summary, inspect_graph, model_summary
from .utils.transforms import (
    ColorJitter,
    ComposeTransform,
    CutMix,
    MixUp,
    RandomCrop,
    RandomRotation,
    augment_chw_batch,
    default_train_transforms,
)
from .data.pipeline import MultiprocessDataLoader


# 定义当你使用 `from MyFlows import *` 时，会暴露哪些名字
# 这也作为所有暴露给用户的 API 列表
__all__ = [
    # Core
    'Node', 'Variable', 'Tensor', 'Graph',
    'set_device', 'get_device', 'use_cuda', 'cuda_available', 'to_device', 'configure_cuda_dll_path',
    # Layers
    'Layer', 'Dense', 'Conv2D', 'GroupedConv2D', 'DepthwiseConv2D', 'DepthwiseSeparableConv2D', 'DilatedConv2D', 'ConvTranspose2D', 'Dropout', 'MaxPool2d', 'Flatten', 'save_checkpoint', 'load_checkpoint', 'export_onnx',
    'BatchNorm2d', 'GlobalAvgPool2d', 'BasicBlock', 'ResNet18',
    'VGG11', 'vgg_fc_input_dim',
    # Ops
    'Add', 'MatMul', 'PerceptionLoss', 'LogLoss', 'CrossEntropy', 'MSELoss', 'Logistic', 'Softmax', 'Tanh', 'ReLU', 'LeakyReLU', 'Conv2D_Op', 'ConvTranspose2D_Op', 'Dropout_Op', 'MaxPool2d_Op', 'Flatten_Op', 'effective_kernel_size', 'im2col', 'col2im',
    'BatchNorm2d_Op', 'GlobalAvgPool2d_Op',
    # Train
    'Optimizer', 'MBGD', 'Momentum', 'AdaGrad', 'RMSProp', 'Adam', 'RegularizationConfig', 'apply_regularization',
    # Utils
    'mse', 'mae', 'rmse', 'r2_score', 'regression_metrics', 'multivariate_regression_metrics',
    'angle_sign_accuracy', 'accuracy', 'confusion_matrix', 'precision_recall_f1', 'classification_report',
    'DonkeyRegressionEvaluator',
    'TrainingHistory', 'plot_training_curves', 'plot_metric_series',
    'TensorBoardLogger', 'has_tensorboard_events',
    'make_initializer', 'model_summary', 'format_model_summary', 'inspect_graph', 'format_inspection_report',
    'ComposeTransform', 'RandomCrop', 'RandomRotation', 'ColorJitter', 'MixUp', 'CutMix',
    'default_train_transforms', 'augment_chw_batch',
    'quantize_onnx_dynamic', 'quantize_onnx_static',
    'MultiprocessDataLoader',
]
