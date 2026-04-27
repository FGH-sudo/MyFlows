# MyFlows 更新日志

- 时间戳：2026-04-28 01:17（本地整理）
- 主题：ResNet-18、BatchNorm/GlobalAvgPool、MSELoss、JSON+NPZ 序列化与 ONNX 导出支持

## 背景与目标

本轮更新围绕 MyFlows 从“基础 CNN 框架”继续向“可构建中等规模视觉模型”的方向推进，重点目标包括：

- 支持 ResNet-18 这类残差网络结构。
- 补齐 ResNet 所需的 BatchNorm2d 与 GlobalAvgPool2d。
- 增加/整理 MSE 回归损失，便于处理连续值预测任务。
- 移除 `layers/layer.py` 中旧的 pickle checkpoint 函数，将模型保存/加载改为更清晰、可检查、跨版本更友好的 JSON + NPZ 格式。
- 提供 ONNX 导出入口，为后续推理部署和第三方工具链兼容做准备。

## 主要变更

### 1) 新增 ResNet-18 结构

- 位置：`layers/resnet.py`
- 新增组件：
  - `BasicBlock`
  - `ResNet18`
- `ResNet18` 支持两种 stem：
  - `stem="imagenet"`：`7x7/s=2 Conv + BN + ReLU + MaxPool`
  - `stem="cifar"`：`3x3/s=1 Conv + BN + ReLU`，不做初始 MaxPool
- 主体结构：
  - 4 个 stage
  - 每个 stage 包含 2 个 `BasicBlock`
  - 通道宽度默认是 `64 -> 128 -> 256 -> 512`
  - stage 间通过 stride=2 下采样
  - 末端使用 `GlobalAvgPool2d + Linear` 分类头

### 2) 新增 BatchNorm2d 与 GlobalAvgPool2d

- 位置：
  - `ops/batchnorm.py`
  - `layers/resnet.py`
- 新增算子：
  - `BatchNorm2d_Op`
  - `GlobalAvgPool2d_Op`
- 新增层：
  - `BatchNorm2d`
  - `GlobalAvgPool2d`
- `BatchNorm2d_Op` 行为：
  - 训练模式下使用 batch 统计量，并更新 `running_mean` / `running_var`
  - 推断模式下使用 running stats
  - 支持对 `x`、`gamma`、`beta` 的反向传播
- `BatchNorm2d` 层提供：
  - `train(mode=True)`
  - `eval()`
- `ResNet18.train()` / `ResNet18.eval()` 会统一切换内部 BN 层状态。

### 3) 新增 MSELoss

- 位置：`ops/loss.py`
- 新增 `MSELoss`，用于回归任务。
- 前向：
  - 计算 `mean((y_pred - y_true) ** 2)`
- 反向：
  - 对预测值回传 `2 * (y_pred - y_true) / n`
- 顶层 `__init__.py` 已导出 `MSELoss`。

### 4) 删除 `layer.py` 中旧 checkpoint 函数

- 位置：`layers/layer.py`
- 删除内容：
  - `save_checkpoint`
  - `load_checkpoint`
- 原实现问题：
  - 使用 pickle 保存整体状态，文件不可读、跨版本兼容性弱。
  - checkpoint 逻辑放在 layer 定义文件末尾，职责不够清晰。
  - 对新加入的 BN running stats、更多 optimizer state 的扩展不够明确。
- 删除后：
  - `layers/layer.py` 回归只负责层定义。
  - 保存/加载逻辑迁移到 `utils/serialization.py`。

### 5) 新增 JSON + NPZ checkpoint 格式

- 位置：`utils/serialization.py`
- 新增 API：
  - `save_checkpoint(layers, optimizer=None, epoch=0, acc=0.0, filepath="checkpoint")`
  - `load_checkpoint(layers, optimizer=None, filepath="checkpoint")`
- 保存结果：
  - `*.json`：保存结构化元信息
  - `*.npz`：保存 NumPy 数组
- JSON 中保存：
  - checkpoint 版本号
  - epoch / best_acc
  - layer 元信息
  - 参数名、shape、dtype、trainable
  - BN running buffers 元信息
  - optimizer 标量与状态索引
- NPZ 中保存：
  - 模型参数数组
  - BN `running_mean` / `running_var`
  - 优化器状态数组（如 `v`、`m`、`s`、`acc_gradient`）

示例：

```python
from MyFlows import save_checkpoint, load_checkpoint

save_checkpoint(model, optimizer=optimizer, epoch=10, acc=0.92, filepath="checkpoints/resnet18")

epoch, best_acc = load_checkpoint(model, optimizer=optimizer, filepath="checkpoints/resnet18")
```

会生成：

- `checkpoints/resnet18.json`
- `checkpoints/resnet18.npz`

### 6) 新增 ONNX 导出入口

- 位置：`utils/serialization.py`
- 新增 API：
  - `export_onnx(graph, filepath, input_nodes=None, output_names=None, opset=13)`
- 当前支持导出的常见算子：
  - `Variable`
  - `Linear`
  - `MatMul`
  - `Add`
  - `ReLU`
  - `LeakyReLU`
  - `Logistic`
  - `Tanh`
  - `Softmax`
  - `Conv2D_Op`
  - `Conv2D_ReLU_Op`
  - `Conv2D_LeakyReLU_Op`
  - `ConvTranspose2D_Op`
  - `ConvTranspose2D_ReLU_Op`
  - `ConvTranspose2D_LeakyReLU_Op`
  - `MaxPool2d_Op`
  - `Flatten_Op`
  - `BatchNorm2d_Op`
  - `GlobalAvgPool2d_Op`
- ONNX 依赖：
  - 已安装 `onnx 1.21.0`

示例：

```python
from MyFlows import Variable, Graph, Dense, export_onnx
import numpy as np

x = Variable(np.random.randn(2, 3), name="input")
model = Dense(3, 2, name="dense")
y = model(x)

graph = Graph(y)
graph.forward()

export_onnx(graph, "linear.onnx", input_nodes=[x], output_names=["output"])
```

## 新增/修改文件列表

### 新增

- `ops/batchnorm.py`
  - `BatchNorm2d_Op`
  - `GlobalAvgPool2d_Op`
- `layers/resnet.py`
  - `BatchNorm2d`
  - `GlobalAvgPool2d`
  - `BasicBlock`
  - `ResNet18`
- `utils/serialization.py`
  - JSON+NPZ checkpoint
  - ONNX 导出
- `tests/test_resnet18_smoke.py`
  - ResNet 与 BN/GAP smoke test
- `tests/test_serialization.py`
  - JSON+NPZ 保存/加载测试
  - BN buffer 恢复测试
  - 可选 ONNX 导出测试

### 修改

- `__init__.py`
  - 导出 `BatchNorm2d`
  - 导出 `GlobalAvgPool2d`
  - 导出 `BasicBlock`
  - 导出 `ResNet18`
  - 导出 `BatchNorm2d_Op`
  - 导出 `GlobalAvgPool2d_Op`
  - 导出 `MSELoss`
  - 从 `utils.serialization` 导出 `save_checkpoint`
  - 从 `utils.serialization` 导出 `load_checkpoint`
  - 从 `utils.serialization` 导出 `export_onnx`
- `layers/layer.py`
  - 删除 pickle checkpoint 相关 import
  - 删除旧 `save_checkpoint`
  - 删除旧 `load_checkpoint`
- `ops/loss.py`
  - 新增 `MSELoss`

## 验证结果

已通过以下验证：

- `python tests/test_resnet18_smoke.py`
  - BN forward shape
  - BN gradcheck
  - GlobalAvgPool2d gradcheck
  - BasicBlock forward/backward
  - ResNet18 CIFAR 前向输出
  - train/eval 状态切换
- `python tests/test_serialization.py`
  - JSON+NPZ checkpoint roundtrip
  - BN running buffers roundtrip
  - ONNX 可选导出测试入口
- `python -c "import onnx; print('onnx', onnx.__version__)"`
  - 输出：`onnx 1.21.0`
- 相关文件 linter 检查无报错。

## 当前限制与后续方向

- ONNX 导出当前覆盖的是 MyFlows 中常用推断算子，复杂自定义节点仍需要继续补映射。
- JSON+NPZ checkpoint 目前按参数顺序与元信息恢复，后续可以进一步增强为“按稳定参数路径恢复”，以支持模型结构轻微变化后的部分加载。
- ResNet-18 已能构建和前向/反向 smoke test，但纯 NumPy 后端训练较大图像时仍会偏慢；后续优化可优先考虑：
  - Conv+BN 推断期折叠
  - Conv 快路径优化
  - BN 反向进一步向量化/缓存优化
  - Numba/Cython 加速热点算子

