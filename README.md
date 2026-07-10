# MyFlows

教学向的 NumPy / CuPy **动态计算图**深度学习框架。通过 `Variable` → `Node` / `Layer` 即时构图，由 `Graph` 完成拓扑排序与前向 / 反向传播；可选 CuPy 加速。典型应用是父仓库 DonkeyCar 道路图像的 `[angle, throttle]` 回归。

```python
import MyFlows as ms
from MyFlows.layers.resnet import ResNet18

ms.use_cuda()  # 或 ms.set_device("cpu")
```

本目录是独立 Git 仓库，通常作为父项目 `testmyflow` 的嵌套子模块使用，不摊平成普通源码目录。

## 目录结构

| 路径 | 职责 |
|------|------|
| `core/` | 设备、`Tensor` / `Variable` / `Node`、`Graph`、图优化 |
| `ops/` | 基础算子、激活、卷积（im2col+GEMM）、BN、Dropout、损失 |
| `layers/` | Dense / Conv / Pool / Dropout、`ResNet18`、`VGG11` |
| `train/` | 优化器（MBGD / Momentum / AdaGrad / RMSProp / Adam）、L1/L2 正则 |
| `data/` | `MultiprocessDataLoader` 生产者-消费者流水线 |
| `utils/` | checkpoint、ONNX、指标、增强、可视化、Grad-CAM、量化、初始化、模型检查 |
| `tests/` | `unittest` 测试套件 |
| `update_logs/` | 功能变更记录 |

## 主要能力

- **动态图**：`ms.Graph(logits)` 自动拓扑排序，支持 `forward()` / `backward()`
- **设备**：`set_device` / `use_cuda` / `cuda_available`；Windows 下可复用本机 CUDA DLL
- **CNN**：Conv2D（含分组 / 深度 / 空洞 / 转置）、MaxPool、BN、GlobalAvgPool；卷积实现为 im2col + GEMM
- **模型**：`ResNet18`、`VGG11`，回归头使用 `output_dim`（DonkeyCar 为 2）
- **训练周边**：Dropout、权重正则、Xavier / Kaiming 等初始化、早停与诊断由父仓库训练脚本编排
- **图优化**：常量折叠、Linear 融合、Conv+ReLU、推理态 Conv+BN 折叠（`Graph(optimize=True)`）
- **序列化**：JSON + NPZ checkpoint；`export_onnx` 导出部署图
- **可视化**：TensorBoard logger、训练 dashboard / observers、PNG 曲线（`utils/viz.py`）
- **指标与增强**：回归 / 分类指标、`DonkeyRegressionEvaluator`；RandomCrop / Rotation / ColorJitter / MixUp / CutMix
- **解释与量化**：Grad-CAM；ONNX 动态 / 静态 INT8（评测闭环在父仓库 `scripts/`）

## 安装

在父仓库根目录（或将父目录加入 `PYTHONPATH`）安装依赖：

```bash
pip install numpy opencv-python-headless onnx

# 可选：GPU（CuPy，需匹配本机 CUDA）
pip install -r MyFlows/requirements-gpu.txt

# 可选：TensorBoard 可视化（需 torch SummaryWriter）
pip install -r MyFlows/requirements-tb.txt
```

无独立 `setup.py`；以包目录形式导入：

```python
import MyFlows as ms
from MyFlows.layers.resnet import ResNet18
from MyFlows.utils.checkpoint import save_checkpoint, load_checkpoint
```

## 测试

在父仓库根目录执行：

```bash
python -m unittest discover -s MyFlows/tests -p "test_*.py"
```

单测示例：

```bash
python -m unittest MyFlows.tests.test_convolution -v
python -m unittest MyFlows.tests.test_resnet18_smoke MyFlows.tests.test_vgg_smoke -v
```

## 与父项目的关系

| 层级 | 职责 |
|------|------|
| **MyFlows** | 框架：计算图、算子、层、优化器、checkpoint / ONNX、指标与可视化 |
| **testmyflow** | 应用：DonkeyCar 数据、`apps/train` / `apps/eval` / `apps/serve`、部署与实验文档 |

训练、评估、服务入口在父仓库，例如：

```bash
python -m apps.train.train_myflows_donkey --max-samples 200 --epochs 1 --device auto
python -m apps.train.train_vgg_donkey_regression --max-samples 64 --epochs 1 --device cpu
```

ONNX 导出由训练脚本 `--export-onnx` 触发，底层调用本仓库的 `utils/onnx_exporter.py`。

## 快速示例

```python
import numpy as np
import MyFlows as ms
from MyFlows.layers.resnet import ResNet18

ms.set_device("cpu")
x = ms.Variable(np.random.randn(2, 3, 120, 160).astype(np.float32))
model = ResNet18(output_dim=2)
logits = model(x)
graph = ms.Graph(logits)
graph.forward()
print(logits.value.shape)  # (2, 2)
```

## 变更记录

功能演进说明见 [`update_logs/`](update_logs/)。
