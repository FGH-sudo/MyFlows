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
| `ops/cuda_native/` | 原生 C/C++ 调度自写 Conv/Pool CUDA kernel，并调用 cuBLAS |
| `distributed/` | CPU 同步 PS，server/worker/launcher、消息校验与计时 |
| `examples/stage1_cnn.py` | 固定 32 样本 FP32 CNN 训练示例 |
| `layers/` | Dense / Conv / Pool / Dropout、`ResNet18` |
| `train/` | 优化器（MBGD / Momentum / AdaGrad / RMSProp / Adam）、L1/L2 正则 |
| `data/` | `MultiprocessDataLoader` 生产者-消费者流水线 |
| `utils/` | checkpoint、ONNX、指标、增强、可视化、Grad-CAM、量化、初始化、模型检查 |
| `tests/` | `unittest` 测试套件 |

## 主要能力

- **动态图**：`ms.Graph(logits)` 自动拓扑排序，支持 `forward()` / `backward()`
- **设备**：`set_device` / `use_cuda` / `cuda_available`；Windows 下可复用本机 CUDA DLL
- **CNN**：Conv2D（含分组 / 深度 / 空洞 / 转置）、MaxPool、BN、GlobalAvgPool；默认卷积为 im2col + GEMM，可显式选用阶段一直接 CUDA C 或阶段二 CUDA im2col/col2im 实验后端
- **模型**：当前训练和评估主线使用 `ResNet18`，回归头使用 `output_dim`（DonkeyCar 为 2）
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
python -m tools.run_tests --scope framework
```

单测示例：

```bash
python -m unittest MyFlows.tests.test_convolution -v
python -m tools.run_tests --scope framework --pattern test_resnet18_smoke.py
```

## 第一阶段 CUDA 与 PS

在父仓库使用 `requirements-stage1-lock.txt` 创建独立 Python 3.11 环境；CuPy 的 CUDA 用户态依赖由 pip 安装，无须安装 PyTorch 才能运行新算子。旧 TensorBoard logger 仍可选依赖 PyTorch。

```python
import numpy as np
import MyFlows as ms
from MyFlows.utils.initializers import make_initializer

ms.set_device("cuda")
conv = ms.Conv2D(1, 4, kernel_size=3, padding=1, backend="cuda_native_cublas",
                 dtype=np.float32, initializer=make_initializer(seed=0),
                 fuse_activation=False)
pool = ms.MaxPool2d(2, 2, backend="cuda_native_cublas")
```

`auto` 保留 CPU/NumPy 与 GPU/CuPy 默认行为。`cuda_native_cublas` 要求当前 GPU 上的 FP32 输入；卷积由原生 C/C++ 层调度自写 im2col/col2im kernel 和 cuBLAS，池化由同一原生层调度自写 max-pool kernel。当前原生卷积限定 groups=1、dilation=1，池化不带 padding；最大值相等时选择首个位置，重叠窗口梯度求和。输入须为有限值；wrapper 接受非连续 view 并连续化，普通调用不主动同步设备。

父仓库 `benchmark.cuda_ops` 和 `benchmark.ps_demo` 提供完整运行入口，`benchmark.profile_cuda` 生成并检查真实 Nsight 报告。范围、命令和证据见 [第一阶段报告](../docs/experiments/semester_2026_fall/stage1/README.md)。

## 与父项目的关系

| 层级 | 职责 |
|------|------|
| **MyFlows** | 框架：计算图、算子、层、优化器、checkpoint / ONNX、指标与可视化 |
| **testmyflow** | 应用：DonkeyCar 数据、`apps/train` / `apps/eval` / `apps/serve`、部署与实验文档 |

训练、评估、服务入口在父仓库，例如：

```bash
python -m apps.train.train_myflows_donkey --max-samples 200 --epochs 1 --device auto
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
