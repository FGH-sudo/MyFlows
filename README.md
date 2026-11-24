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
| `core/` | 设备（含多卡 `cuda:<rank>` 隔离）、`Module`/`Parameter`/`Buffer`、`Tensor`/`Variable`/`Node`、`Graph`、图优化 |
| `ops/` | 基础算子、激活、卷积（im2col+GEMM 及融合 FP32 CUDA 深度可分离卷积）、BN/`SyncBatchNorm2d_Op`、Dropout、损失 |
| `ops/cuda_native/` | 原生 C/C++ 调度自写 Conv/Pool CUDA kernel，`device_capabilities()` 与 `cc >= 8.0` `cp.async` autotune 守卫，GEMM 支持寄存器分块与 cuBLAS（`native.gemm`）；第 19 讲矩阵乘 kernel（`native.matmul`） |
| `distributed/` | 同步 Parameter Server（Socket JSON / gRPC）与 Ring AllReduce（gRPC）、`cuda:rank` 进程设备绑定、`make_sync_bn_allreduce`、状态校验与故障注入 |
| `monitoring/` | NVML 优先、nvidia-smi 兜底的 GPU 资源采样与 JSONL 记录 |
| `transfer/` | 预训练权重导入、Adapter / Regression Head、分层冻结与 ResNet LoRA 微调 |
| `examples/stage1_cnn.py` | 固定 32 样本 FP32 CNN 训练示例 |
| `layers/` | Dense / Conv / Pool / AdaptiveAvgPool / Dropout / `SyncBatchNorm2d`（含 `convert_sync_batchnorm`）、`ResNet18` / `ResNet34` / `AlexNet` / `MobileNetV2` |
| `train/` | 优化器（MBGD / Momentum / AdaGrad / RMSProp / Adam）、L1/L2 正则、`CosineWarmupScheduler` |
| `data/` | `MultiprocessDataLoader` 生产者-消费者流水线 |
| `utils/` | checkpoint（含 `TrainingState` 原子发布与 `Module.state_dict`）、动态 batch ONNX 导出、指标、驾驶/通用增强、可视化、Grad-CAM、量化、初始化、模型检查 |
| `tests/` | `unittest` 测试套件 |

## 主要能力

- **动态图与模块化**：`ms.Graph(logits)` 自动拓扑排序，支持 `forward()` / `backward()`；`Module` / `Parameter` / `Buffer` 自动注册与 `state_dict()` / `load_state_dict()`（与旧 `Layer` 无缝互通）
- **设备与多卡隔离**：`set_device` / `use_cuda` / `cuda_available` / `cuda_device_count` / `resolve_rank_device` 支持显式 `cuda:<index>`，Windows 下可复用本机 CUDA DLL
- **数值与融合算子**：层参数默认 float32（`ms.set_default_dtype(np.float64)` 可切换）；BN 统计量、bias 梯度、交叉熵、MSE 等规约在 FP64 中累加；BN / 交叉熵 / Adam / 深度可分离卷积前反向在 GPU 上提供融合 CUDA kernel（`MobileNetV2 64x160 b32` 训练步实测 `50.9 ms/step`、`648.6 MiB` 显存池，较优化前 `2281.7 ms/step` / `8810 MiB` 提速 **44.8×**，并通过 `compute-sanitizer` `memcheck`/`racecheck`/`synccheck` 零错误零竞争验证）
- **分布式同步 BN**：`SyncBatchNorm2d` + `convert_sync_batchnorm` 通过高精度 FP64 split-float32 collective 同步前向均值/方差与反向激活梯度，支持不均等分片与评估态 Conv+BN 折叠
- **CNN 与迁移**：Conv2D（含分组 / 深度 / 空洞 / 转置）、MaxPool、AdaptiveAvgPool、BN、ReLU6、GlobalAvgPool；主线支持 `ResNet18` / `ResNet34`（可选 `stem` / `base_width`）以及 `AlexNet` / `MobileNetV2`、可配置回归头与预训练 / 冻结 / LoRA 微调
- **训练周边**：Dropout、权重正则、Huber/加权控制损失、余弦预热调度、Xavier / Kaiming 等初始化、早停与诊断由父仓库训练脚本编排
- **图优化**：常量折叠、Linear 融合、Conv+ReLU、推理态 Conv+BN 折叠（`apply_graph_optimizations(..., mode="inference")`，兼容 ONNX 导出）
- **序列化与动态 ONNX**：JSON + NPZ checkpoint、`TrainingState` 原子发布与滚动修订清理；`export_onnx(..., dynamic_batch=True)` 支持第 0 维符号 `"batch"`、自定义 metadata、`ReLU6` / `AdaptiveAvgPool2d` / 评估态 `SyncBatchNorm2d_Op` 导出
- **可视化与监控**：TensorBoard logger（含指标与资源曲线）、训练 dashboard / observers、PNG 曲线（`utils/viz.py`）与 `monitoring` GPU 采样
- **指标与增强**：回归 / 分类指标、`DonkeyRegressionEvaluator`；RandomCrop / Rotation / ColorJitter / MixUp / CutMix
- **解释与量化**：Grad-CAM；ONNX 动态 / 静态 INT8（闭环评测与服务部署在父仓库 `apps/` / `scripts/`）

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
conv = ms.Conv2D(1, 4, kernel_size=3, padding=1, backend="cuda_native_register",
                 dtype=np.float32, initializer=make_initializer(seed=0),
                 fuse_activation=False)
pool = ms.MaxPool2d(2, 2, backend="cuda_native_register")
```

`ResNet18(conv_backend=...)` 把后端传给全部卷积与池化；`Dense(backend=...)` 同样可选 `cuda_native_register` / `cuda_native_cublas`，另有实验用的补偿累加后端 `cuda_native_kahan` / `cuda_native_dot2`（结果与 shared-memory kernel 逐位相同）。输入和标签可用 `Variable(..., requires_grad=False)` 标记，卷积与 Linear 据此跳过 dX。

`auto` 保留 CPU/NumPy 与 GPU/CuPy 默认行为。两个原生后端要求当前 GPU 上的 FP32 输入；卷积由原生 C/C++ 层调度自写 im2col/col2im kernel，GEMM 在 `cuda_native_register` 下用第 19 讲的寄存器分块 kernel（`apps/train` 与分布式任务在 CUDA 上的默认），在 `cuda_native_cublas` 下调用 cuBLAS；池化由同一原生层调度自写 max-pool kernel。当前原生卷积限定 groups=1、dilation=1，池化不带 padding；最大值相等时选择首个位置，重叠窗口梯度求和。输入须为有限值；wrapper 接受非连续 view 并连续化，普通调用不主动同步设备。

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
