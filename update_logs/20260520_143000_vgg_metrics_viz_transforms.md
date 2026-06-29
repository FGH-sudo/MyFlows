# MyFlows 更新日志

- 时间戳：2026-05-20 14:30（本地整理）
- 主题：VGG-11、评估指标、数据增强、训练可视化、TensorBoard、ONNX 量化与工具链重构

## 背景与目标

本轮更新面向 Donkey 小车转向角等视觉回归/分类任务，在已有 ResNet、序列化与 ONNX 能力之上继续补齐「训练周边」能力：

- 提供 VGG-11 等经典 CNN 结构，适配小分辨率输入（如 120×160）。
- 统一回归与分类评估指标，并封装 Donkey 任务专用评估器。
- 用纯 NumPy + OpenCV 实现可组合的数据增强流水线（含 MixUp/CutMix）。
- 支持 Matplotlib 曲线导出与可选 TensorBoard 标量/图像日志。
- 提供 ONNX 动态/静态 INT8 量化入口。
- 移除旧的 `utils/data.py` 杂项工具，将公开 API 收敛到更清晰的 `metrics` / `transforms` / `viz` 模块。

## 主要变更

### 1) 新增 VGG-11 回归网络

- 位置：`layers/vgg.py`
- 新增：
  - `VGG11`：5 段 Conv+ReLU + MaxPool，通道 64→128→256→512→512，三层全连接 4096→4096→`output_dim`
  - `vgg_fc_input_dim()`：根据输入高宽推算展平后特征维度
- 默认假设小图输入（`image_h=120`, `image_w=160`），便于 Donkey 数据集。
- 顶层 `__init__.py` 已导出 `VGG11`、`vgg_fc_input_dim`。

### 2) 统一评估指标模块

- 位置：`utils/metrics.py`
- 回归：`mse`、`mae`、`rmse`、`r2_score`、`regression_metrics`、`multivariate_regression_metrics`
- Donkey 专用：`angle_sign_accuracy`（转向角符号正确率，含近零样本特殊处理）
- 分类：`accuracy`、`confusion_matrix`、`precision_recall_f1`、`classification_report`
- 任务封装：`DonkeyRegressionEvaluator`（批量汇总回归 + 符号指标 + 可选分类指标）

### 3) 数据增强（NumPy + OpenCV）

- 位置：`utils/transforms.py`
- 约定：单样本为 HWC、`[0,1]` 浮点图像；提供 `chw_to_hwc` / `hwc_to_chw` 与 NCHW batch 接口。
- 变换基类：`Transform`、`ComposeTransform`
- 单样本增强：
  - `RandomCrop`：随机比例裁剪后 resize 回原尺寸
  - `RandomRotation`：小角度旋转（`BORDER_REFLECT_101`）
  - `ColorJitter`：亮度 / 对比度 / 饱和度扰动
- Batch 级混合（需 `set_partner` 或 `apply_batch_pairwise_mix`）：
  - `MixUp`：Beta 分布混合系数，图像与向量标签线性混合
  - `CutMix`：随机矩形区域粘贴，标签按面积比混合
- 便捷 API：
  - `default_train_transforms()`：Crop + Rotation + ColorJitter
  - `augment_chw_batch()`：对 NCHW batch 应用流水线，可选 MixUp/CutMix
- 依赖：`opencv-python`（`import cv2` 失败时抛出明确 ImportError）

### 4) 训练过程可视化

- 位置：`utils/viz.py`
- `TrainingHistory`：记录 step / epoch 级 loss，支持 `to_dict` / `from_dict` / `save_json`
- `plot_training_curves()`：导出 per-step、per-epoch、combined 三类 PNG
- `plot_metric_series()`：通用单序列折线图（评估指标等）
- 使用 `matplotlib` Agg 后端，适合无 GUI 服务器环境。

### 5) TensorBoard 日志封装

- 位置：`utils/tensorboard_logger.py`
- `TensorBoardLogger`：封装 `torch.utils.tensorboard.SummaryWriter`
- 支持：`log_scalar`、`log_scalars`、`log_image`（CHW）、`log_images_grid`
- 未安装 `tensorboard`/`torch` 时自动禁用并打印提示，不中断训练主流程。
- 可选依赖见 `requirements-tb.txt`。
- 辅助：`has_tensorboard_events()` 检查日志目录是否已有事件文件。

### 6) ONNX INT8 量化

- 位置：`utils/quantize.py`
- `quantize_onnx_dynamic()`：动态权重量化（激活仍为 float）
- `quantize_onnx_static()`：静态量化（需 `CalibrationDataReader`）
- 依赖：`onnxruntime` 量化 API。

### 7) 序列化小改进

- 位置：`utils/serialization.py`
- `load_checkpoint`：仅在传入 `optimizer` 时打印「从 Epoch N 继续训练」提示，推断加载更安静。
- `export_onnx`：浮点张量统一转为 `float32`，避免 ONNX 类型不一致。

### 8) 移除旧 `utils/data.py` 并更新导出

- 删除：`utils/data.py`（原 `DataGenerator`、`StandardScaler`、`augment_data`、`gradient_check`、`plot_computational_graph` 等）
- `__init__.py` 改为从 `metrics`、`transforms`、`viz`、`tensorboard_logger`、`quantize` 导出上述新 API。
- 数据增强与可视化职责分离，避免单文件职责混杂。

## 新增/修改文件列表

### 新增

- `layers/vgg.py`
- `utils/metrics.py`
- `utils/transforms.py`
- `utils/viz.py`
- `utils/tensorboard_logger.py`
- `utils/quantize.py`
- `requirements-tb.txt`
- `tests/test_vgg_smoke.py`
- `tests/test_metrics.py`
- `tests/test_transforms.py`
- `tests/test_tensorboard_logger.py`

### 修改

- `__init__.py`：导出 VGG、指标、增强、可视化、TensorBoard、量化 API
- `utils/serialization.py`：加载提示与 ONNX float32 规范化

### 删除

- `utils/data.py`

## 验证结果

在 `PYTHONPATH=d:\DL\testmyflow` 下已通过：

- `python -m unittest MyFlows.tests.test_transforms`（5 项）
- `python -m unittest MyFlows.tests.test_metrics`（5 项，含 `TrainingHistory` 绘图）
- `python -m unittest MyFlows.tests.test_vgg_smoke`（2 项）
- `python -m unittest MyFlows.tests.test_tensorboard_logger`（2 项，需 tensorboard 可选依赖）

## 当前限制与后续方向

- VGG-11 全连接层参数量较大，小 batch 纯 NumPy 训练仍偏慢；可结合图融合与 Conv 快路径继续优化。
- MixUp/CutMix 需在 batch 级配对样本，与 DataLoader 集成时建议在 collate 后调用 `augment_chw_batch`。
- TensorBoard 图像网格依赖 `torchvision.utils.make_grid`，缺失时退化为单张 `log_image`。
- 静态量化需提供校准数据读取器，后续可结合 Donkey 验证集封装示例。
- 旧 `utils/data.py` 中的 `gradient_check`、计算图绘图等若仍需要，可迁回独立 `utils/debug.py` 模块。
