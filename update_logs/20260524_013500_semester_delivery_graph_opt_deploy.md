# MyFlows 更新日志

- 时间戳：2026-05-24 01:35（本地整理）
- 主题：学期收官 — 计算图 Conv+BN 折叠、生产者-消费者 DataLoader、VGG/指标/可视化/量化；父仓库部署与文档（见文末）

## 背景与目标

对照本学期任务书基本任务 (1)–(4) 与拓展 (3)(4)(5)，在 5 月 VGG/训练周边能力之上继续收口：

- **拓展 (3)**：计算图优化 — 推理态 **Conv+BN 折叠**（不做 Winograd / Loop Tiling）
- **拓展 (4)**：框架结构 — `MultiprocessDataLoader` 生产者-消费者数据流水线
- **任务书 (1)**：推理部署与量化闭环（部分在父仓库 `testmyflow` 根目录实现，见 §父仓库）
- **任务书 (4)**：设计文档与期末报告正文（父仓库 `docs/`）

本日志以 **MyFlows 包内代码** 为主；父目录脚本与 `deploy/` 单独列出便于联调。

## 主要变更（MyFlows 包内）

### 1) 计算图优化：Conv+BN 折叠（拓展 3）

- 位置：`core/graph_opt.py`
- 新增：
  - `_fold_bn_weights()`：将 eval 态 `BatchNorm2d_Op` 的 γ/β/μ/σ² 折入上游 `Conv2D_Op` 的 kernel/bias
  - `fold_bn_into_conv()`：遍历子图，跳过 BN 节点（`replace_node`）
  - `apply_graph_optimizations(..., mode="train"|"inference", fold_bn=...)`
    - `mode="inference"` 时默认 `fold_bn=True`；训练态不折叠，避免破坏 BN 统计
- 数学（与 `docs/algorithm_design.md` 一致）：

  ```
  W' = W · (γ/√(σ²+ε)).reshape(C_out,1,1,1)
  b' = (b - μ) · γ/√(σ²+ε) + β
  ```

- 测试：`tests/test_graph_opt_bn_fold.py`（ResNet18 eval 折叠前后数值一致 + BN 节点数下降）

### 2) 生产者-消费者 DataLoader（拓展 4）

- 位置：`data/pipeline.py`、`data/__init__.py`
- 新增 `MultiprocessDataLoader`：
  - worker 进程：`load_fn` 做 IO/解码，可选 `transform_fn`
  - 主进程：从 `multiprocessing.Queue` 取 batch
  - `num_workers=0` 时退化为进程内同步加载
- 顶层 `__init__.py` 导出 `MultiprocessDataLoader`
- 父仓库 `train_myflows_donkey.py` 增加 `--num-workers` 参数；吞吐基准见父仓库 `benchmark/dataloader_bench.py`

### 3) 延续 5 月能力（VGG / 指标 / 增强 / TB / 量化）

若尚未合并，本分支同时包含（详见 `20260520_143000_vgg_metrics_viz_transforms.md`）：

| 模块 | 路径 | 说明 |
|------|------|------|
| VGG-11 | `layers/vgg.py` | Donkey 小图分类 |
| 指标 | `utils/metrics.py` | 回归/分类 + `DonkeyRegressionEvaluator` |
| 增强 | `utils/transforms.py` | Crop/Rotation/ColorJitter/MixUp/CutMix |
| 可视化 | `utils/viz.py`、`utils/tensorboard_logger.py` | PNG 曲线 + TensorBoard |
| 量化 | `utils/quantize.py` | ONNX 动态/静态 INT8 |
| 设备 | `core/device.py` | CPU/CuPy CUDA |
| 删除 | `utils/data.py` | 职责拆分到 metrics/transforms/viz |

### 4) 其它框架改动

- `core/graph.py`：构图期优化开关与 `graph_opt` 对接
- `core/tensor.py`、`train/opt.py`、各 `ops/*`：设备 `xp` 与 dtype 一致性
- `utils/serialization.py`：checkpoint / ONNX float32 规范化
- 依赖：`requirements-gpu.txt`、`requirements-tb.txt`

## 父仓库 testmyflow（与本 git 仓库同级，需单独管理）

以下文件位于 `D:\DL\testmyflow\` 根目录，**不在 MyFlows 子仓库的 git 跟踪范围内**，联调时请一并备份或在上级目录初始化仓库：

| 类别 | 路径 |
|------|------|
| FastAPI | `serve_fastapi.py`、`onnx_predictor.py` |
| gRPC 共用 | `serve_grpc.py`（改为引用 `onnx_predictor`） |
| INT8 闭环 | `scripts/calibration_reader.py`、`scripts/run_quantize_eval.py` |
| 压测/对比 | `benchmark/serve_bench.py`、`benchmark/plot_compare.py`、`benchmark/dataloader_bench.py`、`benchmark/requirements-bench.txt` |
| Docker/K8s | `deploy/docker/`、`deploy/k8s/` |
| 文档 | `docs/system_design.md`、`docs/final_report.md`、`docs/experiments/` |
| 交付说明 | `update_logs/20260524_final_semester_delivery.md`（若存在） |

部署依赖：`requirements-deploy.txt`（grpcio、onnxruntime、fastapi、uvicorn 等）。

## 新增/修改文件列表（MyFlows git）

### 新增

- `core/device.py`
- `data/pipeline.py`、`data/__init__.py`
- `layers/vgg.py`
- `utils/metrics.py`、`utils/transforms.py`、`utils/viz.py`、`utils/tensorboard_logger.py`、`utils/quantize.py`
- `requirements-gpu.txt`、`requirements-tb.txt`
- `tests/test_device.py`、`tests/test_graph_opt_bn_fold.py`、`tests/test_metrics.py`、`tests/test_transforms.py`、`tests/test_tensorboard_logger.py`、`tests/test_vgg_smoke.py`
- `update_logs/20260520_143000_vgg_metrics_viz_transforms.md`
- `update_logs/20260524_013500_semester_delivery_graph_opt_deploy.md`（本文件）

### 修改

- `__init__.py`：导出 VGG、指标、增强、TB、量化、DataLoader、设备 API
- `core/graph_opt.py`：BN 折叠 + `mode` 参数
- `core/graph.py`、`core/tensor.py`、`layers/*`、`ops/*`、`train/opt.py`、`utils/serialization.py`

### 删除

- `utils/data.py`
- `update_logs/20260323_005819_第四周主题一简要讲解参考.md`（仓库内已移除）

## 验证结果

在 `PYTHONPATH` 包含 `D:\DL\testmyflow` 的前提下：

```bash
python -m unittest MyFlows.tests.test_graph_opt_bn_fold -v
python -m unittest MyFlows.tests.test_vgg_smoke MyFlows.tests.test_metrics MyFlows.tests.test_transforms -v
```

- `test_graph_opt_bn_fold`：2 项通过（数值一致 + BN 节点减少）

父仓库建议验证：

```bash
python scripts/run_quantize_eval.py --fp32 mycar/models/myflow_resnet18_best.onnx --max-samples 200
python serve_fastapi.py --model mycar/models/myflow_resnet18_best.onnx
python benchmark/compare_frameworks.py --epochs 1 --samples 32 --device cpu
```

TensorFlow 对比需：`pip install -r benchmark/requirements-bench.txt`。

## 当前限制与后续

- 父仓库与 MyFlows 为两个目录层级，提交代码时请确认是否需在 `testmyflow` 根目录单独 `git init`。
- `MultiprocessDataLoader` 在 Windows 上需 `spawn`；训练脚本内全流程接入仍可继续加强。
- 期末答辩 PPT 建议按 `docs/final_report.md` 与交付说明增补 FastAPI、INT8、图优化、Docker 四页。
