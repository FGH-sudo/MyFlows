# MyFlows 更新日志

- 时间戳：2026-03-21 19:09:45 CST (+0800)
- 主题：Tensor 封装与卷积 `im2col` 优化

## 本次更新内容

### 1. 为框架增加统一 Tensor 封装层

- 新增 `core/tensor.py`，提供轻量级 `Tensor` 类型，底层仍基于 `numpy.ndarray`。
- 修改 `core/node.py`，将 `Node` 内部存储改为 `_value/_grad -> Tensor`。
- 保留 `value/grad` 的 numpy 兼容接口，现有算子、层与训练逻辑无需整体重写。
- 新增 `tensor` 和 `grad_tensor` 访问方式，便于后续扩展更明确的张量接口。
- 在 `train/opt.py` 中补强 `Optimizer.update(var_gradients=...)` 路径，使其可直接接收 `Tensor` 梯度。
- 在 `__init__.py` 中导出 `Tensor`，支持 `from MyFlows import Tensor`。

### 2. 为卷积模块整理并接入 `im2col/col2im`

- 在 `ops/convolution.py` 中新增显式 `im2col` 和 `col2im` 函数。
- 将 `Conv2D_Op` 切换到 `im2col -> 矩阵乘法 -> col2im` 的标准路径。
- 将 `MaxPool2d_Op` 也统一到相同的 patch 展开与还原流程。
- 为窗口索引增加上下文缓存，避免同一层在重复训练过程中反复构造索引。
- 在 `__init__.py` 中导出 `im2col` 和 `col2im`，支持外部直接调用。

## 涉及文件

- `core/tensor.py`
- `core/node.py`
- `train/opt.py`
- `ops/convolution.py`
- `__init__.py`
- `tests/test_tensor.py`
- `tests/test_convolution.py`

## 验证结果

已通过以下测试：

- `python -m unittest tests.test_tensor`
- `python -m unittest tests.test_convolution`
- `python -m unittest tests.test_framework`
- `python -m unittest tests.test_digits_conv`

## 结果说明

本次更新后，MyFlows 在不破坏现有 API 的前提下获得了统一的张量封装入口，同时卷积与池化算子具备了更标准、更清晰的 `im2col/col2im` 数据路径，为后续继续优化性能和扩展后端打下了基础。
