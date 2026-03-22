# MyFlows 更新日志

- 时间戳：2026-03-23 00:44:59 CST (+0800)
- 主题：高级卷积扩展与多尺度卷积教学支持

## 本次更新内容

### 1. 扩展 `Conv2D_Op` 的卷积表达能力

- 在 `ops/convolution.py` 中为 `Conv2D_Op` 增加 `groups` 与 `dilation` 参数。
- 将 `im2col/col2im` 的上下文构造扩展为支持扩张卷积。
- 新增 `effective_kernel_size()`，用于显式计算扩张后的有效感受野大小。

### 2. 新增转置卷积算子

- 新增 `ConvTranspose2D_Op`，支持 `stride`、`padding`、`output_padding`、`groups` 和 `dilation`。
- 保持与当前框架风格一致：卷积本体负责线性变换，偏置仍由 `Add` 广播完成。

### 3. 在层接口中加入高级卷积入口

- 扩展 `layers/layer.py` 中的 `Conv2D`，使其原生支持 `groups` 和 `dilation`。
- 新增以下层：
  - `GroupedConv2D`
  - `DepthwiseConv2D`
  - `DepthwiseSeparableConv2D`
  - `DilatedConv2D`
  - `ConvTranspose2D`
- `DepthwiseSeparableConv2D` 采用“depthwise + 1x1 pointwise”组合实现，直接体现深度可分离卷积的结构本质。

### 4. 为 `1/3/5/31` 卷积核尺寸加入教学型验证

- 新增测试覆盖 `1x1 / 3x3 / 5x5 / 31x31` 卷积核的形状与输出行为。
- 在测试中同时验证扩张卷积的有效感受野：
  - `3x3, dilation=2 -> effective 5x5`
  - `5x5, dilation=7 -> effective 29x29`
- 这使得“大核卷积”和“扩张卷积放大感受野”之间的关系在框架中可直接验证。

## 概念说明

- `1x1` 卷积：不聚合空间邻域，主要做通道重组与压缩/升维。
- `3x3` 卷积：局部建模与计算量之间的常规平衡点。
- `5x5` 卷积：覆盖更大邻域，但参数量和计算量明显上升。
- `31x31` 卷积：适合强调大范围上下文，但代价很高，常与深度卷积或分解策略结合。
- 分组卷积：将通道分组后独立卷积，降低参数量并形成子空间特征学习。
- 深度可分离卷积：先做逐通道空间卷积，再用 `1x1` 卷积混合通道，常用于轻量化网络。
- 扩张卷积：在不直接增大参数量的前提下扩大感受野。
- 转置卷积：通过可学习上采样恢复更高分辨率特征图，常见于解码器、生成器和分割网络。

## 涉及文件

- `ops/convolution.py`
- `layers/layer.py`
- `__init__.py`
- `tests/test_advanced_convolution.py`

## 验证结果

已通过以下测试：

- `python -m unittest tests.test_convolution`
- `python -m unittest tests.test_advanced_convolution`
- `python -m unittest tests.test_framework`
- `python -m unittest tests.test_digits_conv`
- `python -m unittest tests.test_tensor`

## 结果说明

本次更新后，MyFlows 已从“基础二维卷积框架”扩展为可直接演示和使用多类经典卷积变体的教学型框架。卷积尺寸、分组策略、感受野扩张和可学习上采样现在都能在统一 API 下表达，并配有对应测试作为概念落点。
