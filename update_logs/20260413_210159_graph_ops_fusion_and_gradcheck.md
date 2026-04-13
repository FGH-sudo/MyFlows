# MyFlows 更新日志

- 时间戳：2026-04-13 21:01:59（本地生成）
- 主题：计算图优化、算子融合、转置卷积加速与自动求导校验

## 背景与目标

本次更新围绕以下目标展开：

- 保持 MyFlows 的**动态图构建**方式不变（按前向即时构图）。
- 在不改变数值语义的前提下，降低 Python 调度/节点数量，提升执行效率。
- 为新加入/融合后的算子提供**梯度正确性验证**（数值梯度对比）。

## 主要变更

### 1) 转置卷积 `ConvTranspose2D_Op` 由深层循环优化为批量 GEMM + scatter

- **位置**：`ops/convolution.py`
- **内容**：
  - 将原转置卷积（forward/backward）深层 Python 循环替换为：
    - 仅在 `kH × kW` 上循环
    - 使用矩阵乘（GEMM）批量完成通道混合
    - 使用 `np.add.at` / 累加语义进行 scatter 回写
  - 修复边界情况：当 `input_h == 1` 或 `input_w == 1` 时的广播索引越界问题（显式构造 `(H,W)` 网格）。
- **收益**：
  - 大幅减少 Python 层循环次数，转置卷积可用性与速度明显提升。

### 2) 卷积/转置卷积 bias 融合：减少 `Add` 节点

- **位置**：`ops/convolution.py`, `layers/layer.py`
- **内容**：
  - `Conv2D_Op` 增加 `bias` 可选父节点，并在算子内部完成 `+ bias`（形状约束 `(C_out,)`）。
  - `ConvTranspose2D_Op` 同样支持 `bias` 并在内部加偏置。
  - `layers/Conv2D` 与 `layers/ConvTranspose2D` 构图时直接传入 `bias=self.b`，不再通过外层 `Add` 广播实现偏置。
- **收益**：
  - 每个卷积层减少一个 `Add` 节点与一次调度，同时减少中间张量与反向 broadcast 处理。

### 3) 新增激活融合算子：Conv/Deconv + {ReLU, LeakyReLU}

- **位置**：`ops/convolution.py`, `core/graph_opt.py`
- **新增融合算子**：
  - `Conv2D_ReLU_Op`
  - `ConvTranspose2D_ReLU_Op`
  - `Conv2D_LeakyReLU_Op`（保留 `alpha`）
  - `ConvTranspose2D_LeakyReLU_Op`（保留 `alpha`）
- **实现方式**：
  - forward 先执行卷积本体，再做激活；
  - backward 先对上游梯度做 mask/门控，再调用卷积 backward。

### 4) 计算图优化入口：常量折叠 + 算子融合（可开关）

- **位置**：`core/graph.py`, `core/graph_opt.py`, `core/node.py`
- **内容**：
  - `Variable` 增加 `constant` 标记：`Variable(..., constant=True)` 表示运行期不参与更新且可被折叠。
  - 引入 `core/graph_opt.py`，包含构建期优化：
    - **常量折叠**：`Add(constant, constant)` → 单个 `Variable(constant=True)`，支持链式多轮折叠；根节点被折叠时会返回新根引用。
    - **线性融合**：`Add(MatMul(x, W), b)` / `Add(b, MatMul(x, W))` → `Linear(x, W, b)`。
    - **卷积激活融合**：`{ReLU, LeakyReLU}(Conv/Deconv)` → 对应融合算子。
  - `Graph` 新增控制参数（保持默认行为不变）：
    - `optimize`：是否运行优化 pass
    - `fold_constants`：是否常量折叠
    - `fuse_linear`：是否线性融合
    - `fuse_activations`：是否激活融合（同时控制 ReLU 与 LeakyReLU）
- **统计信息**：
  - 优化统计保存在 `Graph._opt_stats` 中，如 `constant_folds`、`linear_fusions`、`conv_act_fusions`。

### 5) 构图阶段（Layer 内）直接融合 ReLU：用户可控

- **位置**：`layers/layer.py`
- **内容**：
  - `Conv2D`、`ConvTranspose2D` 新增 `fuse_activation=True`。
  - 当 `activation=ReLU` 且 `fuse_activation=True` 时，层在构图阶段直接生成：
    - `Conv2D_ReLU_Op` 或 `ConvTranspose2D_ReLU_Op`
  - 若关闭 `fuse_activation`，仍可通过 `Graph(optimize=True, fuse_activations=True)` 在图优化阶段进行融合。
- **收益**：
  - 对 ReLU 常见路径减少一个节点，且不依赖图优化 pass；同时保留用户选择权。

### 6) 新增 `Linear` 融合算子，更新 `Dense` 默认使用

- **位置**：`ops/basic.py`, `layers/layer.py`
- **内容**：
  - 新增 `Linear`：实现 `y = x @ W + b`，在 backward 中正确处理 bias 广播梯度。
  - `Dense.forward` 由 `Add(MatMul(...), b)` 改为 `Linear(...)`。

### 7) 自动求导（反向传播）验证：数值梯度 gradcheck

- **位置**：`tests/test_gradcheck_autodiff.py`（新增）
- **验证方式**：
  - 使用 finite difference \( (f(x+\epsilon)-f(x-\epsilon)) / (2\epsilon) \) 对比 `backward()` 的梯度。
  - 在小尺寸、`float64` 下对如下梯度做校验：
    - `Linear`: `dW`, `db`
    - `Conv2D_*`: `dKernel`, `dbias`
    - `ConvTranspose2D_*`: `dKernel`, `dbias`
- **目的**：
  - 为算子融合后的反传提供“可复现的正确性证据”。

## 新增/修改文件列表（核心）

- **新增**
  - `core/graph_opt.py`
  - `tests/test_graph_opt.py`（本次增加更多融合相关用例）
  - `tests/test_gradcheck_autodiff.py`
- **修改**
  - `core/graph.py`（优化开关与入口）
  - `core/node.py`（`Variable.constant`）
  - `ops/basic.py`（新增 `Linear`）
  - `ops/convolution.py`（Conv/Deconv 加速、bias 融合、ReLU/LeakyReLU 融合算子）
  - `layers/layer.py`（Conv/Deconv 构图融合 bias 与 ReLU；Dense 默认用 Linear）
  - `__init__.py`（导出 `apply_graph_optimizations`）

## 使用示例（关键开关）

- 仅构图（保持旧行为）：
  - `Graph(loss)`
- 开启图优化（包含融合与折叠）：
  - `Graph(loss, optimize=True, fold_constants=True)`
- 关闭激活融合（ReLU/LeakyReLU 均不融合）：
  - `Graph(loss, optimize=True, fuse_activations=False)`
- 关闭线性融合：
  - `Graph(loss, optimize=True, fuse_linear=False)`
- 层级控制 ReLU 构图融合：
  - `Conv2D(..., activation=ReLU, fuse_activation=False)`（禁用构图即融合）

## 验证结果

已通过：

- `python -m unittest discover -s tests -p "test_*.py" -q`
- `python -m unittest tests.test_gradcheck_autodiff -q`

备注：

- `basedpyright` 对测试文件中 `MyFlows.*` 导入的告警属于静态路径配置问题，不影响运行时测试结果。

