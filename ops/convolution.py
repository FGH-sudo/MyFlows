from functools import cache

from ..core.device import _lazy_cupy, xp, get_device
from ..core.node import Node


# 原生 CUDA 后端 -> GEMM 引擎：im2col/col2im、尾处理与池化都是项目自己的 kernel，卷积的三个 GEMM 用
# 第 19 讲的寄存器分块 kernel（register）或 cuBLAS（cublas）
NATIVE_BACKENDS = {"cuda_native_register": "register", "cuda_native_cublas": "cublas"}


def _backend_name(backend):
    if backend not in ("auto", "numpy", "cupy", *NATIVE_BACKENDS):
        raise ValueError("backend must be auto, numpy, cupy, cuda_native_register or cuda_native_cublas")
    return backend


def _resolve_backend(backend, *arrays):
    device = get_device()
    actual = ("numpy" if device == "cpu" else "cupy") if backend == "auto" else backend
    if (actual == "numpy") != (device == "cpu"):
        raise ValueError(f"backend={actual} is incompatible with device={device}")
    for value in arrays:
        if value is not None and not isinstance(value, xp.ndarray):
            raise TypeError("operator arrays must match the active device")
        if value is not None and device == "cuda" and value.device.id != xp.cuda.runtime.getDevice():
            raise ValueError("operator arrays must be on the current CUDA device")
    return actual


def _depthwise_windows(shape, kernel_size, stride, padding, dilation):
    """Output size and, per kernel tap (i, j), the strided slices of the padded input it multiplies."""
    _, _, h, w = shape
    (kh, kw), (sh, sw), (ph, pw), (dh, dw) = kernel_size, stride, padding, dilation
    out_h = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    out_w = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    taps = [(i, j, slice(i * dh, i * dh + sh * (out_h - 1) + 1, sh), slice(j * dw, j * dw + sw * (out_w - 1) + 1, sw))
            for i in range(kh) for j in range(kw)]
    return (out_h, out_w), taps


def depthwise_conv2d_forward(x, kernel, stride, padding, dilation):
    """Depthwise convolution (groups == C_in == C_out): one strided multiply-add per kernel tap.

    The grouped im2col path runs one GEMM per channel (up to 960 in MobileNetV2); this one touches only
    activation-sized arrays. Returns ``(y, x_padded)``.
    """
    ph, pw = padding
    x_pad = xp.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw))) if ph or pw else x
    (out_h, out_w), taps = _depthwise_windows(x.shape, kernel.shape[2:], stride, padding, dilation)
    y = xp.zeros((x.shape[0], x.shape[1], out_h, out_w), dtype=x.dtype)
    for i, j, rows, cols in taps:
        y += x_pad[:, :, rows, cols] * kernel[:, 0, i, j].reshape(1, -1, 1, 1)
    return y, x_pad


def depthwise_conv2d_backward(x_pad, kernel, grad_y, input_shape, stride, padding, dilation, *, need_input_grad):
    """Kernel gradient (FP64 accumulation, like the other reductions) and, optionally, the input gradient."""
    _, taps = _depthwise_windows(input_shape, kernel.shape[2:], stride, padding, dilation)
    grad_k = xp.zeros(kernel.shape, dtype=kernel.dtype)
    grad_pad = xp.zeros(x_pad.shape, dtype=grad_y.dtype) if need_input_grad else None
    for i, j, rows, cols in taps:
        grad_k[:, 0, i, j] = (x_pad[:, :, rows, cols] * grad_y).sum(axis=(0, 2, 3), dtype=xp.float64)
        if need_input_grad:
            grad_pad[:, :, rows, cols] += grad_y * kernel[:, 0, i, j].reshape(1, -1, 1, 1)
    if grad_pad is None:
        return grad_k, None
    ph, pw = padding
    h, w = input_shape[2:]
    return grad_k, grad_pad[:, :, ph:ph + h, pw:pw + w]


def _needs_grad(node):
    # 输入图像、标签等 requires_grad=False 的叶子变量不需要梯度，可以跳过 dX 的 GEMM 与 col2im
    return getattr(node, "requires_grad", True)


def _normalize_pair(value, name, allow_zero=False):
    if isinstance(value, int):
        pair = (value, value)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        pair = (int(value[0]), int(value[1]))
    else:
        raise TypeError(f"{name} must be an int or a pair of ints, got {value!r}")

    minimum = 0 if allow_zero else 1
    if pair[0] < minimum or pair[1] < minimum:
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {relation}, got {value!r}")
    return pair


def _normalize_groups(groups):
    groups = int(groups)
    if groups <= 0:
        raise ValueError(f"groups must be positive, got {groups}")
    return groups


def effective_kernel_size(kernel_size, dilation=1):
    kernel_h, kernel_w = _normalize_pair(kernel_size, "kernel_size")
    dilation_h, dilation_w = _normalize_pair(dilation, "dilation")
    return (
        (kernel_h - 1) * dilation_h + 1,
        (kernel_w - 1) * dilation_w + 1,
    )


def _infer_conv_output_size(input_size, kernel_size, stride, padding, dilation):
    input_h, input_w = input_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    effective_h, effective_w = effective_kernel_size(kernel_size, dilation)

    out_h = (input_h + 2 * pad_h - effective_h) // stride_h + 1
    out_w = (input_w + 2 * pad_w - effective_w) // stride_w + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("kernel size is larger than the padded input")
    return out_h, out_w


def _infer_transposed_output_size(input_size, kernel_size, stride, padding, dilation, output_padding):
    input_h, input_w = input_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    output_pad_h, output_pad_w = output_padding
    effective_h, effective_w = effective_kernel_size(kernel_size, dilation)

    out_h = (input_h - 1) * stride_h - 2 * pad_h + effective_h + output_pad_h
    out_w = (input_w - 1) * stride_w - 2 * pad_w + effective_w + output_pad_w
    if out_h <= 0 or out_w <= 0:
        raise ValueError("transposed convolution output size must be positive")
    return out_h, out_w


def _group_bounds(total_channels, groups, group_index):
    channels_per_group = total_channels // groups
    start = group_index * channels_per_group
    return start, start + channels_per_group


def _build_window_indices(channels, kernel_size, output_size, stride, dilation=(1, 1)):
    k_h, k_w = kernel_size
    out_h, out_w = output_size
    stride_h, stride_w = stride
    dilation_h, dilation_w = dilation

    i0 = xp.tile(xp.repeat(xp.arange(k_h) * dilation_h, k_w), channels)
    j0 = xp.tile(xp.arange(k_w) * dilation_w, k_h * channels)
    k = xp.repeat(xp.arange(channels), k_h * k_w).reshape(-1, 1)

    i1 = (xp.repeat(xp.arange(out_h), out_w) * stride_h).reshape(1, -1)
    j1 = (xp.tile(xp.arange(out_w), out_h) * stride_w).reshape(1, -1)

    i = i0.reshape(-1, 1) + i1
    j = j0.reshape(-1, 1) + j1
    return k, i, j


def _extract_patches(x, k_idx, i_idx, j_idx):
    patches = x[:, k_idx, i_idx, j_idx]
    return patches.transpose(0, 2, 1).reshape(-1, k_idx.shape[0])


def _scatter_patches(rows, output_shape, k_idx, i_idx, j_idx):
    batch_size = output_shape[0]
    num_features = k_idx.shape[0]
    num_positions = i_idx.shape[1]

    cols = rows.reshape(batch_size, num_positions, num_features).transpose(0, 2, 1)
    output = xp.zeros(output_shape, dtype=rows.dtype)
    batch_idx = xp.arange(batch_size)[:, None, None]
    xp.add.at(output, (batch_idx, k_idx[None, :, :], i_idx[None, :, :], j_idx[None, :, :]), cols)
    return output


@cache
def _col2im_gather_kernel():
    # 每个输入像素按固定顺序收集覆盖它的所有窗口位置：结果确定（xp.add.at 在 CuPy 上是原子加，
    # 累加顺序随调度变化），也省去全零缓冲、padding 区域与切片拷贝
    return _lazy_cupy().ElementwiseKernel(
        "raw T rows, int32 C, int32 H, int32 W, int32 KH, int32 KW, int32 SH, int32 SW, "
        "int32 PH, int32 PW, int32 DH, int32 DW, int32 OH, int32 OW",
        "T out",
        """
        long long t = i;
        const int w = (int)(t % W); t /= W;
        const int h = (int)(t % H); t /= H;
        const int c = (int)(t % C);
        const long long n = t / C;
        const long long features = (long long)C * KH * KW;
        T acc = 0;
        for (int ki = 0; ki < KH; ++ki) {
            const int hp = h + PH - ki * DH;
            if (hp < 0 || hp % SH != 0 || hp / SH >= OH) continue;
            for (int kj = 0; kj < KW; ++kj) {
                const int wp = w + PW - kj * DW;
                if (wp < 0 || wp % SW != 0 || wp / SW >= OW) continue;
                acc += rows[(n * OH * OW + (long long)(hp / SH) * OW + wp / SW) * features + (c * KH + ki) * KW + kj];
            }
        }
        out = acc;
        """,
        "mf_col2im_gather")


def _build_im2col_context(input_shape, kernel_size, stride, padding, dilation=(1, 1), context=None):
    channels, height, width = input_shape[1:]
    signature = (channels, height, width, kernel_size, stride, padding, dilation)
    if context is not None and context.get("signature") == signature:
        return context

    pad_h, pad_w = padding
    out_h, out_w = _infer_conv_output_size((height, width), kernel_size, stride, padding, dilation)

    padded_h = height + 2 * pad_h
    padded_w = width + 2 * pad_w
    k_idx, i_idx, j_idx = _build_window_indices(
        channels,
        kernel_size,
        (out_h, out_w),
        stride,
        dilation=dilation,
    )
    return {
        "signature": signature,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "dilation": dilation,
        "effective_kernel_size": effective_kernel_size(kernel_size, dilation),
        "output_size": (out_h, out_w),
        "padded_size": (padded_h, padded_w),
        "k_idx": k_idx,
        "i_idx": i_idx,
        "j_idx": j_idx,
    }


def im2col(x, kernel_size, stride=1, padding=0, dilation=1, context=None):
    kernel_size = _normalize_pair(kernel_size, "kernel_size")
    stride = _normalize_pair(stride, "stride")
    padding = _normalize_pair(padding, "padding", allow_zero=True)
    dilation = _normalize_pair(dilation, "dilation")
    context = _build_im2col_context(x.shape, kernel_size, stride, padding, dilation, context)

    pad_h, pad_w = context["padding"]
    x_pad = xp.pad(
        x,
        ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
    )
    cols = _extract_patches(x_pad, context["k_idx"], context["i_idx"], context["j_idx"])
    return cols, context, x_pad


def col2im(rows, input_shape, kernel_size, stride=1, padding=0, dilation=1, context=None):
    kernel_size = _normalize_pair(kernel_size, "kernel_size")
    stride = _normalize_pair(stride, "stride")
    padding = _normalize_pair(padding, "padding", allow_zero=True)
    dilation = _normalize_pair(dilation, "dilation")
    context = _build_im2col_context(input_shape, kernel_size, stride, padding, dilation, context)

    if get_device() == "cuda" and isinstance(rows, _lazy_cupy().ndarray):
        out_h, out_w = context["output_size"]
        output = xp.empty(tuple(input_shape), dtype=rows.dtype)
        _col2im_gather_kernel()(xp.ascontiguousarray(rows), *input_shape[1:], *kernel_size, *stride, *padding,
                                *dilation, out_h, out_w, output)
        return output, context

    padded_h, padded_w = context["padded_size"]
    output = _scatter_patches(
        rows,
        (input_shape[0], input_shape[1], padded_h, padded_w),
        context["k_idx"],
        context["i_idx"],
        context["j_idx"],
    )

    pad_h, pad_w = context["padding"]
    if pad_h == 0 and pad_w == 0:
        return output, context
    return output[:, :, pad_h:pad_h + input_shape[2], pad_w:pad_w + input_shape[3]], context


class Conv2D_Op(Node):
    def _fused_activation(self):
        # 子类在原生后端把激活融合进卷积尾处理时返回 "relu"/"leaky_relu"
        return None

    def _activation_alpha(self):
        return 0.0

    def __init__(self, x, kernel, stride=1, padding=0, groups=1, dilation=1, bias=None, backend="auto"):
        # x: 图像, kernel: 卷积核；可选 bias 与输出通道对齐，前向融合加偏置（减少 Add 节点）
        if bias is None:
            super().__init__(x, kernel)
        else:
            super().__init__(x, kernel, bias)
        self.backend = _backend_name(backend)
        if backend in NATIVE_BACKENDS:
            if groups != 1 or _normalize_pair(dilation, "dilation") != (1, 1):
                raise ValueError(f"{backend} supports only groups=1 and dilation=1")
        self.bias = bias
        self.stride = _normalize_pair(stride, "stride")
        self.padding = _normalize_pair(padding, "padding", allow_zero=True)
        self.groups = _normalize_groups(groups)
        self.dilation = _normalize_pair(dilation, "dilation")
        self._im2col_contexts = []
        self._activation_fused = False

    def _validate_kernel(self, x_val, kernel_val):
        in_channels = x_val.shape[1]
        out_channels, kernel_channels, _, _ = kernel_val.shape

        if in_channels % self.groups != 0:
            raise ValueError("x input channels must be divisible by groups")
        if out_channels % self.groups != 0:
            raise ValueError("kernel output channels must be divisible by groups")

        expected_kernel_channels = in_channels // self.groups
        if kernel_channels != expected_kernel_channels:
            raise ValueError(
                "kernel input channels must equal x input channels divided by groups"
            )

    def forward(self, x_val, kernel_val, bias_val=None):
        self.actual_backend = _resolve_backend(self.backend, x_val, kernel_val, bias_val)
        self._activation_fused = False
        if self.actual_backend in NATIVE_BACKENDS:
            if self.bias is not None and bias_val is None:
                raise ValueError("bias value is missing")
            from .cuda_native.native import conv2d_forward
            activation = self._fused_activation()
            # 输出直接是连续的 NCHW；bias 与激活在同一个尾处理 kernel 中完成
            self.value, self._cuda_im2col_cols = conv2d_forward(
                x_val, kernel_val, bias_val, stride=self.stride, padding=self.padding,
                activation=activation, alpha=self._activation_alpha(),
                engine=NATIVE_BACKENDS[self.actual_backend])
            self._activation_fused = activation is not None
            self._cuda_inputs = (x_val, kernel_val)
            return
        # x_val: (N, C_in, H, W), kernel_val: (C_out, C_in, kH, kW)
        N, C_in, H, W = x_val.shape
        C_out, _, kH, kW = kernel_val.shape

        self._validate_kernel(x_val, kernel_val)
        if self.bias is not None and bias_val is None:
            raise ValueError("bias was set on Conv2D_Op but forward received no bias value")
        if self.bias is not None:
            if bias_val.shape != (C_out,):
                raise ValueError("bias shape must be (C_out,)")

        self._depthwise = self.groups > 1 and self.groups == C_in == C_out
        if self._depthwise:
            self.value, self._depthwise_input = depthwise_conv2d_forward(
                x_val, kernel_val, self.stride, self.padding, self.dilation)
            if self.bias is not None:
                self.value = self.value + bias_val.reshape(1, C_out, 1, 1)
            return

        if len(self._im2col_contexts) != self.groups:
            self._im2col_contexts = [None] * self.groups

        in_channels_per_group = C_in // self.groups
        out_channels_per_group = C_out // self.groups
        self.cols = []
        self.kernel_cols = []
        outputs = []

        for group_index in range(self.groups):
            in_start, in_end = _group_bounds(C_in, self.groups, group_index)
            out_start, out_end = _group_bounds(C_out, self.groups, group_index)

            cols, context, _ = im2col(
                x_val[:, in_start:in_end],
                (kH, kW),
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                context=self._im2col_contexts[group_index],
            )
            self._im2col_contexts[group_index] = context
            group_kernel_cols = kernel_val[out_start:out_end].reshape(out_channels_per_group, -1)
            group_out = cols @ group_kernel_cols.T
            out_H, out_W = context["output_size"]

            self.cols.append(cols)
            self.kernel_cols.append(group_kernel_cols)
            outputs.append(
                group_out.reshape(N, out_H, out_W, out_channels_per_group).transpose(0, 3, 1, 2)
            )

        self.value = xp.concatenate(outputs, axis=1)
        if self.bias is not None:
            self.value = self.value + bias_val.reshape(1, C_out, 1, 1)

    def backward(self):
        if not hasattr(self, "actual_backend"):
            raise RuntimeError("forward must run before backward")
        _resolve_backend(self.actual_backend, self.grad, *(p.value for p in self.parents))
        if self.actual_backend in NATIVE_BACKENDS:
            from .cuda_native.native import conv2d_backward
            grads = conv2d_backward(*self._cuda_inputs, self.grad, self._cuda_im2col_cols,
                                    stride=self.stride, padding=self.padding,
                                    need_bias_grad=self.bias is not None,
                                    need_input_grad=_needs_grad(self.parents[0]),
                                    activation=self._fused_activation() if self._activation_fused else None,
                                    alpha=self._activation_alpha(), output=self.value,
                                    engine=NATIVE_BACKENDS[self.actual_backend])
            for parent, grad in zip(self.parents, grads):
                if grad is None:
                    continue
                if parent.grad is None:
                    parent.clear_grad()
                parent.grad += grad
            return
        if self.bias is None:
            x_node, kernel_node = self.parents
        else:
            x_node, kernel_node, bias_node = self.parents
        if x_node.grad is None:
            x_node.clear_grad()
        if kernel_node.grad is None:
            kernel_node.clear_grad()
        if self.bias is not None and bias_node.grad is None:
            bias_node.clear_grad()

        batch_size, in_channels, input_h, input_w = x_node.value.shape
        out_channels = kernel_node.value.shape[0]
        out_channels_per_group = out_channels // self.groups
        needs_input_grad = _needs_grad(x_node)

        if getattr(self, "_depthwise", False):
            grad_k, grad_x = depthwise_conv2d_backward(
                self._depthwise_input, kernel_node.value, self.grad, x_node.value.shape, self.stride, self.padding,
                self.dilation, need_input_grad=needs_input_grad)
            kernel_node.grad += grad_k
            if grad_x is not None:
                x_node.grad += grad_x
            if self.bias is not None:
                bias_node.grad += xp.sum(self.grad, axis=(0, 2, 3), dtype=xp.float64).astype(bias_node.grad.dtype)
            return

        for group_index in range(self.groups):
            in_start, in_end = _group_bounds(in_channels, self.groups, group_index)
            out_start, out_end = _group_bounds(out_channels, self.groups, group_index)

            grad_y = self.grad[:, out_start:out_end].transpose(0, 2, 3, 1).reshape(-1, out_channels_per_group)
            kernel_node.grad[out_start:out_end] += (
                grad_y.T @ self.cols[group_index]
            ).reshape(kernel_node.value[out_start:out_end].shape)
            if not needs_input_grad:
                continue

            grad_cols = grad_y @ self.kernel_cols[group_index]
            grad_x, context = col2im(
                grad_cols,
                (batch_size, in_end - in_start, input_h, input_w),
                kernel_node.value.shape[2:],
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                context=self._im2col_contexts[group_index],
            )
            self._im2col_contexts[group_index] = context
            x_node.grad[:, in_start:in_end] += grad_x

        if self.bias is not None:
            # FP64 累加后舍入一次，与原生后端的 bias 梯度一致
            bias_node.grad += xp.sum(self.grad, axis=(0, 2, 3), dtype=xp.float64).astype(bias_node.grad.dtype)


class Conv2D_ReLU_Op(Conv2D_Op):
    """融合算子：Conv2D(+bias) 后接 ReLU。原生后端在卷积尾处理 kernel 内完成 ReLU，
    反向门控也在 dY 转置 kernel 内完成。"""

    def _fused_activation(self):
        return "relu"

    def forward(self, x_val, kernel_val, bias_val=None):
        super().forward(x_val, kernel_val, bias_val=bias_val)
        if self._activation_fused:
            return
        # 使用 pre-activation 的符号生成 mask，避免数值差异
        self._relu_mask = self.value > 0
        self.value = xp.maximum(0, self.value)

    def backward(self):
        # 先过 ReLU 的梯度门控，再按 Conv2D_Op 反传
        if not self._activation_fused:
            self.grad = self.grad * self._relu_mask
        super().backward()


class Conv2D_LeakyReLU_Op(Conv2D_Op):
    """融合算子：Conv2D(+bias) 后接 LeakyReLU。"""

    def __init__(self, *args, alpha=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = float(alpha)

    def _fused_activation(self):
        # 只有 alpha > 0 时 y > 0 才等价于 pre-activation > 0，反向才能只用输出做门控
        return "leaky_relu" if self.alpha > 0 else None

    def _activation_alpha(self):
        return self.alpha

    def forward(self, x_val, kernel_val, bias_val=None):
        super().forward(x_val, kernel_val, bias_val=bias_val)
        if self._activation_fused:
            return
        self._leaky_mask = self.value > 0
        self.value = xp.where(self._leaky_mask, self.value, self.alpha * self.value)

    def backward(self):
        if not self._activation_fused:
            self.grad = xp.where(self._leaky_mask, self.grad, self.alpha * self.grad)
        super().backward()


def _conv_transpose2d_accumulate_group(
    x_group,
    kernel_group,
    output,
    out_start,
    out_h,
    out_w,
    stride,
    padding,
    dilation,
):
    """Scatter-add one group's transposed conv output using kh*kw outer loops and GEMM.

    x_group: (N, Cin_g, Hin, Win)
    kernel_group: (Cin_g, Cout_g, kH, kW) — slice of full kernel for this group
    output: (N, Cout, Hout, Wout), updated in-place for channels [out_start, out_start + Cout_g)
    """
    batch_size, _, input_h, input_w = x_group.shape
    _, out_channels_per_group, kernel_h, kernel_w = kernel_group.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation

    ih_grid = xp.arange(input_h, dtype=xp.int64)[:, None]
    iw_grid = xp.arange(input_w, dtype=xp.int64)[None, :]

    for kr in range(kernel_h):
        for kc in range(kernel_w):
            oh = ih_grid * stride_h - pad_h + kr * dilation_h + 0 * iw_grid
            ow = iw_grid * stride_w - pad_w + kc * dilation_w + 0 * ih_grid
            mask = (oh >= 0) & (oh < out_h) & (ow >= 0) & (ow < out_w)
            ih_valid, iw_valid = xp.where(mask)
            if ih_valid.size == 0:
                continue

            oo_h = oh[ih_valid, iw_valid].astype(xp.intp, copy=False)
            oo_w = ow[ih_valid, iw_valid].astype(xp.intp, copy=False)
            v = oo_h.size

            xv = x_group[:, :, ih_valid, iw_valid]
            ks = kernel_group[:, :, kr, kc]
            contrib = xv.swapaxes(1, 2) @ ks

            flat = xp.arange(batch_size * v * out_channels_per_group, dtype=xp.intp)
            n_idx = flat // (v * out_channels_per_group)
            rem = flat % (v * out_channels_per_group)
            v_idx = rem // out_channels_per_group
            g_idx = rem % out_channels_per_group
            c_idx = out_start + g_idx
            oh_idx = oo_h[v_idx]
            ow_idx = oo_w[v_idx]

            xp.add.at(output, (n_idx, c_idx, oh_idx, ow_idx), contrib.ravel())


def _conv_transpose2d_backward_group(
    x_group,
    grad_group,
    grad_x_group,
    grad_kernel_group,
    kernel_group,
    stride,
    padding,
    dilation,
):
    """Backward for one group; updates grad_x_group and grad_kernel_group in-place."""
    batch_size, _, input_h, input_w = x_group.shape
    _, out_channels_per_group, kernel_h, kernel_w = kernel_group.shape
    out_h, out_w = grad_group.shape[2], grad_group.shape[3]
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation

    ih_grid = xp.arange(input_h, dtype=xp.int64)[:, None]
    iw_grid = xp.arange(input_w, dtype=xp.int64)[None, :]

    for kr in range(kernel_h):
        for kc in range(kernel_w):
            oh = ih_grid * stride_h - pad_h + kr * dilation_h + 0 * iw_grid
            ow = iw_grid * stride_w - pad_w + kc * dilation_w + 0 * ih_grid
            mask = (oh >= 0) & (oh < out_h) & (ow >= 0) & (ow < out_w)
            ih_valid, iw_valid = xp.where(mask)
            if ih_valid.size == 0:
                continue

            oo_h = oh[ih_valid, iw_valid]
            oo_w = ow[ih_valid, iw_valid]

            xv = x_group[:, :, ih_valid, iw_valid]
            gv = grad_group[:, :, oo_h, oo_w]

            ks = kernel_group[:, :, kr, kc]
            grad_x_group[:, :, ih_valid, iw_valid] += xp.einsum("cg,ngv->ncv", ks, gv)
            grad_kernel_group[:, :, kr, kc] += xp.einsum("ncv,ngv->cg", xv, gv)


class ConvTranspose2D_Op(Node):
    def __init__(self, x, kernel, stride=1, padding=0, output_padding=0, groups=1, dilation=1, bias=None):
        if bias is None:
            super().__init__(x, kernel)
        else:
            super().__init__(x, kernel, bias)
        self.bias = bias
        self.stride = _normalize_pair(stride, "stride")
        self.padding = _normalize_pair(padding, "padding", allow_zero=True)
        self.output_padding = _normalize_pair(output_padding, "output_padding", allow_zero=True)
        self.groups = _normalize_groups(groups)
        self.dilation = _normalize_pair(dilation, "dilation")

        if any(
            output_pad >= max(stride_val, dilation_val)
            for output_pad, stride_val, dilation_val in zip(self.output_padding, self.stride, self.dilation)
        ):
            raise ValueError("output_padding must be smaller than max(stride, dilation)")

    def _validate_kernel(self, x_val, kernel_val):
        in_channels = x_val.shape[1]
        kernel_in_channels, out_channels_per_group, _, _ = kernel_val.shape

        if kernel_in_channels != in_channels:
            raise ValueError("kernel input channels must match x input channels")
        if in_channels % self.groups != 0:
            raise ValueError("x input channels must be divisible by groups")
        if out_channels_per_group <= 0:
            raise ValueError("kernel must provide at least one output channel per group")

    def forward(self, x_val, kernel_val, bias_val=None):
        batch_size, in_channels, input_h, input_w = x_val.shape
        kernel_in_channels, out_channels_per_group, kernel_h, kernel_w = kernel_val.shape

        self._validate_kernel(x_val, kernel_val)
        if self.bias is not None and bias_val is None:
            raise ValueError("bias was set on ConvTranspose2D_Op but forward received no bias value")

        out_channels = out_channels_per_group * self.groups
        out_h, out_w = _infer_transposed_output_size(
            (input_h, input_w),
            (kernel_h, kernel_w),
            self.stride,
            self.padding,
            self.dilation,
            self.output_padding,
        )

        output = xp.zeros((batch_size, out_channels, out_h, out_w), dtype=x_val.dtype)

        for group_index in range(self.groups):
            in_start, in_end = _group_bounds(in_channels, self.groups, group_index)
            out_start = group_index * out_channels_per_group

            x_group = x_val[:, in_start:in_end]
            kernel_group = kernel_val[in_start:in_end]
            _conv_transpose2d_accumulate_group(
                x_group,
                kernel_group,
                output,
                out_start,
                out_h,
                out_w,
                self.stride,
                self.padding,
                self.dilation,
            )

        self.value = output
        if self.bias is not None:
            if bias_val.shape != (out_channels,):
                raise ValueError("bias shape must be (C_out,)")
            self.value = self.value + bias_val.reshape(1, out_channels, 1, 1)

    def backward(self):
        if self.bias is None:
            x_node, kernel_node = self.parents
        else:
            x_node, kernel_node, bias_node = self.parents
        if x_node.grad is None:
            x_node.clear_grad()
        if kernel_node.grad is None:
            kernel_node.clear_grad()
        if self.bias is not None and bias_node.grad is None:
            bias_node.clear_grad()

        batch_size, in_channels, input_h, input_w = x_node.value.shape
        _, out_channels_per_group, kernel_h, kernel_w = kernel_node.value.shape

        for group_index in range(self.groups):
            in_start, in_end = _group_bounds(in_channels, self.groups, group_index)
            out_start = group_index * out_channels_per_group
            out_end = out_start + out_channels_per_group

            x_group = x_node.value[:, in_start:in_end]
            grad_group = self.grad[:, out_start:out_end]
            grad_x_group = x_node.grad[:, in_start:in_end]
            grad_kernel_group = kernel_node.grad[in_start:in_end]
            kernel_group = kernel_node.value[in_start:in_end]

            _conv_transpose2d_backward_group(
                x_group,
                grad_group,
                grad_x_group,
                grad_kernel_group,
                kernel_group,
                self.stride,
                self.padding,
                self.dilation,
            )

        if self.bias is not None:
            # FP64 累加后舍入一次，与原生后端的 bias 梯度一致
            bias_node.grad += xp.sum(self.grad, axis=(0, 2, 3), dtype=xp.float64).astype(bias_node.grad.dtype)


class ConvTranspose2D_ReLU_Op(ConvTranspose2D_Op):
    """融合算子：ConvTranspose2D(+bias) 后接 ReLU。"""

    def forward(self, x_val, kernel_val, bias_val=None):
        super().forward(x_val, kernel_val, bias_val=bias_val)
        self._relu_mask = self.value > 0
        self.value = xp.maximum(0, self.value)

    def backward(self):
        self.grad = self.grad * self._relu_mask
        super().backward()


class ConvTranspose2D_LeakyReLU_Op(ConvTranspose2D_Op):
    """融合算子：ConvTranspose2D(+bias) 后接 LeakyReLU。"""

    def __init__(self, *args, alpha=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = float(alpha)

    def forward(self, x_val, kernel_val, bias_val=None):
        super().forward(x_val, kernel_val, bias_val=bias_val)
        self._leaky_mask = self.value > 0
        self.value = xp.where(self._leaky_mask, self.value, self.alpha * self.value)

    def backward(self):
        # where 保持梯度 dtype；旧写法乘以 where(mask, 1.0, alpha) 会生成 float64 掩码
        self.grad = xp.where(self._leaky_mask, self.grad, self.alpha * self.grad)
        super().backward()


class MaxPool2d_Op(Node):
    """最大池化。``padding`` 按 -inf 补边（与 torch.nn.MaxPool2d 相同），补边后各后端的路径不变。"""

    def __init__(self, x, kernel_size=2, stride=2, backend="auto", padding=0):
        super().__init__(x)
        self.backend = _backend_name(backend)
        self.kernel_size = _normalize_pair(kernel_size, "kernel_size")
        self.stride = _normalize_pair(stride, "stride")
        self.padding = _normalize_pair(padding, "padding", allow_zero=True)
        self._im2col_context = None

    def forward(self, x_val):
        self.actual_backend = _resolve_backend(self.backend, x_val)
        ph, pw = self.padding
        if ph or pw:
            x_val = xp.pad(x_val, ((0, 0), (0, 0), (ph, ph), (pw, pw)), constant_values=-xp.inf)
        self._pooled_shape = x_val.shape
        if self.actual_backend in NATIVE_BACKENDS:
            from .cuda_native.native import maxpool2d_forward
            self.value, self._native_context = maxpool2d_forward(
                x_val, kernel_size=self.kernel_size, stride=self.stride)
            return
        N, C, H, W = x_val.shape
        k_h, k_w = self.kernel_size
        cols, self._im2col_context, _ = im2col(
            x_val,
            self.kernel_size,
            stride=self.stride,
            padding=0,
            dilation=1,
            context=self._im2col_context,
        )
        out_H, out_W = self._im2col_context["output_size"]
        patches = cols
        patches = patches.reshape(N, out_H * out_W, C, k_h * k_w).transpose(0, 2, 1, 3)
        max_vals = xp.max(patches, axis=-1)
        eq = patches == max_vals[..., None]
        self.max_mask = eq & (eq.cumsum(axis=-1) == 1)
        self.value = max_vals.reshape(N, C, out_H, out_W)

    def backward(self):
        if not hasattr(self, "actual_backend"):
            raise RuntimeError("forward must run before backward")
        _resolve_backend(self.actual_backend, self.grad, self.parents[0].value)
        if self.actual_backend in NATIVE_BACKENDS:
            from .cuda_native.native import maxpool2d_backward
            parent = self.parents[0]
            if parent.grad is None:
                parent.clear_grad()
            parent.grad += self._unpad(maxpool2d_backward(self.grad, self._native_context))
            return
        x_node = self.parents[0]
        if x_node.grad is None:
            x_node.clear_grad()

        N, C, out_H, out_W = self.value.shape
        grad = self.grad.reshape(N, C, out_H * out_W)
        grad_patches = self.max_mask * grad[..., None]
        grad_rows = grad_patches.transpose(0, 2, 1, 3).reshape(-1, C * self.kernel_size[0] * self.kernel_size[1])
        grad_x, self._im2col_context = col2im(
            grad_rows,
            self._pooled_shape,
            self.kernel_size,
            stride=self.stride,
            padding=0,
            dilation=1,
            context=self._im2col_context,
        )
        x_node.grad += self._unpad(grad_x)

    def _unpad(self, grad):
        ph, pw = self.padding
        if not (ph or pw):
            return grad
        return grad[:, :, ph:grad.shape[2] - ph, pw:grad.shape[3] - pw]


class Flatten_Op(Node):
    def forward(self, x):
        self.in_shape = x.shape
        self.value = x.reshape(x.shape[0], -1)
    def backward(self):
        self.parents[0].grad += self.grad.reshape(self.in_shape)
