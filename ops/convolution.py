import numpy as np
from ..core.node import Node


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

    i0 = np.tile(np.repeat(np.arange(k_h) * dilation_h, k_w), channels)
    j0 = np.tile(np.arange(k_w) * dilation_w, k_h * channels)
    k = np.repeat(np.arange(channels), k_h * k_w).reshape(-1, 1)

    i1 = (np.repeat(np.arange(out_h), out_w) * stride_h).reshape(1, -1)
    j1 = (np.tile(np.arange(out_w), out_h) * stride_w).reshape(1, -1)

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
    output = np.zeros(output_shape, dtype=rows.dtype)
    batch_idx = np.arange(batch_size)[:, None, None]
    np.add.at(output, (batch_idx, k_idx[None, :, :], i_idx[None, :, :], j_idx[None, :, :]), cols)
    return output


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
    x_pad = np.pad(
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
    def __init__(self, x, kernel, stride=1, padding=0, groups=1, dilation=1, bias=None):
        # x: 图像, kernel: 卷积核；可选 bias 与输出通道对齐，前向融合加偏置（减少 Add 节点）
        if bias is None:
            super().__init__(x, kernel)
        else:
            super().__init__(x, kernel, bias)
        self.bias = bias
        self.stride = _normalize_pair(stride, "stride")
        self.padding = _normalize_pair(padding, "padding", allow_zero=True)
        self.groups = _normalize_groups(groups)
        self.dilation = _normalize_pair(dilation, "dilation")
        self._im2col_contexts = []

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
        # x_val: (N, C_in, H, W), kernel_val: (C_out, C_in, kH, kW)
        N, C_in, H, W = x_val.shape
        C_out, _, kH, kW = kernel_val.shape

        self._validate_kernel(x_val, kernel_val)
        if self.bias is not None and bias_val is None:
            raise ValueError("bias was set on Conv2D_Op but forward received no bias value")
        if self.bias is not None:
            if bias_val.shape != (C_out,):
                raise ValueError("bias shape must be (C_out,)")

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

        self.value = np.concatenate(outputs, axis=1)
        if self.bias is not None:
            self.value = self.value + bias_val.reshape(1, C_out, 1, 1)

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
        out_channels = kernel_node.value.shape[0]
        out_channels_per_group = out_channels // self.groups

        for group_index in range(self.groups):
            in_start, in_end = _group_bounds(in_channels, self.groups, group_index)
            out_start, out_end = _group_bounds(out_channels, self.groups, group_index)

            grad_y = self.grad[:, out_start:out_end].transpose(0, 2, 3, 1).reshape(-1, out_channels_per_group)
            kernel_node.grad[out_start:out_end] += (
                grad_y.T @ self.cols[group_index]
            ).reshape(kernel_node.value[out_start:out_end].shape)

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
            bias_node.grad += np.sum(self.grad, axis=(0, 2, 3))


class Conv2D_ReLU_Op(Conv2D_Op):
    """融合算子：Conv2D(+bias) 后接 ReLU。"""

    def forward(self, x_val, kernel_val, bias_val=None):
        super().forward(x_val, kernel_val, bias_val=bias_val)
        # 使用 pre-activation 的符号生成 mask，避免数值差异
        self._relu_mask = self.value > 0
        self.value = np.maximum(0, self.value)

    def backward(self):
        # 先过 ReLU 的梯度门控，再按 Conv2D_Op 反传
        self.grad = self.grad * self._relu_mask
        super().backward()


class Conv2D_LeakyReLU_Op(Conv2D_Op):
    """融合算子：Conv2D(+bias) 后接 LeakyReLU。"""

    def __init__(self, *args, alpha=0.01, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = float(alpha)

    def forward(self, x_val, kernel_val, bias_val=None):
        super().forward(x_val, kernel_val, bias_val=bias_val)
        self._leaky_mask = self.value > 0
        self.value = np.where(self._leaky_mask, self.value, self.alpha * self.value)

    def backward(self):
        self.grad = self.grad * np.where(self._leaky_mask, 1.0, self.alpha)
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

    ih_grid = np.arange(input_h, dtype=np.int64)[:, None]
    iw_grid = np.arange(input_w, dtype=np.int64)[None, :]

    for kr in range(kernel_h):
        for kc in range(kernel_w):
            oh = ih_grid * stride_h - pad_h + kr * dilation_h + 0 * iw_grid
            ow = iw_grid * stride_w - pad_w + kc * dilation_w + 0 * ih_grid
            mask = (oh >= 0) & (oh < out_h) & (ow >= 0) & (ow < out_w)
            ih_valid, iw_valid = np.where(mask)
            if ih_valid.size == 0:
                continue

            oo_h = oh[ih_valid, iw_valid].astype(np.intp, copy=False)
            oo_w = ow[ih_valid, iw_valid].astype(np.intp, copy=False)
            v = oo_h.size

            xv = x_group[:, :, ih_valid, iw_valid]
            ks = kernel_group[:, :, kr, kc]
            contrib = xv.swapaxes(1, 2) @ ks

            flat = np.arange(batch_size * v * out_channels_per_group, dtype=np.intp)
            n_idx = flat // (v * out_channels_per_group)
            rem = flat % (v * out_channels_per_group)
            v_idx = rem // out_channels_per_group
            g_idx = rem % out_channels_per_group
            c_idx = out_start + g_idx
            oh_idx = oo_h[v_idx]
            ow_idx = oo_w[v_idx]

            np.add.at(output, (n_idx, c_idx, oh_idx, ow_idx), contrib.ravel())


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

    ih_grid = np.arange(input_h, dtype=np.int64)[:, None]
    iw_grid = np.arange(input_w, dtype=np.int64)[None, :]

    for kr in range(kernel_h):
        for kc in range(kernel_w):
            oh = ih_grid * stride_h - pad_h + kr * dilation_h + 0 * iw_grid
            ow = iw_grid * stride_w - pad_w + kc * dilation_w + 0 * ih_grid
            mask = (oh >= 0) & (oh < out_h) & (ow >= 0) & (ow < out_w)
            ih_valid, iw_valid = np.where(mask)
            if ih_valid.size == 0:
                continue

            oo_h = oh[ih_valid, iw_valid]
            oo_w = ow[ih_valid, iw_valid]

            xv = x_group[:, :, ih_valid, iw_valid]
            gv = grad_group[:, :, oo_h, oo_w]

            ks = kernel_group[:, :, kr, kc]
            grad_x_group[:, :, ih_valid, iw_valid] += np.einsum("cg,ngv->ncv", ks, gv)
            grad_kernel_group[:, :, kr, kc] += np.einsum("ncv,ngv->cg", xv, gv)


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

        output = np.zeros((batch_size, out_channels, out_h, out_w), dtype=x_val.dtype)

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
            bias_node.grad += np.sum(self.grad, axis=(0, 2, 3))


class ConvTranspose2D_ReLU_Op(ConvTranspose2D_Op):
    """融合算子：ConvTranspose2D(+bias) 后接 ReLU。"""

    def forward(self, x_val, kernel_val, bias_val=None):
        super().forward(x_val, kernel_val, bias_val=bias_val)
        self._relu_mask = self.value > 0
        self.value = np.maximum(0, self.value)

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
        self.value = np.where(self._leaky_mask, self.value, self.alpha * self.value)

    def backward(self):
        self.grad = self.grad * np.where(self._leaky_mask, 1.0, self.alpha)
        super().backward()


class MaxPool2d_Op(Node):
    def __init__(self, x, kernel_size=2, stride=2):
        super().__init__(x)
        self.kernel_size = _normalize_pair(kernel_size, "kernel_size")
        self.stride = _normalize_pair(stride, "stride")
        self._im2col_context = None

    def forward(self, x_val):
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
        max_vals = np.max(patches, axis=-1)
        self.max_mask = patches == max_vals[..., None]
        self.value = max_vals.reshape(N, C, out_H, out_W)

    def backward(self):
        x_node = self.parents[0]
        if x_node.grad is None:
            x_node.clear_grad()

        N, C, out_H, out_W = self.value.shape
        grad = self.grad.reshape(N, C, out_H * out_W)
        grad_patches = self.max_mask * grad[..., None]
        grad_rows = grad_patches.transpose(0, 2, 1, 3).reshape(-1, C * self.kernel_size[0] * self.kernel_size[1])
        grad_x, self._im2col_context = col2im(
            grad_rows,
            x_node.value.shape,
            self.kernel_size,
            stride=self.stride,
            padding=0,
            dilation=1,
            context=self._im2col_context,
        )
        x_node.grad += grad_x


class Flatten_Op(Node):
    def forward(self, x):
        self.in_shape = x.shape
        self.value = x.reshape(x.shape[0], -1)
    def backward(self):
        self.parents[0].grad += self.grad.reshape(self.in_shape)
