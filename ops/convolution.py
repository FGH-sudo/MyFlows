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
    def __init__(self, x, kernel, stride=1, padding=0, groups=1, dilation=1):
        # x: 图像, kernel: 卷积核
        super().__init__(x, kernel)
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

    def forward(self, x_val, kernel_val):
        # x_val: (N, C_in, H, W), kernel_val: (C_out, C_in, kH, kW)
        N, C_in, H, W = x_val.shape
        C_out, _, kH, kW = kernel_val.shape

        self._validate_kernel(x_val, kernel_val)

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

    def backward(self):
        x_node, kernel_node = self.parents
        if x_node.grad is None:
            x_node.clear_grad()
        if kernel_node.grad is None:
            kernel_node.clear_grad()

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


class ConvTranspose2D_Op(Node):
    def __init__(self, x, kernel, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
        super().__init__(x, kernel)
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

    def forward(self, x_val, kernel_val):
        batch_size, in_channels, input_h, input_w = x_val.shape
        kernel_in_channels, out_channels_per_group, kernel_h, kernel_w = kernel_val.shape

        self._validate_kernel(x_val, kernel_val)

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
        in_channels_per_group = in_channels // self.groups

        for group_index in range(self.groups):
            in_start, in_end = _group_bounds(in_channels, self.groups, group_index)
            out_start = group_index * out_channels_per_group
            out_end = out_start + out_channels_per_group

            for batch_index in range(batch_size):
                for in_channel in range(in_start, in_end):
                    for input_row in range(input_h):
                        base_row = input_row * self.stride[0] - self.padding[0]
                        for input_col in range(input_w):
                            input_value = x_val[batch_index, in_channel, input_row, input_col]
                            base_col = input_col * self.stride[1] - self.padding[1]
                            if input_value == 0:
                                continue

                            for kernel_row in range(kernel_h):
                                out_row = base_row + kernel_row * self.dilation[0]
                                if out_row < 0 or out_row >= out_h:
                                    continue

                                for kernel_col in range(kernel_w):
                                    out_col = base_col + kernel_col * self.dilation[1]
                                    if out_col < 0 or out_col >= out_w:
                                        continue

                                    output[batch_index, out_start:out_end, out_row, out_col] += (
                                        input_value * kernel_val[in_channel, :, kernel_row, kernel_col]
                                    )

        self.value = output

    def backward(self):
        x_node, kernel_node = self.parents
        if x_node.grad is None:
            x_node.clear_grad()
        if kernel_node.grad is None:
            kernel_node.clear_grad()

        batch_size, in_channels, input_h, input_w = x_node.value.shape
        _, out_channels_per_group, kernel_h, kernel_w = kernel_node.value.shape

        for group_index in range(self.groups):
            in_start, in_end = _group_bounds(in_channels, self.groups, group_index)
            out_start = group_index * out_channels_per_group
            out_end = out_start + out_channels_per_group

            for batch_index in range(batch_size):
                for in_channel in range(in_start, in_end):
                    for input_row in range(input_h):
                        base_row = input_row * self.stride[0] - self.padding[0]
                        for input_col in range(input_w):
                            input_value = x_node.value[batch_index, in_channel, input_row, input_col]
                            base_col = input_col * self.stride[1] - self.padding[1]

                            for kernel_row in range(kernel_h):
                                out_row = base_row + kernel_row * self.dilation[0]
                                if out_row < 0 or out_row >= self.grad.shape[2]:
                                    continue

                                for kernel_col in range(kernel_w):
                                    out_col = base_col + kernel_col * self.dilation[1]
                                    if out_col < 0 or out_col >= self.grad.shape[3]:
                                        continue

                                    grad_slice = self.grad[batch_index, out_start:out_end, out_row, out_col]
                                    x_node.grad[batch_index, in_channel, input_row, input_col] += np.dot(
                                        grad_slice,
                                        kernel_node.value[in_channel, :, kernel_row, kernel_col],
                                    )
                                    kernel_node.grad[in_channel, :, kernel_row, kernel_col] += (
                                        input_value * grad_slice
                                    )


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
