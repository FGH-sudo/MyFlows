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


def _build_window_indices(channels, kernel_size, output_size, stride):
    k_h, k_w = kernel_size
    out_h, out_w = output_size
    stride_h, stride_w = stride

    i0 = np.tile(np.repeat(np.arange(k_h), k_w), channels)
    j0 = np.tile(np.arange(k_w), k_h * channels)
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


class Conv2D_Op(Node):
    def __init__(self, x, kernel, stride=1, padding=0):
        # x: 图像, kernel: 卷积核
        super().__init__(x, kernel)
        self.stride = _normalize_pair(stride, "stride")
        self.padding = _normalize_pair(padding, "padding", allow_zero=True)

    def forward(self, x_val, kernel_val):
        # x_val: (N, C_in, H, W), kernel_val: (C_out, C_in, kH, kW)
        N, C_in, H, W = x_val.shape
        C_out, _, kH, kW = kernel_val.shape

        if kernel_val.shape[1] != C_in:
            raise ValueError("kernel input channels must match x input channels")

        pad_h, pad_w = self.padding
        stride_h, stride_w = self.stride
        H_pad = H + 2 * pad_h
        W_pad = W + 2 * pad_w
        out_H = (H_pad - kH) // stride_h + 1
        out_W = (W_pad - kW) // stride_w + 1
        if out_H <= 0 or out_W <= 0:
            raise ValueError("kernel size is larger than the padded input")

        self.x_pad = np.pad(
            x_val,
            ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
            mode="constant",
        )
        self.k_idx, self.i_idx, self.j_idx = _build_window_indices(
            C_in,
            (kH, kW),
            (out_H, out_W),
            self.stride,
        )
        self.cols = _extract_patches(self.x_pad, self.k_idx, self.i_idx, self.j_idx)
        self.kernel_cols = kernel_val.reshape(C_out, -1)

        out = self.cols @ self.kernel_cols.T
        self.value = out.reshape(N, out_H, out_W, C_out).transpose(0, 3, 1, 2)

    def backward(self):
        x_node, kernel_node = self.parents
        if x_node.grad is None:
            x_node.clear_grad()
        if kernel_node.grad is None:
            kernel_node.clear_grad()

        grad_y = self.grad.transpose(0, 2, 3, 1).reshape(-1, kernel_node.value.shape[0])
        kernel_node.grad += (grad_y.T @ self.cols).reshape(kernel_node.value.shape)

        grad_cols = grad_y @ self.kernel_cols
        grad_x_pad = _scatter_patches(grad_cols, self.x_pad.shape, self.k_idx, self.i_idx, self.j_idx)

        pad_h, pad_w = self.padding
        if pad_h == 0 and pad_w == 0:
            x_node.grad += grad_x_pad
        else:
            x_node.grad += grad_x_pad[
                :,
                :,
                pad_h:pad_h + x_node.value.shape[2],
                pad_w:pad_w + x_node.value.shape[3],
            ]


class MaxPool2d_Op(Node):
    def __init__(self, x, kernel_size=2, stride=2):
        super().__init__(x)
        self.kernel_size = _normalize_pair(kernel_size, "kernel_size")
        self.stride = _normalize_pair(stride, "stride")

    def forward(self, x_val):
        N, C, H, W = x_val.shape
        k_h, k_w = self.kernel_size
        s_h, s_w = self.stride
        out_H = (H - k_h) // s_h + 1
        out_W = (W - k_w) // s_w + 1
        if out_H <= 0 or out_W <= 0:
            raise ValueError("kernel size is larger than the input")

        self.k_idx, self.i_idx, self.j_idx = _build_window_indices(
            C,
            self.kernel_size,
            (out_H, out_W),
            self.stride,
        )
        patches = _extract_patches(x_val, self.k_idx, self.i_idx, self.j_idx)
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
        x_node.grad += _scatter_patches(grad_rows, x_node.value.shape, self.k_idx, self.i_idx, self.j_idx)


class Flatten_Op(Node):
    def forward(self, x):
        self.in_shape = x.shape
        self.value = x.reshape(x.shape[0], -1)
    def backward(self):
        self.parents[0].grad += self.grad.reshape(self.in_shape)

