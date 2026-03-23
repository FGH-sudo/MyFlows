import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from MyFlows.core.graph import Graph
from MyFlows.core.node import Variable
from MyFlows.layers.layer import (
    Conv2D,
    ConvTranspose2D,
    DepthwiseConv2D,
    DepthwiseSeparableConv2D,
    DilatedConv2D,
    GroupedConv2D,
)
from MyFlows.ops.convolution import Conv2D_Op, ConvTranspose2D_Op, effective_kernel_size


def parameter_count(layer):
    return int(sum(np.prod(param.value.shape) for param in layer.params))


def naive_conv_forward(x, kernel, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation
    batch_size, in_channels, height, width = x.shape
    out_channels, kernel_channels, kernel_h, kernel_w = kernel.shape
    in_channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups

    if in_channels % groups != 0:
        raise ValueError("x input channels must be divisible by groups")
    if out_channels % groups != 0:
        raise ValueError("kernel output channels must be divisible by groups")
    if kernel_channels != in_channels_per_group:
        raise ValueError("kernel input channels must match grouped input channels")

    effective_h = (kernel_h - 1) * dilation_h + 1
    effective_w = (kernel_w - 1) * dilation_w + 1
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    out_h = (height + 2 * pad_h - effective_h) // stride_h + 1
    out_w = (width + 2 * pad_w - effective_w) // stride_w + 1
    output = np.zeros((batch_size, out_channels, out_h, out_w), dtype=x.dtype)

    for group_index in range(groups):
        in_start = group_index * in_channels_per_group
        in_end = in_start + in_channels_per_group
        out_start = group_index * out_channels_per_group
        out_end = out_start + out_channels_per_group

        for out_channel in range(out_start, out_end):
            for out_row in range(out_h):
                h_start = out_row * stride_h
                for out_col in range(out_w):
                    w_start = out_col * stride_w
                    window = x_pad[
                        :,
                        in_start:in_end,
                        h_start:h_start + effective_h:dilation_h,
                        w_start:w_start + effective_w:dilation_w,
                    ]
                    output[:, out_channel, out_row, out_col] = np.sum(
                        window * kernel[out_channel],
                        axis=(1, 2, 3),
                    )
    return output


def naive_conv_backward(x, kernel, grad_y, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation
    batch_size, in_channels, height, width = x.shape
    out_channels, kernel_channels, kernel_h, kernel_w = kernel.shape
    in_channels_per_group = in_channels // groups
    out_channels_per_group = out_channels // groups
    effective_h = (kernel_h - 1) * dilation_h + 1
    effective_w = (kernel_w - 1) * dilation_w + 1

    if kernel_channels != in_channels_per_group:
        raise ValueError("kernel input channels must match grouped input channels")

    x_pad = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    grad_x_pad = np.zeros_like(x_pad)
    grad_kernel = np.zeros_like(kernel)
    out_h, out_w = grad_y.shape[2:]

    for group_index in range(groups):
        in_start = group_index * in_channels_per_group
        in_end = in_start + in_channels_per_group
        out_start = group_index * out_channels_per_group
        out_end = out_start + out_channels_per_group

        for out_row in range(out_h):
            h_start = out_row * stride_h
            for out_col in range(out_w):
                w_start = out_col * stride_w
                window = x_pad[
                    :,
                    in_start:in_end,
                    h_start:h_start + effective_h:dilation_h,
                    w_start:w_start + effective_w:dilation_w,
                ]
                for out_channel in range(out_start, out_end):
                    local_grad = grad_y[:, out_channel, out_row, out_col][:, None, None, None]
                    grad_kernel[out_channel] += np.sum(window * local_grad, axis=0)
                    for kernel_row in range(kernel_h):
                        h_index = h_start + kernel_row * dilation_h
                        for kernel_col in range(kernel_w):
                            w_index = w_start + kernel_col * dilation_w
                            grad_x_pad[:, in_start:in_end, h_index, w_index] += (
                                grad_y[:, out_channel, out_row, out_col][:, None]
                                * kernel[out_channel, :, kernel_row, kernel_col][None, :]
                            )

    if pad_h == 0 and pad_w == 0:
        return grad_x_pad, grad_kernel
    return grad_x_pad[:, :, pad_h:pad_h + height, pad_w:pad_w + width], grad_kernel


def naive_conv_transpose_forward(
    x,
    kernel,
    stride=(1, 1),
    padding=(0, 0),
    output_padding=(0, 0),
    dilation=(1, 1),
    groups=1,
):
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    output_pad_h, output_pad_w = output_padding
    dilation_h, dilation_w = dilation
    batch_size, in_channels, input_h, input_w = x.shape
    kernel_in_channels, out_channels_per_group, kernel_h, kernel_w = kernel.shape

    if kernel_in_channels != in_channels:
        raise ValueError("kernel input channels must match x input channels")
    if in_channels % groups != 0:
        raise ValueError("x input channels must be divisible by groups")

    effective_h = (kernel_h - 1) * dilation_h + 1
    effective_w = (kernel_w - 1) * dilation_w + 1
    out_h = (input_h - 1) * stride_h - 2 * pad_h + effective_h + output_pad_h
    out_w = (input_w - 1) * stride_w - 2 * pad_w + effective_w + output_pad_w
    out_channels = out_channels_per_group * groups
    output = np.zeros((batch_size, out_channels, out_h, out_w), dtype=x.dtype)
    in_channels_per_group = in_channels // groups

    for group_index in range(groups):
        in_start = group_index * in_channels_per_group
        in_end = in_start + in_channels_per_group
        out_start = group_index * out_channels_per_group
        out_end = out_start + out_channels_per_group

        for batch_index in range(batch_size):
            for in_channel in range(in_start, in_end):
                for input_row in range(input_h):
                    base_row = input_row * stride_h - pad_h
                    for input_col in range(input_w):
                        base_col = input_col * stride_w - pad_w
                        value = x[batch_index, in_channel, input_row, input_col]
                        for kernel_row in range(kernel_h):
                            out_row = base_row + kernel_row * dilation_h
                            if out_row < 0 or out_row >= out_h:
                                continue
                            for kernel_col in range(kernel_w):
                                out_col = base_col + kernel_col * dilation_w
                                if out_col < 0 or out_col >= out_w:
                                    continue
                                output[batch_index, out_start:out_end, out_row, out_col] += (
                                    value * kernel[in_channel, :, kernel_row, kernel_col]
                                )
    return output


def naive_conv_transpose_backward(
    x,
    kernel,
    grad_y,
    stride=(1, 1),
    padding=(0, 0),
    output_padding=(0, 0),
    dilation=(1, 1),
    groups=1,
):
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    dilation_h, dilation_w = dilation
    batch_size, in_channels, input_h, input_w = x.shape
    kernel_in_channels, out_channels_per_group, kernel_h, kernel_w = kernel.shape

    if kernel_in_channels != in_channels:
        raise ValueError("kernel input channels must match x input channels")
    if in_channels % groups != 0:
        raise ValueError("x input channels must be divisible by groups")

    grad_x = np.zeros_like(x)
    grad_kernel = np.zeros_like(kernel)
    in_channels_per_group = in_channels // groups

    for group_index in range(groups):
        in_start = group_index * in_channels_per_group
        in_end = in_start + in_channels_per_group
        out_start = group_index * out_channels_per_group
        out_end = out_start + out_channels_per_group

        for batch_index in range(batch_size):
            for in_channel in range(in_start, in_end):
                for input_row in range(input_h):
                    base_row = input_row * stride_h - pad_h
                    for input_col in range(input_w):
                        base_col = input_col * stride_w - pad_w
                        value = x[batch_index, in_channel, input_row, input_col]
                        for kernel_row in range(kernel_h):
                            out_row = base_row + kernel_row * dilation_h
                            if out_row < 0 or out_row >= grad_y.shape[2]:
                                continue
                            for kernel_col in range(kernel_w):
                                out_col = base_col + kernel_col * dilation_w
                                if out_col < 0 or out_col >= grad_y.shape[3]:
                                    continue
                                grad_slice = grad_y[batch_index, out_start:out_end, out_row, out_col]
                                grad_x[batch_index, in_channel, input_row, input_col] += np.dot(
                                    grad_slice,
                                    kernel[in_channel, :, kernel_row, kernel_col],
                                )
                                grad_kernel[in_channel, :, kernel_row, kernel_col] += value * grad_slice
    return grad_x, grad_kernel


class AdvancedConvolutionOpsTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(23)

    def test_kernel_size_and_dilation_control_receptive_field(self):
        x = self.rng.normal(size=(1, 4, 35, 35))

        for kernel_size in [1, 3, 5, 31]:
            layer = Conv2D(4, 6, kernel_size=kernel_size, stride=1, padding=kernel_size // 2, name=f"k{kernel_size}")
            output_node = layer(Variable(x.copy()))
            Graph(output_node).forward()

            self.assertEqual(layer.kernel.value.shape, (6, 4, kernel_size, kernel_size))
            self.assertEqual(output_node.value.shape, (1, 6, 35, 35))

        self.assertEqual(effective_kernel_size(1), (1, 1))
        self.assertEqual(effective_kernel_size(3), (3, 3))
        self.assertEqual(effective_kernel_size(5), (5, 5))
        self.assertEqual(effective_kernel_size(31), (31, 31))
        self.assertEqual(effective_kernel_size(3, dilation=2), (5, 5))
        self.assertEqual(effective_kernel_size(5, dilation=7), (29, 29))

    def test_grouped_conv2d_forward_and_backward_match_naive_reference(self):
        x = self.rng.normal(size=(2, 4, 5, 5))
        kernel = self.rng.normal(size=(6, 2, 3, 3))
        x_node = Variable(x.copy())
        kernel_node = Variable(kernel.copy(), trainable=True)
        op = Conv2D_Op(x_node, kernel_node, stride=(1, 2), padding=(1, 1), groups=2, dilation=1)
        op.forward(x_node.value, kernel_node.value)

        expected = naive_conv_forward(x, kernel, stride=(1, 2), padding=(1, 1), dilation=(1, 1), groups=2)
        self.assertTrue(np.allclose(op.value, expected, atol=1e-10))

        upstream_grad = self.rng.normal(size=op.value.shape)
        x_node.clear_grad()
        kernel_node.clear_grad()
        op.grad = upstream_grad
        op.backward()

        expected_x_grad, expected_kernel_grad = naive_conv_backward(
            x,
            kernel,
            upstream_grad,
            stride=(1, 2),
            padding=(1, 1),
            dilation=(1, 1),
            groups=2,
        )
        self.assertTrue(np.allclose(x_node.grad, expected_x_grad, atol=1e-10))
        self.assertTrue(np.allclose(kernel_node.grad, expected_kernel_grad, atol=1e-10))

    def test_dilated_conv2d_forward_and_backward_match_naive_reference(self):
        x = self.rng.normal(size=(1, 2, 6, 6))
        kernel = self.rng.normal(size=(3, 2, 2, 2))
        x_node = Variable(x.copy())
        kernel_node = Variable(kernel.copy(), trainable=True)
        op = Conv2D_Op(x_node, kernel_node, stride=1, padding=1, groups=1, dilation=2)
        op.forward(x_node.value, kernel_node.value)

        expected = naive_conv_forward(x, kernel, stride=(1, 1), padding=(1, 1), dilation=(2, 2), groups=1)
        self.assertTrue(np.allclose(op.value, expected, atol=1e-10))

        upstream_grad = self.rng.normal(size=op.value.shape)
        x_node.clear_grad()
        kernel_node.clear_grad()
        op.grad = upstream_grad
        op.backward()

        expected_x_grad, expected_kernel_grad = naive_conv_backward(
            x,
            kernel,
            upstream_grad,
            stride=(1, 1),
            padding=(1, 1),
            dilation=(2, 2),
            groups=1,
        )
        self.assertTrue(np.allclose(x_node.grad, expected_x_grad, atol=1e-10))
        self.assertTrue(np.allclose(kernel_node.grad, expected_kernel_grad, atol=1e-10))

    def test_specialized_layers_expose_expected_shapes_and_parameter_cost(self):
        grouped = GroupedConv2D(8, 16, kernel_size=3, padding=1, groups=4, name="grouped")
        depthwise = DepthwiseConv2D(8, kernel_size=5, padding=2, depth_multiplier=2, name="depthwise")
        separable = DepthwiseSeparableConv2D(8, 16, kernel_size=5, padding=2, depth_multiplier=1, name="separable")
        standard = Conv2D(8, 16, kernel_size=5, padding=2, name="standard")

        self.assertEqual(grouped.kernel.value.shape, (16, 2, 3, 3))
        self.assertEqual(depthwise.kernel.value.shape, (16, 1, 5, 5))
        self.assertLess(parameter_count(separable), parameter_count(standard))

        x = Variable(self.rng.normal(size=(2, 8, 16, 16)))
        grouped_out = grouped(x)
        separable_out = separable(Variable(x.value.copy()))
        Graph(grouped_out).forward()
        Graph(separable_out).forward()

        self.assertEqual(grouped_out.value.shape, (2, 16, 16, 16))
        self.assertEqual(separable_out.value.shape, (2, 16, 16, 16))

    def test_dilated_and_transposed_layers_preserve_expected_spatial_shapes(self):
        x = Variable(self.rng.normal(size=(1, 4, 12, 12)))

        dilated = DilatedConv2D(4, 6, kernel_size=3, padding=2, dilation=2, name="dilated")
        dilated_out = dilated(x)
        Graph(dilated_out).forward()
        self.assertEqual(dilated_out.value.shape, (1, 6, 12, 12))

        deconv = ConvTranspose2D(4, 3, kernel_size=3, stride=2, padding=1, output_padding=1, name="deconv")
        deconv_out = deconv(Variable(x.value.copy()))
        Graph(deconv_out).forward()
        self.assertEqual(deconv_out.value.shape, (1, 3, 24, 24))

    def test_convtranspose2d_forward_and_backward_match_naive_reference(self):
        x = self.rng.normal(size=(1, 2, 3, 4))
        kernel = self.rng.normal(size=(2, 3, 3, 2))
        x_node = Variable(x.copy())
        kernel_node = Variable(kernel.copy(), trainable=True)
        op = ConvTranspose2D_Op(
            x_node,
            kernel_node,
            stride=(2, 1),
            padding=(1, 0),
            output_padding=(1, 0),
            groups=1,
            dilation=1,
        )
        op.forward(x_node.value, kernel_node.value)

        expected = naive_conv_transpose_forward(
            x,
            kernel,
            stride=(2, 1),
            padding=(1, 0),
            output_padding=(1, 0),
            dilation=(1, 1),
            groups=1,
        )
        self.assertTrue(np.allclose(op.value, expected, atol=1e-10))

        upstream_grad = self.rng.normal(size=op.value.shape)
        x_node.clear_grad()
        kernel_node.clear_grad()
        op.grad = upstream_grad
        op.backward()

        expected_x_grad, expected_kernel_grad = naive_conv_transpose_backward(
            x,
            kernel,
            upstream_grad,
            stride=(2, 1),
            padding=(1, 0),
            output_padding=(1, 0),
            dilation=(1, 1),
            groups=1,
        )
        self.assertTrue(np.allclose(x_node.grad, expected_x_grad, atol=1e-10))
        self.assertTrue(np.allclose(kernel_node.grad, expected_kernel_grad, atol=1e-10))


if __name__ == "__main__":
    unittest.main()
