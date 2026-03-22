import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from MyFlows.core.node import Variable
from MyFlows.layers.layer import Conv2D
from MyFlows.ops.convolution import Conv2D_Op, MaxPool2d_Op, im2col, col2im


def naive_conv_forward(x, kernel, stride=(1, 1), padding=(0, 0)):
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    batch_size, in_channels, _, _ = x.shape
    out_channels, kernel_channels, kernel_h, kernel_w = kernel.shape
    if in_channels != kernel_channels:
        raise ValueError("kernel input channels must match x input channels")

    x_pad = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    out_h = (x_pad.shape[2] - kernel_h) // stride_h + 1
    out_w = (x_pad.shape[3] - kernel_w) // stride_w + 1
    output = np.zeros((batch_size, out_channels, out_h, out_w), dtype=x.dtype)

    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride_h
            w_start = j * stride_w
            window = x_pad[:, :, h_start:h_start + kernel_h, w_start:w_start + kernel_w]
            for channel in range(out_channels):
                output[:, channel, i, j] = np.sum(window * kernel[channel], axis=(1, 2, 3))
    return output


def naive_conv_backward(x, kernel, grad_y, stride=(1, 1), padding=(0, 0)):
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    grad_x_pad = np.zeros_like(x_pad)
    grad_kernel = np.zeros_like(kernel)
    _, out_channels, out_h, out_w = grad_y.shape
    _, _, kernel_h, kernel_w = kernel.shape

    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride_h
            w_start = j * stride_w
            window = x_pad[:, :, h_start:h_start + kernel_h, w_start:w_start + kernel_w]
            for channel in range(out_channels):
                local_grad = grad_y[:, channel, i, j][:, None, None, None]
                grad_kernel[channel] += np.sum(window * local_grad, axis=0)
                grad_x_pad[:, :, h_start:h_start + kernel_h, w_start:w_start + kernel_w] += kernel[channel] * local_grad

    if pad_h == 0 and pad_w == 0:
        grad_x = grad_x_pad
    else:
        grad_x = grad_x_pad[:, :, pad_h:pad_h + x.shape[2], pad_w:pad_w + x.shape[3]]
    return grad_x, grad_kernel


def naive_maxpool_forward(x, kernel_size=(2, 2), stride=(2, 2)):
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    batch_size, channels, height, width = x.shape
    out_h = (height - kernel_h) // stride_h + 1
    out_w = (width - kernel_w) // stride_w + 1
    output = np.zeros((batch_size, channels, out_h, out_w), dtype=x.dtype)

    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride_h
            w_start = j * stride_w
            window = x[:, :, h_start:h_start + kernel_h, w_start:w_start + kernel_w]
            output[:, :, i, j] = np.max(window, axis=(2, 3))
    return output


def naive_maxpool_backward(x, grad_y, kernel_size=(2, 2), stride=(2, 2)):
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    grad_x = np.zeros_like(x)
    _, _, out_h, out_w = grad_y.shape

    for i in range(out_h):
        for j in range(out_w):
            h_start = i * stride_h
            w_start = j * stride_w
            window = x[:, :, h_start:h_start + kernel_h, w_start:w_start + kernel_w]
            mask = window == np.max(window, axis=(2, 3), keepdims=True)
            grad_x[:, :, h_start:h_start + kernel_h, w_start:w_start + kernel_w] += mask * grad_y[:, :, i, j][:, :, None, None]
    return grad_x


def naive_im2col(x, kernel_size, stride=(1, 1), padding=(0, 0)):
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    batch_size, channels, _, _ = x.shape
    out_h = (x_pad.shape[2] - kernel_h) // stride_h + 1
    out_w = (x_pad.shape[3] - kernel_w) // stride_w + 1

    rows = []
    for batch in range(batch_size):
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride_h
                w_start = j * stride_w
                window = x_pad[batch, :, h_start:h_start + kernel_h, w_start:w_start + kernel_w]
                rows.append(window.reshape(channels * kernel_h * kernel_w))
    return np.array(rows)


def naive_col2im(rows, input_shape, kernel_size, stride=(1, 1), padding=(0, 0)):
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    batch_size, channels, height, width = input_shape
    padded = np.zeros((batch_size, channels, height + 2 * pad_h, width + 2 * pad_w), dtype=rows.dtype)
    out_h = (padded.shape[2] - kernel_h) // stride_h + 1
    out_w = (padded.shape[3] - kernel_w) // stride_w + 1

    row_index = 0
    for batch in range(batch_size):
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * stride_h
                w_start = j * stride_w
                window = rows[row_index].reshape(channels, kernel_h, kernel_w)
                padded[batch, :, h_start:h_start + kernel_h, w_start:w_start + kernel_w] += window
                row_index += 1

    if pad_h == 0 and pad_w == 0:
        return padded
    return padded[:, :, pad_h:pad_h + height, pad_w:pad_w + width]


class ConvolutionOpsTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(7)

    def test_conv2d_forward_matches_naive_reference(self):
        x = self.rng.normal(size=(2, 3, 5, 6))
        kernel = self.rng.normal(size=(4, 3, 3, 2))

        x_node = Variable(x.copy())
        kernel_node = Variable(kernel.copy(), trainable=True)
        op = Conv2D_Op(x_node, kernel_node, stride=(2, 1), padding=(1, 0))
        op.forward(x_node.value, kernel_node.value)

        expected = naive_conv_forward(x, kernel, stride=(2, 1), padding=(1, 0))
        self.assertTrue(np.allclose(op.value, expected, atol=1e-10))

    def test_im2col_and_col2im_match_naive_reference(self):
        x = self.rng.normal(size=(2, 3, 4, 5))
        expected_cols = naive_im2col(x, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0))

        cols, context, _ = im2col(x, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0))
        self.assertTrue(np.allclose(cols, expected_cols, atol=1e-10))

        rows = self.rng.normal(size=cols.shape)
        recovered, _ = col2im(rows, x.shape, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0), context=context)
        expected_recovered = naive_col2im(rows, x.shape, kernel_size=(2, 3), stride=(1, 2), padding=(1, 0))
        self.assertTrue(np.allclose(recovered, expected_recovered, atol=1e-10))

    def test_conv2d_backward_matches_naive_reference(self):
        x = self.rng.normal(size=(2, 2, 4, 5))
        kernel = self.rng.normal(size=(3, 2, 2, 3))
        x_node = Variable(x.copy())
        kernel_node = Variable(kernel.copy(), trainable=True)
        op = Conv2D_Op(x_node, kernel_node, stride=(1, 2), padding=(1, 1))
        op.forward(x_node.value, kernel_node.value)

        upstream_grad = self.rng.normal(size=op.value.shape)
        x_node.clear_grad()
        kernel_node.clear_grad()
        op.grad = upstream_grad
        op.backward()

        expected_x_grad, expected_kernel_grad = naive_conv_backward(
            x, kernel, upstream_grad, stride=(1, 2), padding=(1, 1)
        )
        self.assertTrue(np.allclose(x_node.grad, expected_x_grad, atol=1e-10))
        self.assertTrue(np.allclose(kernel_node.grad, expected_kernel_grad, atol=1e-10))

    def test_maxpool_forward_and_backward_match_reference(self):
        x = self.rng.normal(size=(2, 2, 5, 6))
        x_node = Variable(x.copy())
        op = MaxPool2d_Op(x_node, kernel_size=(2, 3), stride=(2, 1))
        op.forward(x_node.value)

        expected = naive_maxpool_forward(x, kernel_size=(2, 3), stride=(2, 1))
        self.assertTrue(np.allclose(op.value, expected, atol=1e-10))

        upstream_grad = self.rng.normal(size=op.value.shape)
        x_node.clear_grad()
        op.grad = upstream_grad
        op.backward()

        expected_x_grad = naive_maxpool_backward(x, upstream_grad, kernel_size=(2, 3), stride=(2, 1))
        self.assertTrue(np.allclose(x_node.grad, expected_x_grad, atol=1e-10))

    def test_conv2d_layer_supports_non_square_kernel(self):
        np.random.seed(0)
        layer = Conv2D(in_channels=3, out_channels=5, kernel_size=(3, 2), stride=(2, 1), padding=(1, 0), name="conv")
        self.assertEqual(layer.kernel.value.shape, (5, 3, 3, 2))
        self.assertEqual(layer.kernel_size, (3, 2))


if __name__ == "__main__":
    unittest.main()
