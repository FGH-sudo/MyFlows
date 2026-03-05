import numpy as np
from ..core.node import Node


class Conv2D_Op(Node):
    def __init__(self, x, kernel, stride=1, padding=0):
        # x: 图像, kernel: 卷积核
        super().__init__(x, kernel)
        self.stride = stride
        self.padding = padding

    def forward(self, x_val, kernel_val):
        # x_val: (N, C_in, H, W), kernel_val: (C_out, C_in, kH, kW)
        self.x_val = x_val
        self.kernel_val = kernel_val
        N, C_in, H, W = x_val.shape
        C_out, _, kH, kW = kernel_val.shape
        
        out_H = (H + 2 * self.padding - kH) // self.stride + 1
        out_W = (W + 2 * self.padding - kW) // self.stride + 1
        
        # 填充
        self.x_pad = np.pad(x_val, ((0,0), (0,0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        res = np.zeros((N, C_out, out_H, out_W))
        
        for i in range(out_H):
            for j in range(out_W):
                h_s, w_s = i * self.stride, j * self.stride
                window = self.x_pad[:, :, h_s:h_s+kH, w_s:w_s+kW]
                for c in range(C_out):
                    # 每个卷积核与窗口做内积
                    res[:, c, i, j] = np.sum(window * kernel_val[c], axis=(1, 2, 3))
        self.value = res

    def backward(self):
        x_node, kernel_node = self.parents
        grad_y = self.grad # 上层传来的梯度 (N, C_out, out_H, out_W)
        
        N, C_in, H, W = x_node.value.shape
        C_out, _, kH, kW = kernel_node.value.shape
        _, _, out_H, out_W = grad_y.shape

        if x_node.grad is None: x_node.clear_grad()
        if kernel_node.grad is None: kernel_node.clear_grad()

        grad_x_pad = np.zeros_like(self.x_pad)
        
        for i in range(out_H):
            for j in range(out_W):
                h_s, w_s = i * self.stride, j * self.stride
                window = self.x_pad[:, :, h_s:h_s+kH, w_s:w_s+kW]
                for c in range(C_out):
                    # 对卷积核的梯度：窗口 * 输出梯度
                    kernel_node.grad[c] += np.sum(window * grad_y[:, c, i, j][:, None, None, None], axis=0)
                    # 对图像的梯度：卷积核 * 输出梯度
                    grad_x_pad[:, :, h_s:h_s+kH, w_s:w_s+kW] += kernel_node.value[c] * grad_y[:, c, i, j][:, None, None, None]
        
        # 剔除 padding
        if self.padding > 0:
            x_node.grad += grad_x_pad[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            x_node.grad += grad_x_pad


class MaxPool2d_Op(Node):
    def __init__(self, x, kernel_size=2, stride=2):
        super().__init__(x)
        self.k, self.s = kernel_size, stride

    def forward(self, x_val):
        N, C, H, W = x_val.shape
        out_H, out_W = H // self.s, W // self.s
        self.value = np.zeros((N, C, out_H, out_W))

        for i in range(out_H):
            for j in range(out_W):
                h_s, w_s = i * self.s, j * self.s
                window = x_val[:, :, h_s:h_s+self.k, w_s:w_s+self.k]
                self.value[:, :, i, j] = np.max(window, axis=(2, 3))
        
    def backward(self):
        x_node = self.parents[0]
        if x_node.grad is None: x_node.clear_grad()
        N, C, out_H, out_W = self.value.shape
        for i in range(out_H):
            for j in range(out_W):
                h_s, w_s = i * self.s, j * self.s
                window = x_node.value[:, :, h_s:h_s+self.k, w_s:w_s+self.k]
                # 只有最大值的位置才有梯度
                mask = (window == np.max(window, axis=(2, 3), keepdims=True))
                x_node.grad[:, :, h_s:h_s+self.k, w_s:w_s+self.k] += mask * self.grad[:, :, i, j][:, :, None, None]


class Flatten_Op(Node):
    def forward(self, x):
        self.in_shape = x.shape
        self.value = x.reshape(x.shape[0], -1)
    def backward(self):
        self.parents[0].grad += self.grad.reshape(self.in_shape)


