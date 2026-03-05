import pickle
import os
import numpy as np
from ..core.node import Variable
from ..ops.basic import MatMul, Add
from ..ops.convolution import Conv2D_Op, MaxPool2d_Op, Flatten_Op

class Layer:
    """层基类"""
    def __init__(self, name=None):
        self.name = name
        self.params = [] # 记录层内包含的所有 Variable

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Dense(Layer):
    """全连接层：实现 Z = XW + b"""
    def __init__(self, input_dim, output_dim, activation=None, name=None):
        super().__init__(name=name)
        
        # 参数初始化 (He 初始化，适合 ReLU)
        w_init = np.random.randn(input_dim, output_dim) * np.sqrt(2.0 / input_dim)
        b_init = np.zeros((1, output_dim))
        
        self.W = Variable(w_init, trainable=True, name=f"{self.name}_W" if self.name else "W")
        self.b = Variable(b_init, trainable=True, name=f"{self.name}_b" if self.name else "b")
        
        self.params = [self.W, self.b]
        self.activation = activation # 传入类名

    def forward(self, input_node):
        """构建计算图分支"""
        # Z = XW + b
        z = Add(MatMul(input_node, self.W), self.b)
        
        if self.activation:
            return self.activation(z)
        return z


class Conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, activation=None, name=None):
        super().__init__(name=name)

        fan_in = in_channels * kernel_size**2
        k_init = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2./fan_in)
        b_init = np.zeros((out_channels,))
        
        self.kernel = Variable(k_init, trainable=True, name=f"{self.name}_kernel" if self.name else "kernel")
        self.b = Variable(b_init, trainable=True, name=f"{self.name}_bias" if self.name else "bias")
        self.params = [self.kernel, self.b]
        
        self.stride, self.padding = stride, padding
        self.activation = activation

    def forward(self, input_node):
        # 卷积：传入图像和卷积核
        conv_out = Conv2D_Op(input_node, self.kernel, self.stride, self.padding)
        # 加偏置：Add 算子处理 4D + 1D 的广播
        z = Add(conv_out, self.b)
        return self.activation(z) if self.activation else z

class MaxPool2d(Layer):
    def __init__(self, kernel_size=2, stride=2, name=None):
        super().__init__(name=name)
        self.k, self.s = kernel_size, stride
        self.params = []
    def forward(self, input_node):
        return MaxPool2d_Op(input_node, self.k, self.s)


class Flatten(Layer):
    def forward(self, input_node):
        return Flatten_Op(input_node)


def save_checkpoint(layers, optimizer, epoch, acc, filepath="checkpoint.pkl"):
    """
    保存断点：包含模型、优化器和进度
    """
    # 提取模型权重
    model_state = []
    for layer in layers:
        layer_params = {param.name: param.value for param in layer.params}
        model_state.append({"name": layer.name, "params": layer_params})

    # 提取优化器状态 
    # 保存各个 Variable 对应的梯度平滑值
    opt_state = {}
    if hasattr(optimizer, 'v'): # 针对 Adam 或动量优化器
        opt_state['v'] = {k: v for k, v in optimizer.v.items()}
    if hasattr(optimizer, 'm'):
        opt_state['m'] = {k: v for k, v in optimizer.m.items()}

    checkpoint = {
        "epoch": epoch,
        "model_state": model_state,
        "optimizer_state": opt_state,
        "best_acc": acc
    }

    with open(filepath, 'wb') as f:
        pickle.dump(checkpoint, f)


def load_checkpoint(layers, optimizer, filepath="checkpoint.pkl"):
    """
    加载断点：恢复模型权重、优化器状态并返回起始 Epoch
    """
    if not os.path.exists(filepath):
        print("未发现 Checkpoint，从零开始训练。")
        return -1, 0.0 # 返回起始 epoch 和初始 acc

    with open(filepath, 'rb') as f:
        checkpoint = pickle.load(f)

    # 恢复模型权重
    for i, layer in enumerate(layers):
        saved_params = checkpoint["model_state"][i]["params"]
        for param in layer.params:
            if param.name in saved_params:
                param.value = saved_params[param.name]

    # 恢复优化器状态
    if "optimizer_state" in checkpoint:
        opt_state = checkpoint["optimizer_state"]
        if 'v' in opt_state: optimizer.v = opt_state['v']
        if 'm' in opt_state: optimizer.m = opt_state['m']

    print(f"--- 已成功加载断点，准备从 Epoch {checkpoint['epoch']+1} 继续训练 ---")
    return checkpoint["epoch"], checkpoint.get("best_acc", 0.0)
    