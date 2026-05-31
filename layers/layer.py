from ..core.device import xp
from ..core.node import Variable
from ..ops.basic import Linear
from ..ops.convolution import Conv2D_Op, ConvTranspose2D_Op, MaxPool2d_Op, Flatten_Op


def _pair(value, name, allow_zero=False):
    if isinstance(value, int):
        minimum = 0 if allow_zero else 1
        if value < minimum:
            relation = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be {relation}, got {value}")
        return (value, value)
    if isinstance(value, (tuple, list)) and len(value) == 2:
        first, second = int(value[0]), int(value[1])
        minimum = 0 if allow_zero else 1
        if first < minimum or second < minimum:
            relation = "non-negative" if allow_zero else "positive"
            raise ValueError(f"{name} must be {relation}, got {value}")
        return (first, second)
    raise TypeError(f"{name} must be an int or a pair of ints, got {value!r}")


def _positive_int(value, name):
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


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
        w_init = xp.random.randn(input_dim, output_dim) * xp.sqrt(2.0 / input_dim)
        b_init = xp.zeros((1, output_dim))
        
        self.W = Variable(w_init, trainable=True, name=f"{self.name}_W" if self.name else "W")
        self.b = Variable(b_init, trainable=True, name=f"{self.name}_b" if self.name else "b")
        
        self.params = [self.W, self.b]
        self.activation = activation # 传入类名

    def forward(self, input_node):
        """构建计算图分支"""
        # 融合：Z = XW + b（单算子，减少节点）
        z = Linear(input_node, self.W, self.b)
        
        if self.activation:
            return self.activation(z)
        return z


class Conv2D(Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        groups=1,
        dilation=1,
        activation=None,
        fuse_activation=True,
        name=None,
    ):
        super().__init__(name=name)

        in_channels = _positive_int(in_channels, "in_channels")
        out_channels = _positive_int(out_channels, "out_channels")
        groups = _positive_int(groups, "groups")
        kernel_h, kernel_w = _pair(kernel_size, "kernel_size")
        stride = _pair(stride, "stride")
        padding = _pair(padding, "padding", allow_zero=True)
        dilation = _pair(dilation, "dilation")

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        fan_in = (in_channels // groups) * kernel_h * kernel_w
        k_init = xp.random.randn(out_channels, in_channels // groups, kernel_h, kernel_w) * xp.sqrt(2. / fan_in)
        b_init = xp.zeros((out_channels,))

        self.kernel = Variable(k_init, trainable=True, name=f"{self.name}_kernel" if self.name else "kernel")
        self.b = Variable(b_init, trainable=True, name=f"{self.name}_bias" if self.name else "bias")
        self.params = [self.kernel, self.b]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_h, kernel_w)
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation
        self.activation = activation
        self.fuse_activation = bool(fuse_activation)

    def forward(self, input_node):
        # 卷积 + 偏置在同一算子内融合；若 activation=ReLU 且允许融合，则构图阶段直接用融合算子
        if self.fuse_activation and self.activation is not None:
            try:
                from ..ops.activation import ReLU
                from ..ops.convolution import Conv2D_ReLU_Op
                if self.activation is ReLU:
                    conv_out = Conv2D_ReLU_Op(
                        input_node,
                        self.kernel,
                        stride=self.stride,
                        padding=self.padding,
                        groups=self.groups,
                        dilation=self.dilation,
                        bias=self.b,
                    )
                    return conv_out
            except Exception:
                # 保持动态图语义：导入/判定失败时退化为非融合路径
                pass

        conv_out = Conv2D_Op(
            input_node,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
            dilation=self.dilation,
            bias=self.b,
        )
        return self.activation(conv_out) if self.activation else conv_out


class GroupedConv2D(Conv2D):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, groups=2, activation=None, name=None):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=1,
            activation=activation,
            name=name,
        )


class DepthwiseConv2D(Conv2D):
    def __init__(
        self,
        in_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        depth_multiplier=1,
        dilation=1,
        activation=None,
        name=None,
    ):
        depth_multiplier = _positive_int(depth_multiplier, "depth_multiplier")
        super().__init__(
            in_channels,
            in_channels * depth_multiplier,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            dilation=dilation,
            activation=activation,
            name=name,
        )
        self.depth_multiplier = depth_multiplier


class DepthwiseSeparableConv2D(Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        depth_multiplier=1,
        dilation=1,
        activation=None,
        name=None,
    ):
        super().__init__(name=name)
        self.depth_multiplier = _positive_int(depth_multiplier, "depth_multiplier")
        mid_channels = _positive_int(in_channels, "in_channels") * self.depth_multiplier

        depthwise_name = f"{self.name}_depthwise" if self.name else "depthwise"
        pointwise_name = f"{self.name}_pointwise" if self.name else "pointwise"
        self.depthwise = DepthwiseConv2D(
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            depth_multiplier=self.depth_multiplier,
            dilation=dilation,
            activation=None,
            name=depthwise_name,
        )
        self.pointwise = Conv2D(
            mid_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            dilation=1,
            activation=None,
            name=pointwise_name,
        )
        self.params = self.depthwise.params + self.pointwise.params
        self.activation = activation

    def forward(self, input_node):
        z = self.pointwise(self.depthwise(input_node))
        return self.activation(z) if self.activation else z


class DilatedConv2D(Conv2D):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=2,
        activation=None,
        name=None,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=1,
            dilation=dilation,
            activation=activation,
            name=name,
        )


class ConvTranspose2D(Layer):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        dilation=1,
        activation=None,
        fuse_activation=True,
        name=None,
    ):
        super().__init__(name=name)

        in_channels = _positive_int(in_channels, "in_channels")
        out_channels = _positive_int(out_channels, "out_channels")
        groups = _positive_int(groups, "groups")
        kernel_h, kernel_w = _pair(kernel_size, "kernel_size")
        stride = _pair(stride, "stride")
        padding = _pair(padding, "padding", allow_zero=True)
        output_padding = _pair(output_padding, "output_padding", allow_zero=True)
        dilation = _pair(dilation, "dilation")

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        if any(output_pad >= max(stride_val, dilation_val) for output_pad, stride_val, dilation_val in zip(output_padding, stride, dilation)):
            raise ValueError("output_padding must be smaller than max(stride, dilation)")

        fan_in = (out_channels // groups) * kernel_h * kernel_w
        k_init = xp.random.randn(in_channels, out_channels // groups, kernel_h, kernel_w) * xp.sqrt(2. / fan_in)
        b_init = xp.zeros((out_channels,))

        self.kernel = Variable(k_init, trainable=True, name=f"{self.name}_kernel" if self.name else "kernel")
        self.b = Variable(b_init, trainable=True, name=f"{self.name}_bias" if self.name else "bias")
        self.params = [self.kernel, self.b]

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_h, kernel_w)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        self.dilation = dilation
        self.activation = activation
        self.fuse_activation = bool(fuse_activation)

    def forward(self, input_node):
        if self.fuse_activation and self.activation is not None:
            try:
                from ..ops.activation import ReLU
                from ..ops.convolution import ConvTranspose2D_ReLU_Op
                if self.activation is ReLU:
                    conv_out = ConvTranspose2D_ReLU_Op(
                        input_node,
                        self.kernel,
                        stride=self.stride,
                        padding=self.padding,
                        output_padding=self.output_padding,
                        groups=self.groups,
                        dilation=self.dilation,
                        bias=self.b,
                    )
                    return conv_out
            except Exception:
                pass

        conv_out = ConvTranspose2D_Op(
            input_node,
            self.kernel,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            groups=self.groups,
            dilation=self.dilation,
            bias=self.b,
        )
        return self.activation(conv_out) if self.activation else conv_out

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
