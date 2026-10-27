"""MobileNetV2 与 AlexNet：与 torchvision.models.mobilenet_v2 / alexnet 逐层一致，可载入官方 ImageNet 权重。

两者与 ResNet18(compat='torchvision') 一样提供 ``torchvision_tensors()``（torchvision 名称 -> MyFlows 张量）、
``head_keys``（替换任务头时跳过的名称）与 ``stages()``（按输入到输出排列的可冻结阶段），供
``MyFlows.utils.pretrained`` 载入权重和冻结阶段使用。
"""

from ..core.device import xp
from ..core.dtype import resolve_dtype
from ..core.node import Variable
from ..ops.activation import ReLU, ReLU6
from ..ops.basic import Add, Linear
from ..ops.batchnorm import GlobalAvgPool2d_Op
from ..ops.convolution import Flatten_Op, MaxPool2d_Op
from ..ops.pooling import AdaptiveAvgPool2d_Op, ChannelNormalize_Op
from ..utils.initializers import make_initializer
from .layer import Conv2D, Dropout, Layer
from .resnet import BatchNorm2d, bn_tensors, conv_tensors


def _check_input_norm(input_norm):
    if input_norm not in (None, "imagenet"):
        raise ValueError("input_norm must be None or 'imagenet'")
    return input_norm


def _make_linear(in_dim, out_dim, initializer, name):
    """全连接参数：W 形状 (in, out)（torchvision 为 (out, in)，载入时转置）。"""
    init = make_initializer(initializer)
    w = Variable(xp.asarray(init((in_dim, out_dim)), dtype=resolve_dtype()), trainable=True, name=f"{name}_W")
    b = Variable(xp.zeros((out_dim,), dtype=w.value.dtype), trainable=True, name=f"{name}_b")
    return w, b


class ConvBNAct(Layer):
    """Conv(无偏置) + BN + 可选 ReLU6：torchvision 的 Conv2dNormActivation。"""

    def __init__(self, in_c, out_c, kernel_size=3, stride=1, groups=1, activation=True, initializer=None,
                 conv_backend="auto", name=None):
        super().__init__(name=name)
        # 原生卷积后端只支持 groups=1；深度卷积走 CuPy/NumPy 的逐抽头路径
        backend = conv_backend if groups == 1 else "auto"
        self.conv = Conv2D(in_c, out_c, kernel_size=kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                           groups=groups, activation=None, fuse_activation=False, initializer=initializer,
                           backend=backend, bias=False)
        self.bn = BatchNorm2d(out_c, unbiased_running_var=True)
        self.activation = bool(activation)
        self.params = self.conv.params + self.bn.params
        self.sub_layers = [self.conv, self.bn]

    def forward(self, x):
        x = self.bn(self.conv(x))
        return ReLU6(x) if self.activation else x

    def torchvision_tensors(self, prefix):
        return conv_tensors(f"{prefix}.0", self.conv) + bn_tensors(f"{prefix}.1", self.bn)


class InvertedResidual(Layer):
    """倒残差块：1x1 扩张(ReLU6) -> 3x3 深度卷积(ReLU6) -> 1x1 线性投影；stride=1 且通道不变时加恒等残差。"""

    def __init__(self, in_c, out_c, stride, expand_ratio, initializer=None, conv_backend="auto", name=None):
        super().__init__(name=name)
        hidden = int(round(in_c * expand_ratio))
        self.use_residual = stride == 1 and in_c == out_c
        self.expand = None
        if expand_ratio != 1:
            self.expand = ConvBNAct(in_c, hidden, kernel_size=1, initializer=initializer, conv_backend=conv_backend)
        self.depthwise = ConvBNAct(hidden, hidden, kernel_size=3, stride=stride, groups=hidden,
                                   initializer=initializer, conv_backend=conv_backend)
        self.project = Conv2D(hidden, out_c, kernel_size=1, activation=None, fuse_activation=False,
                              initializer=initializer, backend=conv_backend, bias=False)
        self.project_bn = BatchNorm2d(out_c, unbiased_running_var=True)
        parts = ([self.expand] if self.expand is not None else []) + [self.depthwise, self.project, self.project_bn]
        self.sub_layers = []
        self.params = []
        for part in parts:
            self.sub_layers.append(part)
            self.sub_layers.extend(getattr(part, "sub_layers", []))
            self.params += part.params

    def forward(self, x):
        out = self.expand(x) if self.expand is not None else x
        out = self.project_bn(self.project(self.depthwise(out)))
        return Add(out, x) if self.use_residual else out

    def torchvision_tensors(self, prefix):
        # torchvision: block.conv = [expand?, depthwise, project conv, project BN]
        index = 0
        items = []
        if self.expand is not None:
            items += self.expand.torchvision_tensors(f"{prefix}.conv.{index}")
            index += 1
        items += self.depthwise.torchvision_tensors(f"{prefix}.conv.{index}")
        items += conv_tensors(f"{prefix}.conv.{index + 1}", self.project)
        items += bn_tensors(f"{prefix}.conv.{index + 2}", self.project_bn)
        return items


# torchvision 的 (t 扩张倍数, c 输出通道, n 重复次数, s 首块步长)
MOBILENET_V2_SETTINGS = ((1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1),
                         (6, 160, 3, 2), (6, 320, 1, 1))


class MobileNetV2(Layer):
    """MobileNetV2（width_mult=1.0），结构与 torchvision.models.mobilenet_v2 相同。

    output_dim 为分类头输出数（ImageNet 为 1000）；dropout 为分类头前的 dropout（torchvision 为 0.2）。
    """

    arch = "mobilenet_v2"
    head_keys = ("classifier.1.weight", "classifier.1.bias")

    def __init__(self, output_dim=1000, dropout=0.2, initializer=None, conv_backend="auto", dropout_seed=None,
                 input_norm=None, name=None):
        super().__init__(name=name)
        from ..ops.convolution import _backend_name
        self.conv_backend = _backend_name(conv_backend)
        self.input_norm = _check_input_norm(input_norm)
        self.output_dim = int(output_dim)
        self.stem = ConvBNAct(3, 32, kernel_size=3, stride=2, initializer=initializer, conv_backend=self.conv_backend)
        self.blocks = []
        in_c = 32
        for t, c, n, s in MOBILENET_V2_SETTINGS:
            for i in range(n):
                self.blocks.append(InvertedResidual(in_c, c, s if i == 0 else 1, t, initializer=initializer,
                                                    conv_backend=self.conv_backend))
                in_c = c
        self.last_conv = ConvBNAct(in_c, 1280, kernel_size=1, initializer=initializer, conv_backend=self.conv_backend)
        self.dropout = Dropout(float(dropout), seed=dropout_seed, name="mobilenet_dropout") if dropout else None
        self.fc_W, self.fc_b = _make_linear(1280, self.output_dim, initializer, "classifier")

        body = [self.stem] + self.blocks + [self.last_conv]
        self.sub_layers = []
        self.params = []
        for layer in body:
            self.sub_layers.append(layer)
            self.sub_layers.extend(layer.sub_layers)
            self.params += layer.params
        self.params += [self.fc_W, self.fc_b]
        if self.dropout is not None:
            self.sub_layers.append(self.dropout)

    def forward(self, x):
        if self.input_norm == "imagenet":
            x = ChannelNormalize_Op(x)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = GlobalAvgPool2d_Op(self.last_conv(x))
        if self.dropout is not None:
            x = self.dropout(x)
        return Linear(x, self.fc_W, self.fc_b)

    def train(self, mode=True):
        for layer in self.sub_layers:
            if hasattr(layer, "train"):
                layer.train(mode)

    def eval(self):
        self.train(False)

    def torchvision_tensors(self):
        items = self.stem.torchvision_tensors("features.0")
        for index, block in enumerate(self.blocks, start=1):
            items += block.torchvision_tensors(f"features.{index}")
        items += self.last_conv.torchvision_tensors(f"features.{len(self.blocks) + 1}")
        items += [("classifier.1.weight", self.fc_W, "param_t"), ("classifier.1.bias", self.fc_b, "param")]
        return items

    def stages(self):
        """按特征图分辨率分 5 段（112/56/28/14/7，224 输入时），与 ResNet18 的 stem + layer1..4 对应。"""
        b = self.blocks
        return [("stem", [self.stem, b[0]]), ("stage2", b[1:3]), ("stage3", b[3:6]), ("stage4", b[6:13]),
                ("stage5", b[13:] + [self.last_conv])]


class AlexNet(Layer):
    """AlexNet，结构与 torchvision.models.alexnet 相同：5 个带偏置的卷积 + 3 个 3x3/s2 池化 +
    AdaptiveAvgPool2d(6, 6) + 两层 4096 全连接（各带 dropout）+ 分类头。"""

    arch = "alexnet"
    head_keys = ("classifier.6.weight", "classifier.6.bias")
    # (out_c, kernel, stride, padding, 之后是否池化)
    CONVS = ((64, 11, 4, 2, True), (192, 5, 1, 2, True), (384, 3, 1, 1, False), (256, 3, 1, 1, False),
             (256, 3, 1, 1, True))
    FEATURE_INDEX = (0, 3, 6, 8, 10)

    def __init__(self, output_dim=1000, dropout=0.5, initializer=None, conv_backend="auto", dropout_seed=None,
                 input_norm=None, name=None):
        super().__init__(name=name)
        from ..ops.convolution import _backend_name
        self.conv_backend = _backend_name(conv_backend)
        self.input_norm = _check_input_norm(input_norm)
        self.output_dim = int(output_dim)
        self.convs = []
        in_c = 3
        for out_c, k, s, p, _ in self.CONVS:
            self.convs.append(Conv2D(in_c, out_c, kernel_size=k, stride=s, padding=p, activation=ReLU,
                                     initializer=initializer, backend=self.conv_backend))
            in_c = out_c
        seeds = (None, None) if dropout_seed is None else (dropout_seed, dropout_seed + 1)
        self.dropouts = [Dropout(float(dropout), seed=seed, name=f"alexnet_dropout{i}") for i, seed in enumerate(seeds)] \
            if dropout else []
        self.fc6_W, self.fc6_b = _make_linear(256 * 6 * 6, 4096, initializer, "fc6")
        self.fc7_W, self.fc7_b = _make_linear(4096, 4096, initializer, "fc7")
        self.fc_W, self.fc_b = _make_linear(4096, self.output_dim, initializer, "classifier")
        self.sub_layers = list(self.convs) + list(self.dropouts)
        self.params = [p for conv in self.convs for p in conv.params]
        self.params += [self.fc6_W, self.fc6_b, self.fc7_W, self.fc7_b, self.fc_W, self.fc_b]

    def forward(self, x):
        if self.input_norm == "imagenet":
            x = ChannelNormalize_Op(x)
        for conv, (_, _, _, _, pool) in zip(self.convs, self.CONVS):
            x = conv(x)
            if pool:
                x = MaxPool2d_Op(x, kernel_size=3, stride=2, backend=self.conv_backend)
        x = Flatten_Op(AdaptiveAvgPool2d_Op(x, (6, 6)))
        if self.dropouts:
            x = self.dropouts[0](x)
        x = ReLU(Linear(x, self.fc6_W, self.fc6_b))
        if self.dropouts:
            x = self.dropouts[1](x)
        x = ReLU(Linear(x, self.fc7_W, self.fc7_b))
        return Linear(x, self.fc_W, self.fc_b)

    def train(self, mode=True):
        for layer in self.dropouts:
            layer.train(mode)

    def eval(self):
        self.train(False)

    def torchvision_tensors(self):
        items = []
        for index, conv in zip(self.FEATURE_INDEX, self.convs):
            items += conv_tensors(f"features.{index}", conv)
        items += [("classifier.1.weight", self.fc6_W, "param_t"), ("classifier.1.bias", self.fc6_b, "param"),
                  ("classifier.4.weight", self.fc7_W, "param_t"), ("classifier.4.bias", self.fc7_b, "param"),
                  ("classifier.6.weight", self.fc_W, "param_t"), ("classifier.6.bias", self.fc_b, "param")]
        return items

    def stages(self):
        """5 个卷积各为一段；两层 4096 全连接与任务头始终可训练。"""
        return [(f"conv{i + 1}", [conv]) for i, conv in enumerate(self.convs)]
