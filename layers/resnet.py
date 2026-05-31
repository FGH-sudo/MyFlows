"""ResNet 相关网络层与模型构造器。

包含:
  - BatchNorm2d:         2D 批归一化层
  - GlobalAvgPool2d:     全局平均池化层
  - BasicBlock:          ResNet 基础残差块
  - ResNet18:            18 层 ResNet 模型（支持 ImageNet / CIFAR 两种 stem）

设计说明:
  - 保持与 MyFlows 其它层一致的动态图风格: 每次 forward 构建一次子图。
  - BN 的 training/eval 状态由模型级 ``train()``/``eval()`` 统一切换，
    会同时更新最近一次 forward 所创建的 BN 算子节点。
  - Residual Add 使用现有 ``Add`` 算子(支持 4D+4D 等形状)，梯度经拓扑排序自然分叉/汇合。
"""

from ..core.device import xp

from ..core.node import Variable
from ..ops.basic import Add
from ..ops.batchnorm import BatchNorm2d_Op, GlobalAvgPool2d_Op
from ..ops.activation import ReLU
from ..ops.convolution import Conv2D_Op, Conv2D_ReLU_Op, Flatten_Op
from ..ops.basic import Linear
from .layer import Layer, Conv2D


class BatchNorm2d(Layer):
    """2D BatchNorm 层。

    参数:
      num_features: 通道数 C
      momentum:     running stats 动量 (默认 0.1)
      eps:          数值稳定项
    """

    def __init__(self, num_features, momentum=0.1, eps=1e-5, name=None):
        super().__init__(name=name)
        self.num_features = int(num_features)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.training = True

        gamma_init = xp.ones((self.num_features,), dtype=xp.float64)
        beta_init = xp.zeros((self.num_features,), dtype=xp.float64)
        self.gamma = Variable(gamma_init, trainable=True, name=f"{self.name}_gamma" if self.name else "gamma")
        self.beta = Variable(beta_init, trainable=True, name=f"{self.name}_beta" if self.name else "beta")
        self.params = [self.gamma, self.beta]

        self.running_mean = xp.zeros((self.num_features,), dtype=xp.float64)
        self.running_var = xp.ones((self.num_features,), dtype=xp.float64)
        self._last_op = None

    def forward(self, input_node):
        op = BatchNorm2d_Op(
            input_node,
            self.gamma,
            self.beta,
            self.running_mean,
            self.running_var,
            momentum=self.momentum,
            eps=self.eps,
            training=self.training,
        )
        self._last_op = op
        return op

    def train(self, mode=True):
        self.training = bool(mode)
        if self._last_op is not None:
            self._last_op.training = self.training

    def eval(self):
        self.train(False)


class GlobalAvgPool2d(Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.params = []

    def forward(self, input_node):
        return GlobalAvgPool2d_Op(input_node)


def _conv3x3(in_c, out_c, stride=1):
    return Conv2D(in_c, out_c, kernel_size=3, stride=stride, padding=1,
                  activation=None, fuse_activation=False)


def _conv1x1(in_c, out_c, stride=1):
    return Conv2D(in_c, out_c, kernel_size=1, stride=stride, padding=0,
                  activation=None, fuse_activation=False)


class BasicBlock(Layer):
    """ResNet-18/34 的基础块:
        out = ReLU( BN(Conv3x3( ReLU(BN(Conv3x3(x))) )) + shortcut(x) )
    当 stride != 1 或通道数变化时, shortcut 使用 1x1 Conv + BN 做投影。
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, name=None):
        super().__init__(name=name)
        self.conv1 = _conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = BatchNorm2d(out_channels)
        self.conv2 = _conv3x3(out_channels, out_channels, stride=1)
        self.bn2 = BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample_conv = _conv1x1(in_channels, out_channels, stride=stride)
            self.downsample_bn = BatchNorm2d(out_channels)
            self.downsample = (self.downsample_conv, self.downsample_bn)

        self.params = (
            self.conv1.params + self.bn1.params
            + self.conv2.params + self.bn2.params
        )
        if self.downsample is not None:
            self.params += self.downsample_conv.params + self.downsample_bn.params

        self.sub_layers = [self.conv1, self.bn1, self.conv2, self.bn2]
        if self.downsample is not None:
            self.sub_layers += [self.downsample_conv, self.downsample_bn]

    def forward(self, input_node):
        out = self.conv1(input_node)
        out = self.bn1(out)
        out = ReLU(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample_conv(input_node)
            identity = self.downsample_bn(identity)
        else:
            identity = input_node

        # 残差相加后再 ReLU
        merged = Add(out, identity)
        return ReLU(merged)


class ResNet18(Layer):
    """ResNet-18 模型。

    参数:
      in_channels: 输入图像通道数(RGB 为 3)
      num_classes: 分类类别数
      stem:        'imagenet' 使用 7x7/s=2 卷积 + 3x3/s=2 maxpool;
                   'cifar'    使用 3x3/s=1 卷积, 不做 maxpool
      base_width:  各 stage 的基础通道数, 默认 64
    """

    def __init__(self, in_channels=3, num_classes=1000, stem="imagenet",
                 base_width=64, name=None):
        super().__init__(name=name)
        if stem not in ("imagenet", "cifar"):
            raise ValueError("stem must be 'imagenet' or 'cifar'")
        self.stem_type = stem

        widths = [base_width, base_width * 2, base_width * 4, base_width * 8]

        if stem == "imagenet":
            self.stem_conv = Conv2D(in_channels, widths[0], kernel_size=7, stride=2,
                                    padding=3, activation=None, fuse_activation=False)
        else:
            self.stem_conv = Conv2D(in_channels, widths[0], kernel_size=3, stride=1,
                                    padding=1, activation=None, fuse_activation=False)
        self.stem_bn = BatchNorm2d(widths[0])

        # 4 个 stage, 每个 2 个 BasicBlock
        self.layer1 = self._make_stage(widths[0], widths[0], num_blocks=2, stride=1)
        self.layer2 = self._make_stage(widths[0], widths[1], num_blocks=2, stride=2)
        self.layer3 = self._make_stage(widths[1], widths[2], num_blocks=2, stride=2)
        self.layer4 = self._make_stage(widths[2], widths[3], num_blocks=2, stride=2)

        self.avgpool = GlobalAvgPool2d()

        # fc 权重形状 (C, num_classes)
        fan_in = widths[3]
        w_init = xp.random.randn(fan_in, num_classes) * xp.sqrt(2.0 / fan_in)
        b_init = xp.zeros((num_classes,))
        self.fc_W = Variable(w_init, trainable=True, name="fc_W")
        self.fc_b = Variable(b_init, trainable=True, name="fc_b")

        # 收集所有可训练参数与 BN 层(用于 train/eval 切换)
        self.params = list(self.stem_conv.params) + list(self.stem_bn.params)
        self.sub_layers = [self.stem_conv, self.stem_bn]
        for stage in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in stage:
                self.params += block.params
                self.sub_layers.append(block)
                self.sub_layers.extend(block.sub_layers)
        self.params += [self.fc_W, self.fc_b]

        self.widths = widths
        self.num_classes = num_classes

    def _make_stage(self, in_c, out_c, num_blocks, stride):
        blocks = [BasicBlock(in_c, out_c, stride=stride)]
        for _ in range(num_blocks - 1):
            blocks.append(BasicBlock(out_c, out_c, stride=1))
        return blocks

    def forward(self, input_node):
        # stem
        x = self.stem_conv(input_node)
        x = self.stem_bn(x)
        x = ReLU(x)
        if self.stem_type == "imagenet":
            from ..ops.convolution import MaxPool2d_Op
            x = MaxPool2d_Op(x, kernel_size=3, stride=2)

        for block in self.layer1:
            x = block(x)
        for block in self.layer2:
            x = block(x)
        for block in self.layer3:
            x = block(x)
        for block in self.layer4:
            x = block(x)

        x = self.avgpool(x)
        x = Linear(x, self.fc_W, self.fc_b)
        return x

    def train(self, mode=True):
        for layer in self.sub_layers:
            if hasattr(layer, "train") and isinstance(layer, BatchNorm2d):
                layer.train(mode)

    def eval(self):
        self.train(False)
