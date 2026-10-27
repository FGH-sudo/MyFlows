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
from ..core.dtype import resolve_dtype

from ..core.node import Variable
from ..ops.basic import Add
from ..ops.batchnorm import BatchNorm2d_Op, GlobalAvgPool2d_Op
from ..ops.activation import ReLU
from ..ops.convolution import Conv2D_Op, Conv2D_ReLU_Op, Flatten_Op
from ..ops.basic import Linear
from .layer import Layer, Conv2D, Dropout
from ..utils.initializers import make_initializer


class BatchNorm2d(Layer):
    """2D BatchNorm 层。

    参数:
      num_features: 通道数 C
      momentum:     running stats 动量 (默认 0.1)
      eps:          数值稳定项
      unbiased_running_var: running_var 跟踪无偏方差（torch.nn.BatchNorm2d 语义）；默认保持旧行为

    冻结（freeze）后 gamma/beta 不再训练，且始终用 running stats 归一化、不更新它们：
    外层 model.train() 不会覆盖冻结层的 eval 状态。
    """

    def __init__(self, num_features, momentum=0.1, eps=1e-5, name=None, dtype=None, unbiased_running_var=False):
        super().__init__(name=name)
        self.num_features = int(num_features)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.unbiased_running_var = bool(unbiased_running_var)
        self.training = True
        self.frozen = False
        dtype = resolve_dtype(dtype)

        gamma_init = xp.ones((self.num_features,), dtype=dtype)
        beta_init = xp.zeros((self.num_features,), dtype=dtype)
        self.gamma = Variable(gamma_init, trainable=True, name=f"{self.name}_gamma" if self.name else "gamma")
        self.beta = Variable(beta_init, trainable=True, name=f"{self.name}_beta" if self.name else "beta")
        self.params = [self.gamma, self.beta]

        self.running_mean = xp.zeros((self.num_features,), dtype=dtype)
        self.running_var = xp.ones((self.num_features,), dtype=dtype)
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
            unbiased_running_var=self.unbiased_running_var,
        )
        self._last_op = op
        return op

    def train(self, mode=True):
        self.training = bool(mode) and not self.frozen
        if self._last_op is not None:
            self._last_op.training = self.training

    def eval(self):
        self.train(False)

    def freeze(self, frozen=True):
        self.frozen = bool(frozen)
        for param in self.params:
            param.trainable = not self.frozen
        if self.frozen:
            self.train(False)


class GlobalAvgPool2d(Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.params = []

    def forward(self, input_node):
        return GlobalAvgPool2d_Op(input_node)


COMPAT_MODES = ("myflows", "torchvision")


def _conv3x3(in_c, out_c, stride=1, initializer=None, backend="auto", bias=True):
    return Conv2D(in_c, out_c, kernel_size=3, stride=stride, padding=1,
                  activation=None, fuse_activation=False, initializer=initializer, backend=backend, bias=bias)


def _conv1x1(in_c, out_c, stride=1, initializer=None, backend="auto", bias=True):
    return Conv2D(in_c, out_c, kernel_size=1, stride=stride, padding=0,
                  activation=None, fuse_activation=False, initializer=initializer, backend=backend, bias=bias)


def conv_tensors(prefix, conv):
    """torchvision 名称 -> MyFlows 张量：卷积核与（若有）偏置。"""
    items = [(f"{prefix}.weight", conv.kernel, "param")]
    if conv.b is not None:
        items.append((f"{prefix}.bias", conv.b, "param"))
    return items


def bn_tensors(prefix, bn):
    """BN 的 gamma/beta 与两个 running buffer；num_batches_tracked 不映射（momentum 固定 0.1 时 torch 也不用它）。"""
    return [(f"{prefix}.weight", bn.gamma, "param"), (f"{prefix}.bias", bn.beta, "param"),
            (f"{prefix}.running_mean", (bn, "running_mean"), "buffer"),
            (f"{prefix}.running_var", (bn, "running_var"), "buffer")]


class BasicBlock(Layer):
    """ResNet-18/34 的基础块:
        out = ReLU( BN(Conv3x3( ReLU(BN(Conv3x3(x))) )) + shortcut(x) )
    当 stride != 1 或通道数变化时, shortcut 使用 1x1 Conv + BN 做投影。
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, name=None, initializer=None, conv_backend="auto",
                 compat="myflows"):
        super().__init__(name=name)
        # torchvision 兼容：卷积无偏置（后接 BN），running_var 按无偏方差更新
        bias, unbiased = compat != "torchvision", compat == "torchvision"
        self.conv1 = _conv3x3(in_channels, out_channels, stride=stride, initializer=initializer, backend=conv_backend,
                              bias=bias)
        self.bn1 = BatchNorm2d(out_channels, unbiased_running_var=unbiased)
        self.conv2 = _conv3x3(out_channels, out_channels, stride=1, initializer=initializer, backend=conv_backend,
                              bias=bias)
        self.bn2 = BatchNorm2d(out_channels, unbiased_running_var=unbiased)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample_conv = _conv1x1(in_channels, out_channels, stride=stride, initializer=initializer,
                                            backend=conv_backend, bias=bias)
            self.downsample_bn = BatchNorm2d(out_channels, unbiased_running_var=unbiased)
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

    def torchvision_tensors(self, prefix):
        items = conv_tensors(f"{prefix}.conv1", self.conv1) + bn_tensors(f"{prefix}.bn1", self.bn1)
        items += conv_tensors(f"{prefix}.conv2", self.conv2) + bn_tensors(f"{prefix}.bn2", self.bn2)
        if self.downsample is not None:
            items += conv_tensors(f"{prefix}.downsample.0", self.downsample_conv)
            items += bn_tensors(f"{prefix}.downsample.1", self.downsample_bn)
        return items


class ResNet18(Layer):
    """ResNet-18 模型。

    参数:
      in_channels: 输入图像通道数(RGB 为 3)
      output_dim: 回归输出维度，DonkeyCar 默认为 [angle, throttle] 两维
      stem:        'imagenet' 使用 7x7/s=2 卷积 + 3x3/s=2 maxpool;
                   'cifar'    使用 3x3/s=1 卷积, 不做 maxpool
      base_width:  各 stage 的基础通道数, 默认 64
      dropout_seed: dropout 掩码的随机种子；None 用全局随机源
      compat:      'myflows'（默认，旧 checkpoint 的结构）或 'torchvision'：与 torchvision.models.resnet18
                   逐层一致（imagenet stem、maxpool padding=1、卷积无偏置、BN 无偏 running_var），可载入官方权重
      input_norm:  None 或 'imagenet'：在图内先按 ImageNet 均值/标准差归一化 [0, 1] 的 RGB 输入
    """

    def __init__(self, in_channels=3, output_dim=2, stem="imagenet",
                 base_width=64, name=None, dropout=0.0, initializer=None, conv_backend="auto",
                 dropout_seed=None, compat="myflows", input_norm=None):
        super().__init__(name=name)
        if stem not in ("imagenet", "cifar"):
            raise ValueError("stem must be 'imagenet' or 'cifar'")
        if compat not in COMPAT_MODES:
            raise ValueError(f"compat must be one of {COMPAT_MODES}")
        if compat == "torchvision" and (stem != "imagenet" or int(base_width) != 64 or in_channels != 3):
            raise ValueError("compat='torchvision' requires stem='imagenet', base_width=64 and 3 input channels")
        if input_norm not in (None, "imagenet"):
            raise ValueError("input_norm must be None or 'imagenet'")
        from ..ops.convolution import _backend_name
        self.conv_backend = _backend_name(conv_backend)
        self.stem_type = stem
        self.compat = compat
        self.input_norm = input_norm
        self.output_dim = int(output_dim)
        bias, unbiased = compat != "torchvision", compat == "torchvision"

        widths = [base_width, base_width * 2, base_width * 4, base_width * 8]

        if stem == "imagenet":
            self.stem_conv = Conv2D(in_channels, widths[0], kernel_size=7, stride=2,
                                    padding=3, activation=None, fuse_activation=False, initializer=initializer,
                                    backend=self.conv_backend, bias=bias)
        else:
            self.stem_conv = Conv2D(in_channels, widths[0], kernel_size=3, stride=1,
                                    padding=1, activation=None, fuse_activation=False, initializer=initializer,
                                    backend=self.conv_backend)
        self.stem_bn = BatchNorm2d(widths[0], unbiased_running_var=unbiased)

        # 4 个 stage, 每个 2 个 BasicBlock
        self.layer1 = self._make_stage(widths[0], widths[0], num_blocks=2, stride=1, initializer=initializer)
        self.layer2 = self._make_stage(widths[0], widths[1], num_blocks=2, stride=2, initializer=initializer)
        self.layer3 = self._make_stage(widths[1], widths[2], num_blocks=2, stride=2, initializer=initializer)
        self.layer4 = self._make_stage(widths[2], widths[3], num_blocks=2, stride=2, initializer=initializer)

        self.avgpool = GlobalAvgPool2d()
        self.dropout = Dropout(float(dropout), seed=dropout_seed, name="resnet_dropout") if float(dropout or 0.0) > 0.0 else None

        # fc 权重形状 (C, output_dim)
        fan_in = widths[3]
        init = make_initializer(initializer)
        w_init = xp.asarray(init((fan_in, self.output_dim)), dtype=resolve_dtype())
        b_init = xp.zeros((self.output_dim,), dtype=w_init.dtype)
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
        if self.dropout is not None:
            self.sub_layers.append(self.dropout)

        self.widths = widths

    def _make_stage(self, in_c, out_c, num_blocks, stride, initializer=None):
        blocks = [BasicBlock(in_c, out_c, stride=stride, initializer=initializer, conv_backend=self.conv_backend,
                             compat=self.compat)]
        for _ in range(num_blocks - 1):
            blocks.append(BasicBlock(out_c, out_c, stride=1, initializer=initializer, conv_backend=self.conv_backend,
                                     compat=self.compat))
        return blocks

    def forward(self, input_node):
        self._last_feature_nodes = {}
        if self.input_norm == "imagenet":
            from ..ops.pooling import ChannelNormalize_Op
            input_node = ChannelNormalize_Op(input_node)
        # stem
        x = self.stem_conv(input_node)
        x = self.stem_bn(x)
        x = ReLU(x)
        self._last_feature_nodes["stem"] = x
        if self.stem_type == "imagenet":
            from ..ops.convolution import MaxPool2d_Op
            # torchvision 的 stem 池化带 padding=1（224 输入得到 56x56）；旧结构保持无 padding
            x = MaxPool2d_Op(x, kernel_size=3, stride=2, backend=self.conv_backend,
                             padding=1 if self.compat == "torchvision" else 0)

        for block in self.layer1:
            x = block(x)
        self._last_feature_nodes["layer1"] = x
        for block in self.layer2:
            x = block(x)
        self._last_feature_nodes["layer2"] = x
        for block in self.layer3:
            x = block(x)
        self._last_feature_nodes["layer3"] = x
        for block in self.layer4:
            x = block(x)
        self._last_feature_nodes["layer4"] = x

        x = self.avgpool(x)
        if self.dropout is not None:
            x = self.dropout(x)
        x = Linear(x, self.fc_W, self.fc_b)
        return x

    def train(self, mode=True):
        for layer in self.sub_layers:
            if hasattr(layer, "train"):
                layer.train(mode)

    def eval(self):
        self.train(False)

    # --- 预训练权重与分阶段冻结（MyFlows.utils.pretrained） ---
    arch = "resnet18"
    head_keys = ("fc.weight", "fc.bias")

    def torchvision_tensors(self):
        """[(torchvision 名称, 目标, 类型)]；类型 'param_t' 表示载入时转置 [out,in] -> [in,out]。"""
        if self.compat != "torchvision":
            raise ValueError("torchvision weights need ResNet18(compat='torchvision')")
        items = conv_tensors("conv1", self.stem_conv) + bn_tensors("bn1", self.stem_bn)
        for stage_index, stage in enumerate([self.layer1, self.layer2, self.layer3, self.layer4], start=1):
            for block_index, block in enumerate(stage):
                items += block.torchvision_tensors(f"layer{stage_index}.{block_index}")
        items += [("fc.weight", self.fc_W, "param_t"), ("fc.bias", self.fc_b, "param")]
        return items

    def stages(self):
        """可冻结的阶段，按输入到输出排列；任务头（fc）不在其中。"""
        return [("stem", [self.stem_conv, self.stem_bn]), ("layer1", self.layer1), ("layer2", self.layer2),
                ("layer3", self.layer3), ("layer4", self.layer4)]
