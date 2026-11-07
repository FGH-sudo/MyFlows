"""Feature transformations between a backbone and a replacement head."""

from ..core.node import GradientOp
from ..layers.layer import Conv2D, Dropout, Flatten, Layer
from ..layers.resnet import BatchNorm2d, GlobalAvgPool2d


class _VectorToMap(GradientOp):
    def forward(self, value):
        self.value = value[:, :, None, None]

    def backward(self):
        self.parents[0].accumulate_grad(self.grad[:, :, 0, 0])


class FeatureAdapter(Layer):
    def __init__(self, channels, *, spatial_size=(1, 1), vector=False, pooling="global", projection=None,
                 batch_norm=False, dropout=0., initializer=None, backend="auto", seed=None):
        super().__init__(name="feature_adapter")
        if pooling not in ("global", "flatten") or min(channels, *spatial_size) < 1:
            raise ValueError("invalid feature dimensions or pooling")
        self.vector = vector
        self.sub_layers = []
        if projection is not None:
            self.sub_layers.append(Conv2D(channels, projection, kernel_size=1, backend=backend, initializer=initializer,
                                         bias=True, name="feature_projection"))
            channels = projection
        if batch_norm:
            self.sub_layers.append(BatchNorm2d(channels, name="feature_bn"))
        self.sub_layers.append(GlobalAvgPool2d() if pooling == "global" else Flatten())
        if dropout:
            self.sub_layers.append(Dropout(dropout, seed=seed, name="feature_dropout"))
        self.params = [p for layer in self.sub_layers for p in layer.params]
        self.output_dim = channels if pooling == "global" else channels * spatial_size[0] * spatial_size[1]

    def forward(self, features):
        value = _VectorToMap(features) if self.vector else features
        for layer in self.sub_layers:
            value = layer(value)
        return value

    def train(self, mode=True):
        for layer in self.sub_layers:
            if hasattr(layer, "train"):
                layer.train(mode)
