"""Task heads operating on a batch of backbone features."""

from ..layers.layer import Dense, Dropout, Layer
from ..ops.activation import ReLU
from ..utils.initializers import make_initializer


class Head(Layer):
    def __init__(self, layers):
        super().__init__(name="task_head")
        self.sub_layers = list(layers)
        self.params = [p for layer in layers for p in layer.params]

    def forward(self, features):
        for layer in self.sub_layers:
            features = layer(features)
        return features

    def train(self, mode=True):
        for layer in self.sub_layers:
            if hasattr(layer, "train"):
                layer.train(mode)


class HeadAdapterBuilder:
    @staticmethod
    def build(input_dim, output_dim, *, kind="linear", hidden=(512, 512), dropout=.3, initializer=None,
              backend="auto", seed=None):
        if kind not in ("linear", "mlp", "regression"):
            raise ValueError("head must be linear, mlp or regression")
        widths = list(hidden) if kind == "mlp" else []
        if min(input_dim, output_dim, *widths) < 1 or not 0 <= dropout < 1 or (kind == "mlp" and not widths):
            raise ValueError("invalid head dimensions or dropout")
        init = make_initializer(initializer)
        layers = []
        for i, width in enumerate(widths):
            layers.append(Dense(input_dim, width, activation=ReLU, initializer=init, backend=backend, name=f"hidden{i}"))
            if dropout:
                layers.append(Dropout(dropout, seed=None if seed is None else seed + i, name=f"head_dropout{i}"))
            input_dim = width
        layers.append(Dense(input_dim, output_dim, initializer=init, backend=backend, name="output"))
        return Head(layers)
