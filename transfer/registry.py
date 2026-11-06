"""Registered torchvision-compatible backbones and their converted source manifests."""

from __future__ import annotations

import json
from pathlib import Path

TORCHVISION_WEIGHTS = {
    "resnet18": {"file": "resnet18-f37072fd.pth", "url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
                 "weights": "ResNet18_Weights.IMAGENET1K_V1"},
    "mobilenet_v2": {"file": "mobilenet_v2-b0353104.pth",
                     "url": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
                     "weights": "MobileNet_V2_Weights.IMAGENET1K_V1"},
    "alexnet": {"file": "alexnet-owt-7be5be79.pth", "url": "https://download.pytorch.org/models/alexnet-owt-7be5be79.pth",
                "weights": "AlexNet_Weights.IMAGENET1K_V1"},
}
ARCHS = tuple(TORCHVISION_WEIGHTS)


def build_model(arch, output_dim=1000, *, dropout=None, initializer=None, conv_backend="auto", dropout_seed=None,
                input_norm=None, name=None):
    """按名称构造与 torchvision 结构一致的模型。dropout=None 时用 torchvision 默认值。"""
    from ..layers.mobilenet import AlexNet, MobileNetV2
    from ..layers.resnet import ResNet18

    common = dict(initializer=initializer, conv_backend=conv_backend, dropout_seed=dropout_seed, input_norm=input_norm,
                  name=name)
    if arch == "resnet18":
        return ResNet18(in_channels=3, output_dim=output_dim, stem="imagenet", base_width=64,
                        dropout=0.0 if dropout is None else dropout, compat="torchvision", **common)
    if arch == "mobilenet_v2":
        return MobileNetV2(output_dim=output_dim, dropout=0.2 if dropout is None else dropout, **common)
    if arch == "alexnet":
        return AlexNet(output_dim=output_dim, dropout=0.5 if dropout is None else dropout, **common)
    raise ValueError(f"unknown arch {arch!r}; expected one of {ARCHS}")


class PretrainedModelRegistry:
    STAGES = {"resnet18": ("stem", "layer1", "layer2", "layer3", "layer4"),
              "mobilenet_v2": ("stem", "stage2", "stage3", "stage4", "stage5"),
              "alexnet": tuple(f"conv{i}" for i in range(1, 6))}
    HEADS = {"resnet18": "fc", "mobilenet_v2": "classifier.1", "alexnet": "classifier.6"}

    def __init__(self, directory=None):
        self.directory = Path(directory) if directory is not None else Path(__file__).resolve().parents[2] / "runs/pretrained/converted"

    def describe(self, arch):
        if arch not in TORCHVISION_WEIGHTS:
            raise ValueError(f"unknown architecture {arch!r}")
        path = self.directory / f"{arch}.npz"
        manifest_path = path.with_suffix(".json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        return {"arch": arch, **TORCHVISION_WEIGHTS[arch], "source_sha256": manifest.get("source_sha256"),
                "npz": str(path), "npz_sha256": manifest.get("npz_sha256"), "installed": path.is_file(),
                "input_size": (224, 224), "mean": (.485, .456, .406), "std": (.229, .224, .225),
                "stages": self.STAGES[arch], "head": self.HEADS[arch]}

    def build(self, arch, output_dim, **kwargs):
        return build_model(arch, output_dim, **kwargs)
