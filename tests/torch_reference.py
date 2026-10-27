"""torch.nn re-implementations of the torchvision ResNet18 / MobileNetV2 / AlexNet definitions.

torchvision is not installed here (it would have to match the separately installed torch wheel); these modules
follow torchvision's definitions and load the official state_dicts with ``strict=True``, so every name and shape
matches the official weights. Used by the E2 tests and ``tools/pretrained_report.py``.
"""

import torch
from torch import nn
import torch.nn.functional as F


class TorchBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, 1, stride, bias=False),
                                            nn.BatchNorm2d(planes))

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        identity = self.downsample(x) if self.downsample is not None else x
        return self.relu(out + identity)

class TorchResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        inplanes, layers = 64, []
        for planes, stride in ((64, 1), (128, 2), (256, 2), (512, 2)):
            layers.append(nn.Sequential(TorchBasicBlock(inplanes, planes, stride), TorchBasicBlock(planes, planes)))
            inplanes = planes
        self.layer1, self.layer2, self.layer3, self.layer4 = layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))

def _conv_bn_relu6(inp, oup, kernel=3, stride=1, groups=1):
    return nn.Sequential(nn.Conv2d(inp, oup, kernel, stride, (kernel - 1) // 2, groups=groups, bias=False),
                         nn.BatchNorm2d(oup), nn.ReLU6(inplace=True))

class TorchInvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        hidden = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup
        layers = [] if expand_ratio == 1 else [_conv_bn_relu6(inp, hidden, kernel=1)]
        layers += [_conv_bn_relu6(hidden, hidden, stride=stride, groups=hidden),
                   nn.Conv2d(hidden, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup)]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv(x) if self.use_res_connect else self.conv(x)

class TorchMobileNetV2(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        features, inp = [_conv_bn_relu6(3, 32, stride=2)], 32
        for t, c, n, s in ((1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1),
                           (6, 160, 3, 2), (6, 320, 1, 1)):
            for i in range(n):
                features.append(TorchInvertedResidual(inp, c, s if i == 0 else 1, t))
                inp = c
        features.append(_conv_bn_relu6(inp, 1280, kernel=1))
        self.features = nn.Sequential(*features)
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, num_classes))

    def forward(self, x):
        x = F.adaptive_avg_pool2d(self.features(x), (1, 1))
        return self.classifier(torch.flatten(x, 1))

class TorchAlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 11, 4, 2), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 192, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2),
            nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True),
            nn.Dropout(0.5), nn.Linear(4096, 4096), nn.ReLU(inplace=True), nn.Linear(4096, num_classes))

    def forward(self, x):
        return self.classifier(torch.flatten(self.avgpool(self.features(x)), 1))

TORCH_MODELS = {"resnet18": TorchResNet18, "mobilenet_v2": TorchMobileNetV2, "alexnet": TorchAlexNet}
