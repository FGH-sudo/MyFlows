"""Task registry for synthetic, MNIST MLP and DonkeyCar CNN adapters."""

KNOWN_TASKS = ("synthetic", "mnist_mlp", "donkey_cnn", "donkey_resnet18")


def build_task(name):
    name = str(name)
    if name == "synthetic":
        from .synthetic import SyntheticTask
        return SyntheticTask()
    if name == "mnist_mlp":
        from .mnist_mlp import MnistMlpTask
        return MnistMlpTask()
    if name == "donkey_cnn":
        from .donkey_cnn import DonkeyCnnTask
        return DonkeyCnnTask()
    if name == "donkey_resnet18":
        from .donkey_resnet18 import DonkeyResNet18Task
        return DonkeyResNet18Task()
    raise ValueError(f"unknown task {name}")
