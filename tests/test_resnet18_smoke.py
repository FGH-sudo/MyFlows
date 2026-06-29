# -*- coding: utf-8 -*-
"""ResNet-18 烟雾测试 + BatchNorm/GAP 梯度检查。"""

import sys
from pathlib import Path

PROJECT_PARENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_PARENT))

import numpy as np

from MyFlows.core.node import Variable
from MyFlows.core.graph import Graph
from MyFlows.layers.resnet import (
    BatchNorm2d, GlobalAvgPool2d, BasicBlock, ResNet18,
)
from MyFlows.ops.batchnorm import BatchNorm2d_Op, GlobalAvgPool2d_Op


def _numerical_grad(f, x, eps=1e-4):
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        fp = f()
        x[idx] = orig - eps
        fm = f()
        x[idx] = orig
        grad[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


def test_batchnorm2d_forward_shape():
    np.random.seed(0)
    x = Variable(np.random.randn(2, 4, 5, 5), name="x")
    bn = BatchNorm2d(4)
    out = bn(x)
    g = Graph(out)
    g.forward()
    assert out.value.shape == (2, 4, 5, 5)
    # 归一化后每通道均值近似 0
    assert np.allclose(out.value.mean(axis=(0, 2, 3)), np.zeros(4), atol=1e-6)


def test_batchnorm2d_gradcheck():
    """对 BN 的 gamma/beta/x 做数值梯度检查(训练模式)。"""
    np.random.seed(1)
    N, C, H, W = 2, 3, 3, 3
    x_val = np.random.randn(N, C, H, W).astype(np.float64)
    gamma_val = np.random.randn(C).astype(np.float64) * 0.5 + 1.0
    beta_val = np.random.randn(C).astype(np.float64) * 0.1

    x_node = Variable(x_val.copy(), name="x")
    gamma_node = Variable(gamma_val.copy(), trainable=True, name="gamma")
    beta_node = Variable(beta_val.copy(), trainable=True, name="beta")
    running_mean = np.zeros((C,), dtype=np.float64)
    running_var = np.ones((C,), dtype=np.float64)

    op = BatchNorm2d_Op(x_node, gamma_node, beta_node,
                        running_mean, running_var, momentum=0.0, training=True)

    # 使用 sum() 当标量目标以便比较梯度
    class _SumNode:
        pass

    from MyFlows.core.node import Node

    class SumAll(Node):
        def forward(self, v):
            self.value = np.sum(v)
        def backward(self):
            self.parents[0].grad += np.ones_like(self.parents[0].value) * self.grad

    target = SumAll(op)
    g = Graph(target)

    def loss_fn():
        # 手动模拟 forward(考虑 running stats 会被就地更新, 用 momentum=0 避免影响)
        running_mean[:] = 0.0
        running_var[:] = 1.0
        g.forward()
        return float(target.value)

    g.forward()
    g.backward()
    grad_x = x_node.grad.copy()
    grad_gamma = gamma_node.grad.copy()
    grad_beta = beta_node.grad.copy()

    num_grad_gamma = _numerical_grad(loss_fn, gamma_node.value)
    num_grad_beta = _numerical_grad(loss_fn, beta_node.value)
    num_grad_x = _numerical_grad(loss_fn, x_node.value)

    assert np.allclose(grad_gamma, num_grad_gamma, atol=1e-4), \
        f"dgamma mismatch: max diff {np.max(np.abs(grad_gamma - num_grad_gamma))}"
    assert np.allclose(grad_beta, num_grad_beta, atol=1e-4), \
        f"dbeta mismatch: max diff {np.max(np.abs(grad_beta - num_grad_beta))}"
    assert np.allclose(grad_x, num_grad_x, atol=1e-3), \
        f"dx mismatch: max diff {np.max(np.abs(grad_x - num_grad_x))}"


def test_globalavgpool2d_gradcheck():
    np.random.seed(2)
    x_val = np.random.randn(2, 3, 4, 4).astype(np.float64)
    x_node = Variable(x_val.copy(), name="x")
    op = GlobalAvgPool2d_Op(x_node)

    from MyFlows.core.node import Node

    class SumAll(Node):
        def forward(self, v):
            self.value = np.sum(v)
        def backward(self):
            self.parents[0].grad += np.ones_like(self.parents[0].value) * self.grad

    target = SumAll(op)
    g = Graph(target)
    g.forward()
    g.backward()

    expected = np.ones_like(x_val) / (x_val.shape[2] * x_val.shape[3])
    assert np.allclose(x_node.grad, expected, atol=1e-8)


def test_basicblock_forward_backward():
    np.random.seed(3)
    x = Variable(np.random.randn(2, 16, 8, 8), name="x")
    block = BasicBlock(16, 32, stride=2)
    out = block(x)

    from MyFlows.core.node import Node

    class SumAll(Node):
        def forward(self, v):
            self.value = np.sum(v)
        def backward(self):
            self.parents[0].grad += np.ones_like(self.parents[0].value) * self.grad

    target = SumAll(out)
    g = Graph(target)
    g.forward()
    assert out.value.shape == (2, 32, 4, 4)

    g.backward()
    # 所有可训练参数应拿到梯度
    for p in block.params:
        assert p.grad is not None
        assert p.grad.shape == p.value.shape


def test_resnet18_cifar_forward():
    np.random.seed(4)
    model = ResNet18(in_channels=3, output_dim=10, stem="cifar")
    x = Variable(np.random.randn(2, 3, 32, 32), name="x")
    out = model(x)
    g = Graph(out)
    g.forward()
    assert out.value.shape == (2, 10), f"got {out.value.shape}"
    assert model.output_dim == 10


def test_resnet18_rejects_num_classes_keyword():
    try:
        ResNet18(in_channels=3, num_classes=10, stem="cifar")
    except TypeError:
        return
    raise AssertionError("ResNet18 should use output_dim instead of num_classes")


def test_resnet18_train_eval_toggle():
    model = ResNet18(in_channels=3, output_dim=10, stem="cifar")
    x = Variable(np.random.randn(1, 3, 32, 32), name="x")
    out = model(x)
    Graph(out).forward()

    model.eval()
    # 切换后所有 BN 层 training 应为 False
    for layer in model.sub_layers:
        if isinstance(layer, BatchNorm2d):
            assert layer.training is False
            if layer._last_op is not None:
                assert layer._last_op.training is False

    model.train()
    for layer in model.sub_layers:
        if isinstance(layer, BatchNorm2d):
            assert layer.training is True


if __name__ == "__main__":
    test_batchnorm2d_forward_shape(); print("OK: bn forward shape")
    test_batchnorm2d_gradcheck(); print("OK: bn gradcheck")
    test_globalavgpool2d_gradcheck(); print("OK: gap gradcheck")
    test_basicblock_forward_backward(); print("OK: basicblock")
    test_resnet18_cifar_forward(); print("OK: resnet18 cifar forward")
    test_resnet18_rejects_num_classes_keyword(); print("OK: num_classes rejected")
    test_resnet18_train_eval_toggle(); print("OK: train/eval toggle")
    print("All tests passed.")
