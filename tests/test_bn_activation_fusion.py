import unittest

import numpy as np

import MyFlows as ms
from MyFlows.core.device import asnumpy, cuda_available, get_device, set_device
from MyFlows.layers.resnet import BatchNorm2d


class BNActivationTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(set_device, get_device())

    def trajectory(self, optimized, residual_shape, *, retain=False):
        rng = np.random.default_rng(271)
        x = ms.Variable(rng.normal(size=(2, 4, 3, 5)).astype(np.float32), trainable=True)
        bn = BatchNorm2d(4, unbiased_running_var=True)
        base = bn(x)
        residual = ms.Variable(rng.normal(size=residual_shape).astype(np.float32), trainable=True) if residual_shape else None
        total = ms.Add(base, residual) if residual else base
        output = ms.ReLU(total)
        coeff = ms.Variable(rng.normal(size=x.value.shape).astype(np.float32), requires_grad=False)
        graph = ms.Graph(ms.MSELoss(output, coeff), optimize=optimized)
        if retain:
            graph.retain_grad(base, output)
        states = []
        self.assertEqual(len(graph._bn_groups), int(optimized))
        for training in (True, False, True):
            bn.train(training)
            graph.forward()
            graph.backward()
            values = [output.value, graph.target_node.value, bn.running_mean, bn.running_var,
                      x.grad, bn.gamma.grad, bn.beta.grad]
            if residual:
                values.append(residual.grad)
            if retain:
                values.extend((base.value, base.grad, output.grad))
            elif optimized and (residual_shape is None or residual_shape == x.value.shape):
                self.assertIsNone(base.value)
            states.extend(asnumpy(v).tobytes() for v in values)
        return states

    def compare(self, device):
        set_device(device)
        for shape in (None, (2, 4, 3, 5), (4,)):
            for retain in (False, True):
                with self.subTest(device=device, residual=shape, retain=retain):
                    self.assertEqual(self.trajectory(False, shape, retain=retain), self.trajectory(True, shape, retain=retain))

    def test_cpu(self):
        self.compare("cpu")

    def test_shape_and_content_inspection_materializes_values_for_one_step(self):
        from MyFlows.utils.model_inspector import inspect_graph
        set_device("cpu")
        x = ms.Variable(np.ones((2, 3, 3, 3), np.float32), trainable=True)
        bn = BatchNorm2d(3)(x)
        graph = ms.Graph(ms.ReLU(bn), optimize=True)
        graph.materialize_intermediates()
        graph.forward()
        self.assertTrue(inspect_graph(graph, check_shape=True, check_content=True)["ok"])
        graph.backward()
        graph.materialize_intermediates(False)
        graph.forward()
        self.assertIsNone(bn.value)

    @unittest.skipUnless(cuda_available(), "CUDA is required for the fused reduction and epilogue")
    def test_cuda(self):
        self.compare("cuda")

    def test_shared_bn_outputs_and_reused_running_statistics_are_not_fused(self):
        set_device("cpu")
        x = ms.Variable(np.ones((2, 3, 3, 3), np.float32), trainable=True)
        layer = BatchNorm2d(3)
        bn = layer(x)
        self.assertFalse(ms.Graph(ms.Add(ms.ReLU(bn), bn), optimize=True)._bn_groups)
        self.assertFalse(ms.Graph(ms.Add(ms.ReLU(bn), ms.ReLU(layer(x))), optimize=True)._bn_groups)
