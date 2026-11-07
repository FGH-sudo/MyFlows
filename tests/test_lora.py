"""Low-rank updates, frozen base tensors and ordinary inference after merging."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

import MyFlows as ms
from MyFlows.core.device import get_device, set_device
from MyFlows.layers.layer import Conv2D, Dense
from MyFlows.transfer.lora import LoRAConv2d, LoRADense, attach_resnet_lora, merge_resnet_lora


class LoRATest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(set_device, get_device())
        set_device("cpu")
        self.rng = np.random.default_rng(34)

    @staticmethod
    def evaluate(graph):
        graph.forward()
        return graph.target_node.value.copy()

    def check_layer(self, base, wrap, shape):
        x = ms.Variable(self.rng.normal(size=shape).astype(np.float32), requires_grad=False)
        original = self.evaluate(ms.Graph(base(x)))
        layer = wrap(base, 3, alpha=5, seed=7)
        graph = ms.Graph(layer(x))
        self.assertEqual(self.evaluate(graph).tobytes(), original.tobytes())
        graph.backward()
        self.assertEqual(np.count_nonzero(layer.A.grad), 0)
        self.assertGreater(np.linalg.norm(layer.B.grad), 0)
        self.assertTrue(all(p.grad is None for p in base.params))
        layer.B.value[...] = self.rng.normal(scale=0.02, size=layer.B.value.shape)
        expected = self.evaluate(graph)
        self.assertGreater(np.max(np.abs(expected - original)), 1e-5)
        layer.merge()
        merged = self.evaluate(ms.Graph(layer(x)))
        np.testing.assert_allclose(merged, expected, atol=2e-6, rtol=2e-6)
        layer.merge()
        self.assertEqual(self.evaluate(ms.Graph(layer(x))).tobytes(), merged.tobytes())

    def test_dense_update(self):
        self.check_layer(
            Dense(5, 7, initializer=ms.make_initializer("kaiming_normal", seed=2)), LoRADense, (4, 5)
        )

    def test_strided_padded_conv_update(self):
        self.check_layer(
            Conv2D(
                3,
                4,
                3,
                stride=2,
                padding=1,
                bias=False,
                initializer=ms.make_initializer("kaiming_normal", seed=2),
            ),
            LoRAConv2d,
            (2, 3, 11, 13),
        )

    @unittest.skipUnless(
        importlib.util.find_spec("onnxruntime") and importlib.util.find_spec("onnx"),
        "ONNX and ONNX Runtime are required for deployment parity",
    )
    def test_merged_resnet_one_input_and_onnx_inference(self):
        import onnxruntime as ort

        model = ms.ResNet18(
            3,
            2,
            base_width=64,
            stem="imagenet",
            compat="torchvision",
            input_norm="imagenet",
            dropout=0.2,
            initializer=ms.make_initializer("kaiming_normal", seed=4),
        )
        x = ms.Variable(self.rng.uniform(size=(1, 3, 32, 40)).astype(np.float32), requires_grad=False)
        model.eval()
        base = self.evaluate(ms.Graph(model(x)))
        info = attach_resnet_lora(model, 4, seed=8)
        self.assertEqual(len(info["targets"]), 6)
        self.assertEqual(sum(p.trainable for p in model.params), 12)
        graph = ms.Graph(model(x))
        self.assertEqual(self.evaluate(graph).tobytes(), base.tobytes())
        for p in model.params:
            if p.trainable and p.name.endswith("_B"):
                p.value[...] = self.rng.normal(scale=0.001, size=p.value.shape)
        expected = self.evaluate(graph)
        merge_resnet_lora(model)
        merge_resnet_lora(model)
        merged = ms.Graph(model(x))
        np.testing.assert_allclose(self.evaluate(merged), expected, atol=3e-5, rtol=2e-5)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "merged.onnx"
            ms.export_onnx(merged, path)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
            self.assertEqual(len(session.get_inputs()), 1)
            (output,) = session.run(None, {session.get_inputs()[0].name: x.value})
            np.testing.assert_allclose(output, expected, atol=3e-5, rtol=2e-5)
