# -*- coding: utf-8 -*-
"""JSON+NPZ checkpoint 与可选 ONNX 导出测试。"""

import sys
import tempfile
import json
import unittest
from pathlib import Path

PROJECT_PARENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_PARENT))

import numpy as np

from MyFlows import Dense, Variable, Graph, save_checkpoint, load_checkpoint, export_onnx
from MyFlows.layers.resnet import BatchNorm2d


def test_json_npz_checkpoint_roundtrip():
    np.random.seed(0)
    layer = Dense(3, 2, name="dense")
    original_w = layer.W.value.copy()
    original_b = layer.b.value.copy()

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "model_checkpoint"
        json_path, npz_path = save_checkpoint([layer], epoch=7, acc=0.8, filepath=base)

        assert Path(json_path).exists()
        assert Path(npz_path).exists()
        assert not (Path(tmp) / "model_checkpoint.pkl").exists()

        layer.W.value = np.zeros_like(layer.W.value)
        layer.b.value = np.ones_like(layer.b.value)

        epoch, saved_score = load_checkpoint([layer], filepath=base)
        assert epoch == 7
        assert saved_score == 0.8
        assert np.allclose(layer.W.value, original_w)
        assert np.allclose(layer.b.value, original_b)


class CheckpointScoreMetadataTest(unittest.TestCase):
    def test_json_npz_checkpoint_stores_loss_without_accuracy_field(self):
        layer = Dense(3, 2, name="dense")

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "regression_checkpoint"
            json_path, _ = save_checkpoint([layer], epoch=3, loss=0.125, filepath=base)

            with Path(json_path).open("r", encoding="utf-8") as f:
                metadata = json.load(f)

            self.assertEqual(metadata["score_name"], "loss")
            self.assertEqual(metadata["score_value"], 0.125)
            self.assertEqual(metadata["best_loss"], 0.125)
            self.assertNotIn("best_acc", metadata)

            epoch, best_loss = load_checkpoint([layer], filepath=base)
            self.assertEqual(epoch, 3)
            self.assertEqual(best_loss, 0.125)


def test_batchnorm_buffers_checkpoint_roundtrip():
    bn = BatchNorm2d(3, name="bn")
    bn.running_mean[:] = np.array([1.0, 2.0, 3.0])
    bn.running_var[:] = np.array([4.0, 5.0, 6.0])

    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "bn_checkpoint"
        save_checkpoint([bn], epoch=1, filepath=base)

        bn.running_mean[:] = 0.0
        bn.running_var[:] = 1.0
        load_checkpoint([bn], filepath=base)

        assert np.allclose(bn.running_mean, np.array([1.0, 2.0, 3.0]))
        assert np.allclose(bn.running_var, np.array([4.0, 5.0, 6.0]))


def test_optional_onnx_export_linear_graph():
    try:
        import onnx  # noqa: F401
    except ImportError:
        return

    np.random.seed(1)
    x = Variable(np.random.randn(2, 3), name="input")
    layer = Dense(3, 2, name="dense")
    y = layer(x)
    graph = Graph(y)
    graph.forward()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "linear.onnx"
        export_onnx(graph, path, input_nodes=[x], output_names=["output"])
        assert path.exists()


if __name__ == "__main__":
    test_json_npz_checkpoint_roundtrip()
    print("OK: json+npz checkpoint")
    test_batchnorm_buffers_checkpoint_roundtrip()
    print("OK: batchnorm buffers")
    test_optional_onnx_export_linear_graph()
    print("OK: optional onnx export")
