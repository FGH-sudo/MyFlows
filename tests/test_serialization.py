# -*- coding: utf-8 -*-
"""JSON+NPZ checkpoint 与可选 ONNX 导出测试。"""

import sys
import tempfile
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_PARENT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_PARENT))

import numpy as np

from MyFlows import Dense, Variable, Graph, save_checkpoint, load_checkpoint, export_onnx
from MyFlows.layers.resnet import BatchNorm2d
from MyFlows.utils.checkpoint import (TrainingStateError, list_training_states, load_training_state, read_checkpoint_metadata,
                                      save_training_state, verify_training_state)


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


class AdaptivePoolExportTest(unittest.TestCase):
    def test_adaptive_pool_export_matches_rectangular_bin_means(self):
        import importlib.util
        import math

        if not all(importlib.util.find_spec(m) for m in ("onnx", "onnxruntime")):
            self.skipTest("ONNX and ONNX Runtime are optional")
        import onnxruntime as ort

        from MyFlows.core.device import asnumpy, xp
        from MyFlows.ops.pooling import AdaptiveAvgPool2d_Op

        rng = np.random.default_rng(39)
        for shape, target in (((2, 3, 7, 11), (3, 4)), ((1, 2, 2, 3), (5, 4)), ((1, 2, 3, 4), (3, 4))):
            with self.subTest(shape=shape, target=target), tempfile.TemporaryDirectory() as tmp:
                x = Variable(xp.asarray(rng.normal(size=shape).astype(np.float32)), name="image")
                y = AdaptiveAvgPool2d_Op(x, target)
                graph = Graph(y)
                graph.forward()
                path = Path(tmp) / "pool.onnx"
                export_onnx(graph, path, input_nodes=[x], output_names=["pooled"])
                session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                sample = rng.normal(size=shape).astype(np.float32)
                expected = np.empty((*shape[:2], *target), np.float32)
                for i in range(target[0]):
                    for j in range(target[1]):
                        patch = sample[:, :, math.floor(i*shape[2]/target[0]):math.ceil((i+1)*shape[2]/target[0]),
                                       math.floor(j*shape[3]/target[1]):math.ceil((j+1)*shape[3]/target[1])]
                        expected[:, :, i, j] = patch.mean(axis=(2, 3))
                actual = session.run(None, {session.get_inputs()[0].name: sample})[0]
                x.value = xp.asarray(sample)
                graph.forward()
                np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
                np.testing.assert_allclose(actual, asnumpy(y.value), rtol=2e-6, atol=2e-6)


class TrainingStateBestCheckpointTest(unittest.TestCase):
    def test_resume_restores_the_best_at_the_snapshot_not_a_later_best(self):
        layer = Dense(3, 2, name="dense")
        best_weights = layer.W.value.copy()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            best = root / "historical_best"
            save_checkpoint([layer], epoch=0, loss=0.25, filepath=best)
            layer.W.value += 1
            state_weights = layer.W.value.copy()
            state = save_training_state(root / "states", [layer], None, 1, {"best_loss": 0.25}, best_filepath=best)
            layer.W.value += 1
            save_checkpoint([layer], epoch=1, loss=0.1, filepath=best)

            restored_best = root / "resumed" / "selected"
            load_training_state(state, [layer], None, best_filepath=restored_best)
            np.testing.assert_array_equal(layer.W.value, state_weights)
            metadata = read_checkpoint_metadata(restored_best)
            self.assertEqual((metadata["score_value"], metadata["array_file"]), (0.25, "selected.npz"))
            load_checkpoint([layer], filepath=restored_best)
            np.testing.assert_array_equal(layer.W.value, best_weights)

            data = bytearray((state / "best.npz").read_bytes())
            data[len(data) // 2] ^= 1
            (state / "best.npz").write_bytes(data)
            with self.assertRaises(TrainingStateError):
                verify_training_state(state)

    def test_a_state_missing_its_historical_best_is_not_silently_resumed(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = save_training_state(tmp, [], None, 1, {"best_loss": 0.25})
            with self.assertRaisesRegex(TrainingStateError, "历史最佳模型"):
                load_training_state(state, [], None, best_filepath=Path(tmp) / "best")


class TrainingStatePublicationTest(unittest.TestCase):
    def test_retrying_a_step_keeps_the_previous_snapshot_if_publication_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            original = save_training_state(directory, [], None, 4, {"epoch": 1})
            before = (directory / "latest.json").read_bytes()
            replace = os.replace

            def fail_on_publication(source, target):
                if Path(source).name.startswith(".staging-"):
                    raise OSError("simulated publication failure")
                return replace(source, target)

            with patch("MyFlows.utils.checkpoint.os.replace", side_effect=fail_on_publication):
                with self.assertRaisesRegex(OSError, "publication failure"):
                    save_training_state(directory, [], None, 4, {"epoch": 2})
            self.assertEqual((directory / "latest.json").read_bytes(), before)
            self.assertEqual(verify_training_state(original)["step"], 4)

            updated = save_training_state(directory, [], None, 4, {"epoch": 2})
            self.assertNotEqual(updated, original)
            self.assertEqual(list_training_states(directory)[0]["path"], updated)
            self.assertEqual(json.loads((directory / "latest.json").read_bytes())["state"], updated.name)
            (updated / "checkpoint.npz").write_bytes(b"corrupt")
            self.assertEqual(list_training_states(directory)[0]["path"], original)

    def test_concurrent_publications_of_the_same_step_have_distinct_valid_paths(self):
        from concurrent.futures import ThreadPoolExecutor

        with tempfile.TemporaryDirectory() as tmp, ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(save_training_state, tmp, [], None, 7, {"epoch": 1})
            second = pool.submit(save_training_state, tmp, [], None, 7, {"epoch": 2})
            paths = {first.result(timeout=15), second.result(timeout=15)}
            self.assertEqual(len(paths), 2)
            self.assertTrue(all(verify_training_state(path)["step"] == 7 for path in paths))
            latest = json.loads((Path(tmp) / "latest.json").read_bytes())
            self.assertEqual(list_training_states(tmp)[0]["path"].name, latest["state"])

    def test_repeated_step_zero_prunes_older_revisions_and_keeps_newest(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            paths = [save_training_state(directory, [], None, 0, {"epoch": i}, keep=2) for i in range(4)]
            step_dirs = [p for p in directory.iterdir() if p.is_dir() and p.name.startswith("step-")]
            self.assertEqual(len(step_dirs), 2)
            self.assertEqual(set(step_dirs), set(paths[-2:]))
            states = list_training_states(directory)
            self.assertEqual(len(states), 1)
            self.assertEqual(states[0]["path"], paths[-1])
            self.assertEqual(states[0]["state"]["epoch"], 3)

    def test_saving_smaller_step_after_rollback_keeps_base_and_prunes_discarded_future(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            for step in (0, 10, 20, 30):
                save_training_state(directory, [], None, step, {"epoch": 0, "global_step": step}, keep=3)
            save_training_state(directory, [], None, 15, {"epoch": 0, "global_step": 15}, keep=2)
            self.assertEqual([s["step"] for s in list_training_states(directory)], [0, 10, 15])


class OnnxModelsExportTest(unittest.TestCase):
    def _skip_without_onnx(self):
        import importlib.util

        if not (importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime")):
            self.skipTest("onnx and onnxruntime are required")

    def test_mobilenet_v2_exports_with_relu6_and_matches_ort(self):
        self._skip_without_onnx()
        import onnxruntime as ort
        from MyFlows.layers.mobilenet import MobileNetV2
        from MyFlows.utils.initializers import make_initializer

        model = MobileNetV2(output_dim=4, dropout=0.0, initializer=make_initializer("kaiming_normal", seed=3))
        model.eval()
        rng = np.random.default_rng(11)
        x = Variable(rng.normal(size=(1, 3, 64, 64)).astype(np.float32), name="input", requires_grad=False)
        graph = Graph(model(x), optimize=False)
        graph.forward()
        expected = graph.target_node.value.copy()
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "mobilenet_v2.onnx"
            export_onnx(graph, onnx_path, input_nodes=[x], output_names=["output"])
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            (got,) = session.run(None, {session.get_inputs()[0].name: x.value})
            np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)

    def test_optimized_resnet18_exports_and_matches_ort(self):
        self._skip_without_onnx()
        import onnxruntime as ort
        from MyFlows import ResNet18
        from MyFlows.utils.initializers import make_initializer

        model = ResNet18(in_channels=3, output_dim=2, stem="cifar", base_width=8,
                         initializer=make_initializer("kaiming_normal", seed=5))
        model.eval()
        rng = np.random.default_rng(12)
        x = Variable(rng.normal(size=(1, 3, 32, 32)).astype(np.float32), name="input", requires_grad=False)
        graph = Graph(model(x), optimize=True)
        graph.forward()
        expected = graph.target_node.value.copy()
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "resnet18_opt.onnx"
            export_onnx(graph, onnx_path, input_nodes=[x], output_names=["output"])
            session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            (got,) = session.run(None, {session.get_inputs()[0].name: x.value})
            np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)


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
