"""Registry provenance, atomic checkpoint preflight, feature adapters and independent freeze modes."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

import MyFlows as ms
from MyFlows.core.device import get_device, set_device
from MyFlows.layers.layer import Dense
from MyFlows.transfer import CheckpointLoader, FeatureAdapter, HeadAdapterBuilder, ParameterFreezer, PretrainedModelRegistry, TransferLearningManager
from MyFlows.transfer.freeze import parse_schedule
from MyFlows.transfer.loader import PretrainedLoadError


class TransferComponentsTest(unittest.TestCase):
    def setUp(self):
        self.device = get_device()
        set_device("cpu")

    def tearDown(self):
        set_device(self.device)

    def test_registry_separates_available_artifacts_from_source_identity(self):
        with TemporaryDirectory() as tmp:
            reg = PretrainedModelRegistry(tmp)
            self.assertFalse(reg.describe("resnet18")["installed"])
            (Path(tmp) / "resnet18.json").write_text(json.dumps({"source_sha256": "fixture-source", "npz_sha256": "fixture-npz"}), encoding="utf-8")
            entry = reg.describe("resnet18")
            self.assertEqual((entry["source_sha256"], entry["head"], entry["stages"][-1]), ("fixture-source", "fc", "layer4"))
            self.assertEqual(reg.describe("alexnet")["input_size"], (224, 224))

    def test_checkpoint_preflight_is_atomic_on_late_shape_error(self):
        with TemporaryDirectory() as tmp:
            model = Dense(2, 3, name="head")
            path = Path(tmp) / "ckpt"
            ms.save_checkpoint([model], None, 2, filepath=str(path))
            loader = CheckpointLoader()
            self.assertEqual(loader.load_checkpoint(model, path)["loaded"], 2)
            with np.load(path.with_suffix(".npz")) as arrays:
                contents = {k: arrays[k] for k in arrays.files}
            key = next(k for k in contents if k.startswith("param_0001"))
            contents[key] = np.ones((7,), dtype=np.float32)
            np.savez(path.with_suffix(".npz"), **contents)
            model.W.value.fill(42)
            with self.assertRaises(PretrainedLoadError):
                loader.load_checkpoint(model, path)
            np.testing.assert_array_equal(model.W.value, np.full_like(model.W.value, 42))

    def test_adapter_projects_spatial_maps_and_mlp_is_deterministic_in_eval(self):
        x = ms.Variable(np.arange(48, dtype=np.float32).reshape(2, 2, 3, 4), requires_grad=False)
        adapter = FeatureAdapter(2, spatial_size=(3, 4), pooling="flatten", projection=1, batch_norm=True)
        head = HeadAdapterBuilder.build(adapter.output_dim, 2, kind="mlp", hidden=(5, 4), dropout=.3, seed=1)
        feature = adapter(x)
        graph = ms.Graph(head(feature))
        head.train(False)
        adapter.train(False)
        graph.forward()
        first = graph.target_node.value.copy()
        graph.forward()
        np.testing.assert_array_equal(first, graph.target_node.value)
        self.assertEqual(feature.value.shape, (2, 12))
        graph.backward()
        self.assertTrue(all(p.grad is not None for p in head.params + adapter.params))

    def test_named_parameters_and_bn_statistics_can_be_frozen_independently(self):
        model = ms.ResNet18(output_dim=2, base_width=4, stem="cifar")
        freezer = ParameterFreezer(model)
        freezer.stages(2, mode="params")
        self.assertFalse(model.stem_bn.gamma.trainable)
        self.assertFalse(model.stem_bn.frozen)
        freezer.stages(2, mode="bn")
        self.assertTrue(model.stem_bn.frozen)
        freezer.by_name("stem_bn.*", frozen=False, mode="params")
        self.assertTrue(model.stem_bn.gamma.trainable)
        self.assertTrue(model.stem_bn.frozen)
        freezer.stages(0, bn_unfreeze="keep-frozen")
        self.assertTrue(model.stem_bn.frozen)
        freezer.stages(0, bn_unfreeze="follow")
        self.assertFalse(model.stem_bn.frozen)
        self.assertEqual(len(freezer.names), len(set(model.params)))
        self.assertEqual(parse_schedule("0:4,2:3,4:2"), {0: 4, 2: 3, 4: 2})
        with self.assertRaises(ValueError):
            parse_schedule("2:3,2:1")

    def test_manager_attaches_the_requested_head_and_adapter(self):
        manager = TransferLearningManager.build("resnet18", 3, freeze_stages=5, head="mlp", head_hidden=(4,),
                                               feature_adapter={"projection": 4, "batch_norm": True},
                                               initializer=ms.make_initializer("kaiming_normal", seed=4), head_dropout=0.)
        model = manager.model
        x = ms.Variable(np.ones((2, 3, 32, 32), np.float32), requires_grad=False)
        graph = ms.Graph(model(x))
        graph.forward()
        self.assertEqual(graph.target_node.value.shape, (2, 3))
        self.assertNotIn(model.fc_W, graph.nodes)
        graph.backward()
        self.assertTrue(all(p.grad is not None for p in model.transfer_head.params))
        self.assertIsNone(model.stem_conv.kernel.grad)
        groups = manager.parameter_groups()
        flat = [p for group in groups for p in group["params"]]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertEqual(set(flat), set(model.params))
