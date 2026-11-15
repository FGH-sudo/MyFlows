"""Regression tests for inputs the framework used to mishandle: operands of other ranks and broadcast
shapes in dot products and losses, malformed labels, damaged checkpoints and malformed distributed messages."""

import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

PROJECT_PARENT = Path(__file__).resolve().parents[2]
if str(PROJECT_PARENT) not in sys.path:
    sys.path.insert(0, str(PROJECT_PARENT))

from MyFlows import Dense, Graph, load_checkpoint, save_checkpoint
from MyFlows.core.graph_opt import fuse_linear_ops
from MyFlows.core.node import Variable
from MyFlows.data.pipeline import MultiprocessDataLoader
from MyFlows.distributed.launcher import normalize_config
from MyFlows.distributed.protocol import decode_tensors, encode_tensors, schema_from_arrays, tensor_mapping
from MyFlows.layers.layer import Dropout
from MyFlows.layers.resnet import BatchNorm2d
from MyFlows.ops.basic import Add, Linear, MatMul
from MyFlows.ops.dropout import Dropout_Op
from MyFlows.ops.fused import softmax_cross_entropy, softmax_cross_entropy_grad
from MyFlows.ops.loss import CrossEntropy, LogLoss, MSELoss, PerceptionLoss
from MyFlows.train.opt import Adam


def numerical_grad(param, f, eps=1e-6):
    grad = np.zeros_like(param, dtype=np.float64)
    it = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        old = float(param[idx])
        param[idx] = old + eps
        plus = float(f())
        param[idx] = old - eps
        minus = float(f())
        param[idx] = old
        grad[idx] = (plus - minus) / (2 * eps)
        it.iternext()
    return grad


def check_op(testcase, op, weights, *, name, rtol=1e-5, atol=1e-7):
    """Compare ``op.backward`` with central differences of ``sum(op.value * weights)`` for every parent."""
    parents = op.parents

    def forward():
        op.forward(*(p.value for p in parents))
        return float(np.sum(np.asarray(op.value, dtype=np.float64) * weights))

    forward()
    for parent in parents:
        parent.clear_grad()
    op.grad = np.broadcast_to(np.asarray(weights, dtype=np.float64), np.shape(op.value)).copy()
    op.backward()
    for index, parent in enumerate(parents):
        if not parent.requires_grad:
            continue
        expected = numerical_grad(parent.value, forward)
        testcase.assertEqual(parent.grad.shape, parent.value.shape, f"{name}: parent {index} gradient shape")
        np.testing.assert_allclose(parent.grad, expected, rtol=rtol, atol=atol, err_msg=f"{name}: parent {index}")


class DotGradientTest(unittest.TestCase):
    """``MatMul`` and ``Linear`` follow ``numpy.dot``: one contracted axis, any operand rank."""

    SHAPES = [((3, 4), (4, 5)), ((4,), (4,)), ((3, 4), (4,)), ((4,), (4, 5)), ((2, 3, 4), (4, 5)),
              ((3, 4), (2, 4, 5)), ((2, 3, 4), (4,)), ((2, 3, 4), (5, 4, 6)), ((), (3, 4)), ((3, 4), ())]

    def setUp(self):
        self.rng = np.random.default_rng(0)

    def test_matmul_gradients_for_every_operand_rank(self):
        for a_shape, b_shape in self.SHAPES:
            a = Variable(self.rng.normal(size=a_shape), trainable=True)
            b = Variable(self.rng.normal(size=b_shape), trainable=True)
            op = MatMul(a, b)
            op.forward(a.value, b.value)
            self.assertEqual(np.shape(op.value), np.dot(a.value, b.value).shape)
            check_op(self, op, self.rng.normal(size=np.shape(op.value)), name=f"MatMul {a_shape}·{b_shape}")

    def test_linear_gradients_for_every_operand_rank_and_bias_shape(self):
        for x_shape, w_shape in self.SHAPES:
            out_shape = np.dot(np.zeros(x_shape), np.zeros(w_shape)).shape
            bias_shapes = {(), out_shape, out_shape[-1:], (1,) * len(out_shape)}
            for b_shape in bias_shapes:
                x = Variable(self.rng.normal(size=x_shape), trainable=True)
                w = Variable(self.rng.normal(size=w_shape), trainable=True)
                b = Variable(self.rng.normal(size=b_shape), trainable=True)
                op = Linear(x, w, b)
                op.forward(x.value, w.value, b.value)
                np.testing.assert_allclose(op.value, np.dot(x.value, w.value) + b.value)
                check_op(self, op, self.rng.normal(size=np.shape(op.value)),
                         name=f"Linear {x_shape}·{w_shape}+{b_shape}")

    def test_bias_larger_than_the_product_broadcasts_the_product(self):
        # x @ W is (3,), the bias (2, 3): the output takes the bias shape and dX/dW sum over the extra axis.
        x = Variable(self.rng.normal(size=(4,)), trainable=True)
        w = Variable(self.rng.normal(size=(4, 3)), trainable=True)
        b = Variable(self.rng.normal(size=(2, 3)), trainable=True)
        op = Linear(x, w, b)
        op.forward(x.value, w.value, b.value)
        self.assertEqual(op.value.shape, (2, 3))
        check_op(self, op, self.rng.normal(size=(2, 3)), name="Linear with a wider bias")

    def test_fused_add_keeps_the_channel_bias_rule_and_the_operand_order(self):
        # Add treats a 1-D right operand of a 4-D left operand as a per-channel (NCHW) bias, not as a
        # trailing-axis broadcast; fusing Add(MatMul, b) into Linear must not change that.
        x_value = self.rng.normal(size=(2, 3, 5, 4))
        w_value = self.rng.normal(size=(4, 5))
        bias_value = self.rng.normal(size=(3,))

        def build():
            x = Variable(x_value.copy(), trainable=True)
            w = Variable(w_value.copy(), trainable=True)
            b = Variable(bias_value.copy(), trainable=True)
            return x, w, b, Add(MatMul(x, w), b)

        x, w, b, plain = build()
        Graph(plain, optimize=False).forward()
        expected = np.dot(x_value, w_value) + bias_value.reshape(1, 3, 1, 1)
        np.testing.assert_allclose(plain.value, expected)
        x, w, b, root = build()
        fused, count = fuse_linear_ops(root)
        self.assertEqual(count, 1)
        self.assertIsInstance(fused, Linear)
        self.assertIs(fused.bias_first, False)
        fused.forward(x.value, w.value, b.value)
        np.testing.assert_allclose(fused.value, expected)
        check_op(self, fused, self.rng.normal(size=expected.shape), name="fused channel bias")

        # Add(b, MatMul): a 4-D bias with a 1-D product broadcasts the product per channel.
        v = Variable(self.rng.normal(size=(4,)), trainable=True)
        m = Variable(self.rng.normal(size=(4, 3)), trainable=True)
        wide = Variable(self.rng.normal(size=(2, 3, 5, 5)), trainable=True)
        root = Add(wide, MatMul(v, m))
        Graph(root, optimize=False).forward()
        expected = wide.value + np.dot(v.value, m.value).reshape(1, 3, 1, 1)
        np.testing.assert_allclose(root.value, expected)
        fused, count = fuse_linear_ops(root)
        self.assertEqual((count, fused.bias_first), (1, True))
        fused.forward(v.value, m.value, wide.value)
        np.testing.assert_allclose(fused.value, expected)
        check_op(self, fused, self.rng.normal(size=expected.shape), name="fused bias-first")

    def test_dense_layer_results_are_unchanged(self):
        # The common 2-D case goes through the same GEMMs as before: bit-identical to the plain formulas.
        x = Variable(self.rng.normal(size=(6, 4)).astype(np.float32))
        layer = Dense(4, 3, name="dense")
        out = layer(x)
        graph = Graph(out)
        graph.forward()
        w, b = layer.W.value, layer.b.value
        np.testing.assert_array_equal(out.value, np.dot(x.value, w) + b)
        upstream = self.rng.normal(size=out.value.shape).astype(np.float32)
        out.grad = upstream
        out.backward()
        np.testing.assert_array_equal(layer.W.grad, np.dot(x.value.T, upstream))
        self.assertEqual(layer.b.grad.shape, b.shape)
        np.testing.assert_allclose(layer.b.grad.reshape(-1), upstream.sum(axis=0, dtype=np.float64), rtol=1e-6)


class LossGradientTest(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

    def scalar_loss_check(self, cls, pred_shape, true_shape, *, true=None, name):
        pred = Variable(self.rng.normal(size=pred_shape), trainable=True)
        target = Variable(self.rng.normal(size=true_shape) if true is None else true)
        op = cls(pred, target)

        def forward():
            op.forward(pred.value, target.value)
            return float(op.value)

        forward()
        pred.clear_grad()
        op.grad = np.asarray(1.0)
        op.backward()
        self.assertEqual(pred.grad.shape, pred.value.shape, name)
        np.testing.assert_allclose(pred.grad, numerical_grad(pred.value, forward), rtol=1e-5, atol=1e-8, err_msg=name)

    def test_losses_reduce_the_gradient_when_prediction_and_target_broadcast(self):
        # (N, 1) predictions against (N,) targets broadcast to (N, N) in the forward pass; the gradient has to
        # come back in the prediction's shape, normalised by the broadcast element count.
        for pred_shape, true_shape in (((5, 1), (5,)), ((5,), (5,)), ((4, 3), (4, 3)), ((1, 3), (4, 3)), ((), (6,))):
            self.scalar_loss_check(MSELoss, pred_shape, true_shape, name=f"MSELoss {pred_shape} vs {true_shape}")
            signs = np.sign(self.rng.normal(size=true_shape))
            self.scalar_loss_check(LogLoss, pred_shape, true_shape, true=signs,
                                   name=f"LogLoss {pred_shape} vs {true_shape}")
        pred = Variable(np.array([[0.5], [-0.25], [2.0]]), trainable=True)
        target = Variable(np.array([1.0, -1.0, 1.0, -1.0]))
        op = PerceptionLoss(pred, target)
        op.forward(pred.value, target.value)
        op.grad = np.asarray(1.0)
        op.backward()
        self.assertEqual(pred.grad.shape, (3, 1))
        wrong = (target.value * pred.value) < 0
        np.testing.assert_allclose(pred.grad, np.sum(np.where(wrong, -target.value, 0), axis=1, keepdims=True) / 12)

    def test_logloss_gradient_is_finite_for_extreme_margins(self):
        pred = Variable(np.array([[-2000.0], [-40.0], [0.0], [40.0], [2000.0]]), trainable=True)
        target = Variable(np.ones((5, 1)))
        op = LogLoss(pred, target)
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # the old exp(clip(x)) form overflowed to inf here
            op.forward(pred.value, target.value)
            op.grad = np.asarray(1.0)
            op.backward()
        self.assertTrue(np.isfinite(pred.grad).all())
        np.testing.assert_allclose(pred.grad.reshape(-1) * 5, [-1.0, -1.0, -0.5, -4.248354255291589e-18, 0.0],
                                   rtol=1e-12, atol=0)

    def test_cross_entropy_handles_unnormalised_class_weights(self):
        logits = Variable(self.rng.normal(size=(4, 3)), trainable=True)
        for targets in (np.eye(3)[[0, 2, 1, 1]], self.rng.uniform(0.1, 2.0, size=(4, 3)), np.zeros((4, 3))):
            target = Variable(targets)
            op = CrossEntropy(logits, target)

            def forward():
                op.forward(logits.value, target.value)
                return float(op.value)

            forward()
            logits.clear_grad()
            op.grad = np.asarray(1.0)
            op.backward()
            np.testing.assert_allclose(logits.grad, numerical_grad(logits.value, forward), rtol=1e-5, atol=1e-8)
        probs = softmax_cross_entropy(logits.value, np.eye(3)[[0, 2, 1, 1]])[1]
        one_hot = np.eye(3)[[0, 2, 1, 1]]
        np.testing.assert_array_equal(softmax_cross_entropy_grad(probs, None, one_hot), (probs - one_hot) / 4)

    def test_cross_entropy_refuses_malformed_inputs(self):
        logits = self.rng.normal(size=(4, 3))
        loss, _probs, labels, _targets = softmax_cross_entropy(logits, np.array([0.0, 2.0, 1.0, 1.0]))
        np.testing.assert_array_equal(labels, [0, 2, 1, 1])
        self.assertEqual(float(loss), float(softmax_cross_entropy(logits, np.array([0, 2, 1, 1]))[0]))
        for bad in (np.array([0.5, 1, 2, 0]), np.array([0, 1, 3.0, 0]), np.array([-1.0, 0, 1, 2]),
                    np.array([np.nan, 0, 1, 2]), np.array([np.inf, 0, 1, 2]), np.array(["a", "b", "c", "d"])):
            with self.assertRaises(ValueError, msg=bad):
                softmax_cross_entropy(logits, bad)
        for shape in ((4,), (4, 3, 1), (0, 3), (4, 0)):
            with self.assertRaises(ValueError, msg=shape):
                softmax_cross_entropy(np.zeros(shape), np.zeros(shape[0], dtype=np.int64))

    def test_dropout_probability_must_be_a_number_in_range(self):
        x = Variable(np.ones((2, 2)))
        for p in (float("nan"), -0.1, 1.0, 1.5):
            with self.assertRaises(ValueError, msg=p):
                Dropout(p=p)
            with self.assertRaises(ValueError, msg=p):
                Dropout_Op(x, p=p)
        Dropout(p=0.0)
        Dropout_Op(x, p=0.999)


class CheckpointValidationTest(unittest.TestCase):
    """A checkpoint that fails any check leaves the live model and optimizer exactly as they were."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base = Path(self._tmp.name) / "checkpoint"
        np.random.seed(0)
        self.layer = Dense(3, 2, name="dense")
        self.bn = BatchNorm2d(2, name="bn")
        x = Variable(np.random.randn(4, 3).astype(self.layer.W.value.dtype))
        self.graph = Graph(self.layer(x))
        self.optimizer = Adam(self.graph, learning_rate=0.01)
        self.optimizer.v ={self.layer.W: np.full_like(self.layer.W.value, 0.25),
                            self.layer.b: np.full_like(self.layer.b.value, 0.5)}
        self.optimizer.s = {self.layer.W: np.full_like(self.layer.W.value, 0.125),
                            self.layer.b: np.full_like(self.layer.b.value, 0.75)}
        self.optimizer.t = 7
        save_checkpoint([self.layer, self.bn], epoch=3, loss=0.5, filepath=self.base, optimizer=self.optimizer)
        self.json_path, self.npz_path = self.base.with_suffix(".json"), self.base.with_suffix(".npz")
        self.metadata = json.loads(self.json_path.read_text(encoding="utf-8"))
        with np.load(self.npz_path) as archive:
            self.arrays = {key: archive[key] for key in archive.files}
        # Move the live state away from the saved one, then remember it.
        self.layer.W.value = self.layer.W.value + 1
        self.layer.b.value = self.layer.b.value - 1
        self.bn.running_mean[...] = 9.0
        self.optimizer.t = 99
        self.optimizer.learning_rate = 0.5
        self.optimizer.v = {self.layer.W: np.full_like(self.layer.W.value, 7.0)}

    def tearDown(self):
        self._tmp.cleanup()

    def snapshot(self):
        return (self.layer.W.value.copy(), self.layer.b.value.copy(), self.bn.running_mean.copy(),
                self.optimizer.t, self.optimizer.learning_rate,
                {id(k): v.copy() for k, v in self.optimizer.v.items()})

    def assert_unchanged(self, before, label):
        after = self.snapshot()
        for old, new in zip(before[:3], after[:3]):
            np.testing.assert_array_equal(old, new, err_msg=label)
        self.assertEqual(before[3:5], after[3:5], label)
        self.assertEqual(before[5].keys(), after[5].keys(), label)

    def write(self, metadata=None, arrays=None):
        self.json_path.write_text(json.dumps(self.metadata if metadata is None else metadata), encoding="utf-8")
        np.savez(self.npz_path, **(self.arrays if arrays is None else arrays))

    def load(self):
        return load_checkpoint([self.layer, self.bn], optimizer=self.optimizer, filepath=self.base)

    def test_a_valid_checkpoint_restores_model_and_optimizer(self):
        epoch, score = self.load()
        self.assertEqual((epoch, score), (3, 0.5))
        param_keys = [p["key"] for p in self.metadata["model"]["params"]]
        np.testing.assert_array_equal(self.layer.W.value, self.arrays[param_keys[0]])
        np.testing.assert_array_equal(self.bn.running_mean, np.zeros(2))
        self.assertEqual((self.optimizer.t, self.optimizer.learning_rate), (7, 0.01))
        np.testing.assert_array_equal(self.optimizer.v[self.layer.W], np.full_like(self.layer.W.value, 0.25))
        np.testing.assert_array_equal(self.optimizer.s[self.layer.b], np.full_like(self.layer.b.value, 0.75))

    def test_damaged_arrays_change_nothing(self):
        params = [p["key"] for p in self.metadata["model"]["params"]]
        buffers = [b["key"] for b in self.metadata["model"]["buffers"]]
        states = [e["array_key"] for entries in self.metadata["optimizer"]["states"].values() for e in entries]
        last = states[-1]  # the last array checked: everything before it has already passed
        cases = {}
        for key in (params[0], params[-1], buffers[0], last):
            poisoned = self.arrays[key].astype(np.float64).copy()
            poisoned.flat[0] = np.nan
            cases[f"nan in {key}"] = {**self.arrays, key: poisoned}
            cases[f"inf in {key}"] = {**self.arrays, key: np.where(np.isnan(poisoned), np.inf, poisoned)}
            cases[f"shape of {key}"] = {**self.arrays, key: np.zeros(self.arrays[key].shape + (2,))}
            cases[f"missing {key}"] = {k: v for k, v in self.arrays.items() if k != key}
            cases[f"text in {key}"] = {**self.arrays, key: np.full(self.arrays[key].shape, "x")}
            cases[f"overflow in {key}"] = {**self.arrays, key: np.full(self.arrays[key].shape, 1e300)}
        before = self.snapshot()
        for label, arrays in cases.items():
            self.write(arrays=arrays)
            with self.assertRaises(ValueError, msg=label):
                self.load()
            self.assert_unchanged(before, label)
        self.write()
        self.assertEqual(self.load()[0], 3, "the undamaged files still load")

    def test_inconsistent_metadata_changes_nothing(self):
        def edit(fn):
            metadata = json.loads(json.dumps(self.metadata))
            fn(metadata)
            return metadata

        params = self.metadata["model"]["params"]
        states = self.metadata["optimizer"]["states"]
        first_state = next(iter(states))
        cases = {
            "optimizer class": edit(lambda m: m["optimizer"].update({"class": "MBGD"})),
            "negative step count": edit(lambda m: m["optimizer"]["scalars"].update(t=-1)),
            "fractional step count": edit(lambda m: m["optimizer"]["scalars"].update(t=1.5)),
            "text learning rate": edit(lambda m: m["optimizer"]["scalars"].update(learning_rate="fast")),
            "infinite learning rate": edit(lambda m: m["optimizer"]["scalars"].update(learning_rate=float("inf"))),
            "beta of one": edit(lambda m: m["optimizer"]["scalars"].update(beta_1=1.0)),
            "unknown scalar": edit(lambda m: m["optimizer"]["scalars"].update(graph=1)),
            "unknown state": edit(lambda m: m["optimizer"]["states"].update(graph=[])),
            "state of an unknown parameter": edit(
                lambda m: m["optimizer"]["states"][first_state][0].update(param_key="nope")),
            "state listed twice": edit(
                lambda m: m["optimizer"]["states"][first_state].append(m["optimizer"]["states"][first_state][0])),
            "moments of different parameters": edit(lambda m: m["optimizer"]["states"]["s"].pop()),
            "two parameters share one array": edit(
                lambda m: m["model"]["params"][1].update(key=m["model"]["params"][0]["key"])),
            "non-finite score": edit(lambda m: m.update(score_value=float("nan"), best_loss=float("nan"))),
        }
        self.assertGreaterEqual(len(params), 2)
        before = self.snapshot()
        for label, metadata in cases.items():
            self.write(metadata=metadata)
            with self.assertRaises((ValueError, KeyError), msg=label):
                self.load()
            self.assert_unchanged(before, label)


class DistributedInputTest(unittest.TestCase):
    def test_tensor_messages_with_duplicate_or_mismatched_entries_are_refused(self):
        arrays = {"w": np.arange(6, dtype=np.float32).reshape(2, 3), "b": np.ones(3, dtype=np.float32)}
        schema = schema_from_arrays(arrays)
        encoded = encode_tensors(arrays)
        np.testing.assert_array_equal(decode_tensors(encoded, schema)["w"], arrays["w"])
        np.testing.assert_array_equal(decode_tensors(dict(arrays), schema)["w"], arrays["w"])
        w_item = next(item for item in encoded if item["name"] == "w")
        b_item = next(item for item in encoded if item["name"] == "b")
        for label, items in {
            "duplicate entry": [w_item, w_item, b_item],
            "entry without a name": [w_item, {"shape": [3], "dtype": "float32", "data": [1, 1, 1]}],
            "entry that is not an object": [w_item, "b"],
            "unhashable name": [w_item, {**b_item, "name": ["b"]}],
            "wrong shape": [{**w_item, "shape": [3, 2]}, b_item],
            "wrong length": [{**w_item, "data": w_item["data"][:-1]}, b_item],
        }.items():
            with self.assertRaises(ValueError, msg=label):
                decode_tensors(items, schema)
        with self.assertRaises(ValueError):
            decode_tensors(encoded, [*schema, schema[0]])
        # A mapping (the gRPC path) is checked against the schema's shape too, not reshaped to fit.
        with self.assertRaises(ValueError):
            decode_tensors({"w": arrays["w"].reshape(3, 2), "b": arrays["b"]}, schema)
        with self.assertRaises(ValueError):
            decode_tensors({"w": arrays["w"].reshape(-1), "b": arrays["b"]}, schema)
        with self.assertRaises(ValueError):
            tensor_mapping([w_item, w_item])
        with self.assertRaises(ValueError):
            tensor_mapping([{"shape": [1], "data": [1.0]}])
        self.assertEqual(tensor_mapping(None), {})
        self.assertEqual(set(tensor_mapping(encoded)), {"w", "b"})

    def test_protobuf_blobs_are_checked_before_they_become_arrays(self):
        try:
            from MyFlows.distributed.transport_grpc import _array_to_blob, blobs_to_encoded, common_pb2
        except ImportError as exc:  # gRPC stubs are optional
            self.skipTest(f"gRPC transport unavailable: {exc}")
        good = _array_to_blob("w", np.arange(6, dtype=np.float32).reshape(2, 3))
        for as_arrays in (False, True):
            decoded = blobs_to_encoded([good], as_arrays=as_arrays)
            value = decoded["w"] if as_arrays else np.asarray(decoded[0]["data"], dtype=np.float32).reshape(2, 3)
            np.testing.assert_array_equal(value, np.arange(6, dtype=np.float32).reshape(2, 3))
            blob = lambda **kw: common_pb2.TensorBlob(**{"name": "w", "shape": [2, 3], "dtype": "float32",
                                                         "data": good.data, **kw})
            for label, blobs in {"duplicate name": [good, good], "other dtype": [blob(dtype="float64")],
                                 "negative dimension": [blob(shape=[-2, -3])], "short payload": [blob(data=good.data[:-4])],
                                 "shape larger than payload": [blob(shape=[2, 4])]}.items():
                with self.assertRaises(ValueError, msg=(label, as_arrays)):
                    blobs_to_encoded(blobs, as_arrays=as_arrays)
            empty = blobs_to_encoded([blob(shape=[0, 3], data=b"")], as_arrays=as_arrays)
            self.assertEqual(len(empty), 1)

    def test_run_configuration_needs_a_usable_learning_rate_and_optimizer(self):
        self.assertEqual(normalize_config(mode="single", learning_rate="0.05")["learning_rate"], 0.05)
        for value in (0, -0.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError, msg=value):
                normalize_config(mode="single", learning_rate=value)
        with self.assertRaises((ValueError, TypeError)):
            normalize_config(mode="single", learning_rate=None)
        with self.assertRaises(ValueError):
            normalize_config(mode="single", optimizer="lbfgs")


class DataLoaderStartFailureTest(unittest.TestCase):
    def test_workers_already_started_are_stopped_when_a_later_one_fails_to_start(self):
        started = []

        class FakeProcess:
            def __init__(self, *args, **kwargs):
                self.exitcode = None
                self.alive = False
                self.terminated = False

            def start(self):
                if len(started) == 2:
                    raise OSError("cannot start another process")
                self.alive = True
                started.append(self)

            def join(self, timeout=None):
                pass

            def is_alive(self):
                return self.alive

            def terminate(self):
                self.alive, self.terminated = False, True

        class FakeContext:
            def __init__(self, real):
                self.real = real
                self.Process = FakeProcess

            def __getattr__(self, name):
                return getattr(self.real, name)

        import multiprocessing as mp

        loader = MultiprocessDataLoader(list(range(8)), 2, num_workers=4, load_fn=lambda item: (item, item))
        with mock.patch("MyFlows.data.pipeline.mp.get_context", return_value=FakeContext(mp.get_context("spawn"))):
            with self.assertRaisesRegex(OSError, "cannot start"):
                next(iter(loader))
        self.assertEqual(len(started), 2)
        self.assertTrue(all(p.terminated for p in started), "the two started workers were not left running")


class OnnxExportTest(unittest.TestCase):
    def setUp(self):
        try:
            import onnx  # noqa: F401
        except ImportError:
            self.skipTest("onnx is not installed")
        self.rng = np.random.default_rng(2)

    def run_onnx(self, path, value):
        try:
            import onnxruntime as ort
        except ImportError:
            return None
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        return session.run(None, {session.get_inputs()[0].name: value})[0]

    def test_batchnorm_in_training_mode_is_not_exported(self):
        from MyFlows import export_onnx

        x = Variable(self.rng.normal(size=(2, 2, 4, 4)).astype(np.float32), name="input")
        bn = BatchNorm2d(2, name="bn")
        graph = Graph(bn(x))
        graph.forward()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "evaluation mode"):
                export_onnx(graph, Path(tmp) / "bn.onnx", input_nodes=[x], output_names=["output"])
            bn.eval()
            graph.forward()
            export_onnx(graph, Path(tmp) / "bn.onnx", input_nodes=[x], output_names=["output"])
            self.assertTrue((Path(tmp) / "bn.onnx").exists())

    def test_dot_products_of_higher_rank_export_with_numpy_semantics(self):
        from MyFlows import export_onnx

        cases = {"3-D input": ((2, 3, 4), (4, 5), (5,)), "3-D weight": ((3, 4), (2, 4, 5), (5,)),
                 "vector input": ((4,), (4, 5), (5,)), "matrix": ((3, 4), (4, 5), (1, 5))}
        with tempfile.TemporaryDirectory() as tmp:
            for label, (x_shape, w_shape, b_shape) in cases.items():
                x = Variable(self.rng.normal(size=x_shape).astype(np.float32), name="input")
                w = Variable(self.rng.normal(size=w_shape).astype(np.float32), trainable=True, name="w")
                b = Variable(self.rng.normal(size=b_shape).astype(np.float32), trainable=True, name="b")
                for kind, node in (("linear", Linear(x, w, b)), ("matmul", MatMul(x, w))):
                    graph = Graph(node, optimize=False)
                    graph.forward()
                    path = Path(tmp) / f"{kind}-{len(x_shape)}{len(w_shape)}.onnx"
                    export_onnx(graph, path, input_nodes=[x], output_names=["output"])
                    got = self.run_onnx(path, x.value)
                    if got is not None:
                        np.testing.assert_allclose(got, node.value, rtol=1e-5, atol=1e-6, err_msg=f"{label} {kind}")


if __name__ == "__main__":
    unittest.main()
