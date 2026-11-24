# -*- coding: utf-8 -*-
"""Route C Batch 2 tests: multi-GPU cuda:rank isolation and true cross-rank SyncBatchNorm2d."""

from __future__ import annotations

import importlib.util
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import MyFlows as ms
from MyFlows.core.device import (
    _lazy_cupy,
    asnumpy,
    cuda_available,
    cuda_device_count,
    get_device,
    get_device_index,
    get_device_string,
    resolve_rank_device,
    set_device,
    xp,
)
from MyFlows.core.dtype import get_default_dtype, set_default_dtype
from MyFlows.core.graph import Graph
from MyFlows.core.node import Variable
from MyFlows.distributed.sync import SyncRun, load_state, make_sync_bn_allreduce, save_state
from MyFlows.layers.layer import Conv2D, Dense
from MyFlows.layers.mobilenet import MobileNetV2
from MyFlows.layers.resnet import BatchNorm2d, GlobalAvgPool2d, ResNet18, SyncBatchNorm2d, convert_sync_batchnorm
from MyFlows.ops.activation import ReLU
from MyFlows.ops.loss import MSELoss
from MyFlows.train.opt import Adam
from MyFlows.utils.initializers import make_initializer


def _sync_bn_ps_worker(rank: int, sync, init_arrays: dict, x_shard: np.ndarray, y_shard: np.ndarray, out_dir: str):
    """Spawned worker for multi-process SyncBatchNorm2d test over Socket JSON PS."""
    set_device("cpu")
    set_default_dtype(np.float32)

    allreduce_fn = make_sync_bn_allreduce(sync)
    conv = Conv2D(3, 4, kernel_size=3, padding=1, bias=False, name="conv")
    bn = SyncBatchNorm2d(4, momentum=0.1, eps=1e-5, name="bn", allreduce_fn=allreduce_fn)
    pool = GlobalAvgPool2d(name="gap")
    fc = Dense(4, 2, activation=None, name="fc")

    # Align names cleanly with init_arrays
    params_map = {
        "conv.kernel": conv.kernel,
        "bn.gamma": bn.gamma,
        "bn.beta": bn.beta,
        "fc.W": fc.W,
        "fc.b": fc.b,
    }
    init = sync.initialize({
        **{k: v.value for k, v in params_map.items()},
        "bn.running_mean": bn.running_mean,
        "bn.running_var": bn.running_var,
    })
    for k, node in params_map.items():
        node.value = np.array(init[k], dtype=np.float32)
    bn.running_mean[...] = init["bn.running_mean"]
    bn.running_var[...] = init["bn.running_var"]

    bn.train()
    x_var = Variable(x_shard.astype(np.float32), name="X", requires_grad=False)
    y_var = Variable(y_shard.astype(np.float32), name="y", requires_grad=False)
    loss = MSELoss(fc(pool(ReLU(bn(conv(x_var))))), y_var)
    graph = Graph(loss, optimize=True)
    opt = Adam(graph, learning_rate=0.02)

    graph.forward()
    graph.backward()
    local_grads = {k: node.grad for k, node in params_map.items()}
    local_payload = {
        **local_grads,
        "bn.running_mean": bn.running_mean,
        "bn.running_var": bn.running_var,
    }
    means, step_loss = sync.allreduce(local_payload, x_shard.shape[0], float(loss.value))
    opt.update({node: means[k] for k, node in params_map.items()})
    bn.running_mean[...] = means["bn.running_mean"]
    bn.running_var[...] = means["bn.running_var"]

    digest = ms.distributed.protocol.digest_state(
        {**{k: n.value for k, n in params_map.items()}, "bn.running_mean": bn.running_mean, "bn.running_var": bn.running_var},
        {"step": 1},
        step=1,
    )
    sync.commit(digest)
    np.savez(
        Path(out_dir) / f"rank_{rank}.npz",
        loss=np.array([step_loss], dtype=np.float32),
        rm=bn.running_mean,
        rv=bn.running_var,
        **{f"p_{k}": n.value for k, n in params_map.items()},
        **{f"g_{k}": means[k] for k in params_map},
    )


class _ThreadBarrierAllreduce:
    """In-process rank collective simulating sync.reduce (exact sum across all ranks)."""

    def __init__(self, world_size: int):
        self.world_size = int(world_size)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._slots: dict[str, dict[int, dict[str, np.ndarray]]] = {}
        self._reduced: dict[str, dict[str, np.ndarray]] = {}

    def for_rank(self, rank: int):
        seq = 0

        def _fn(tag: str, values: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            nonlocal seq
            seq += 1
            key = f"{seq}:{tag}"
            with self._cv:
                bucket = self._slots.setdefault(key, {})
                bucket[rank] = {k: np.array(v, dtype=np.float32, copy=True) for k, v in values.items()}
                if len(bucket) == self.world_size:
                    summed = {}
                    for name in values:
                        acc = np.zeros_like(bucket[0][name], dtype=np.float32)
                        for r in range(self.world_size):
                            acc = acc + bucket[r][name]
                        summed[name] = acc
                    self._reduced[key] = summed
                    self._cv.notify_all()
                else:
                    while key not in self._reduced:
                        if not self._cv.wait(timeout=15.0):
                            raise TimeoutError(f"thread collective timed out on {key}")
                return {k: v.copy() for k, v in self._reduced[key].items()}

        return _fn


class DeviceRankIsolationTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(set_device, get_device())

    def test_resolve_rank_device_and_validation(self):
        self.assertEqual(resolve_rank_device("cpu", 3), "cpu")
        self.assertEqual(resolve_rank_device("cuda:2", 0), "cuda:2")
        self.assertEqual(resolve_rank_device("gpu:1", 2, gpu_ids=[4, 5]), "cuda:4")

        with patch("MyFlows.core.device.cuda_device_count", return_value=4):
            self.assertEqual([resolve_rank_device("cuda", r) for r in range(6)],
                             ["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:0", "cuda:1"])
            self.assertEqual(resolve_rank_device("auto", 3), "cuda:3")
            self.assertEqual(resolve_rank_device("cuda", 2, gpu_ids=[1, 3]), "cuda:1")

        with patch("MyFlows.core.device.cuda_device_count", return_value=1):
            self.assertEqual([resolve_rank_device("cuda", r) for r in range(3)], ["cuda:0", "cuda:0", "cuda:0"])

        with patch("MyFlows.core.device.cuda_device_count", return_value=0):
            self.assertEqual(resolve_rank_device("auto", 1), "cpu")
            self.assertEqual(resolve_rank_device("cuda", 1), "cuda:0")

        for bad in ("cuda:-1", "cuda:abc", "tpu"):
            with self.assertRaises(ValueError):
                set_device(bad)

    def test_set_device_selects_explicit_cuda_index_with_mocked_multi_gpu(self):
        used = []

        class _FakeDev:
            def __init__(self, idx):
                self.idx = idx

            def use(self):
                used.append(self.idx)

        fake_cp = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                Device=_FakeDev,
                runtime=SimpleNamespace(getDeviceCount=lambda: 2),
            )
        )
        with patch("MyFlows.core.device._lazy_cupy", return_value=fake_cp):
            self.assertEqual(set_device("cuda:1"), "cuda")
            self.assertEqual(get_device(), "cuda")
            self.assertEqual(get_device_index(), 1)
            self.assertEqual(get_device_string(), "cuda:1")
            with self.assertRaises(ValueError):
                set_device("cuda:2")
        self.assertEqual(used, [1])
        set_device("cpu")
        self.assertIsNone(get_device_index())
        self.assertEqual(get_device_string(), "cpu")

    @unittest.skipUnless(cuda_available(), "CUDA required for real device index check")
    def test_real_cuda_zero_selection(self):
        self.assertGreaterEqual(cuda_device_count(), 1)
        set_device("cuda:0")
        self.assertEqual(get_device(), "cuda")
        self.assertEqual(get_device_index(), 0)
        self.assertEqual(get_device_string(), "cuda:0")


class SyncBatchNorm2dTest(unittest.TestCase):
    def setUp(self):
        self.addCleanup(set_device, get_device())
        self.addCleanup(set_default_dtype, get_default_dtype())
        set_device("cpu")
        set_default_dtype(np.float32)

    def _run_reference_step(self, x_np, y_np, init_w, *, unbiased_running_var: bool):
        conv = Conv2D(3, 4, kernel_size=3, padding=1, name="conv")
        bn = BatchNorm2d(4, momentum=0.1, eps=1e-5, name="bn", unbiased_running_var=unbiased_running_var)
        pool = GlobalAvgPool2d(name="gap")
        fc = Dense(4, 2, activation=None, name="fc")
        for key, node in (("ck", conv.kernel), ("cb", conv.b), ("bg", bn.gamma), ("bb", bn.beta), ("fw", fc.W), ("fb", fc.b)):
            node.value[...] = xp.asarray(init_w[key], dtype=np.float32)
        bn.running_mean[...] = xp.asarray(init_w["rm"], dtype=np.float32)
        bn.running_var[...] = xp.asarray(init_w["rv"], dtype=np.float32)
        bn.train()
        x_var = Variable(xp.asarray(x_np), requires_grad=False)
        y_var = Variable(xp.asarray(y_np), requires_grad=False)
        out = fc(pool(ReLU(bn(conv(x_var)))))
        loss = MSELoss(out, y_var)
        graph = Graph(loss, optimize=True)
        graph.forward()
        graph.backward()
        return {
            "loss": float(asnumpy(loss.value)),
            "out": asnumpy(out.value).copy(),
            "rm": asnumpy(bn.running_mean).copy(),
            "rv": asnumpy(bn.running_var).copy(),
            "grads": {k: asnumpy(n.grad).copy() for k, n in (("ck", conv.kernel), ("cb", conv.b), ("bg", bn.gamma), ("bb", bn.beta), ("fw", fc.W), ("fb", fc.b))},
        }

    def test_uneven_shards_and_high_offset_match_global_batch_cpu_and_cuda(self):
        devices = ("cpu", "cuda") if cuda_available() else ("cpu",)
        rng = np.random.default_rng(123)
        for device in devices:
            for shards, offset, unbiased in (((3, 5), 0.0, False), ((2, 3, 3), 10000.0, True)):
                with self.subTest(device=device, shards=shards, offset=offset, unbiased=unbiased):
                    set_device(device)
                    x_np = (offset + 0.01 * rng.normal(size=(sum(shards), 3, 8, 8))).astype(np.float32)
                    y_np = rng.normal(size=(sum(shards), 2)).astype(np.float32)
                    init_w = {
                        "ck": (0.1 * rng.normal(size=(4, 3, 3, 3))).astype(np.float32),
                        "cb": rng.normal(scale=0.1, size=(4,)).astype(np.float32),
                        "bg": rng.uniform(0.7, 1.3, size=(4,)).astype(np.float32),
                        "bb": np.full((4,), 0.35, dtype=np.float32),  # away from ReLU 0 kink
                        "fw": (0.1 * rng.normal(size=(4, 2))).astype(np.float32),
                        "fb": rng.normal(scale=0.1, size=(2,)).astype(np.float32),
                        "rm": (offset + rng.normal(scale=0.05, size=(4,))).astype(np.float32),
                        "rv": rng.uniform(0.5, 1.5, size=(4,)).astype(np.float32),
                    }
                    ref = self._run_reference_step(x_np, y_np, init_w, unbiased_running_var=unbiased)

                    world = len(shards)
                    collective = _ThreadBarrierAllreduce(world)
                    slices = []
                    pos = 0
                    for sz in shards:
                        slices.append((x_np[pos : pos + sz], y_np[pos : pos + sz]))
                        pos += sz

                    def _rank_job(rank: int):
                        set_device(device)
                        set_default_dtype(np.float32)
                        conv = Conv2D(3, 4, kernel_size=3, padding=1, name="conv")
                        bn = SyncBatchNorm2d(
                            4, momentum=0.1, eps=1e-5, name="bn",
                            unbiased_running_var=unbiased,
                            allreduce_fn=collective.for_rank(rank),
                        )
                        pool = GlobalAvgPool2d(name="gap")
                        fc = Dense(4, 2, activation=None, name="fc")
                        for key, node in (("ck", conv.kernel), ("cb", conv.b), ("bg", bn.gamma), ("bb", bn.beta), ("fw", fc.W), ("fb", fc.b)):
                            node.value[...] = xp.asarray(init_w[key], dtype=np.float32)
                        bn.running_mean[...] = xp.asarray(init_w["rm"], dtype=np.float32)
                        bn.running_var[...] = xp.asarray(init_w["rv"], dtype=np.float32)
                        bn.train()
                        rx, ry = slices[rank]
                        x_v = Variable(xp.asarray(rx), requires_grad=False)
                        y_v = Variable(xp.asarray(ry), requires_grad=False)
                        out = fc(pool(ReLU(bn(conv(x_v)))))
                        loss = MSELoss(out, y_v)
                        # SyncBatchNorm2d_Op must remain unfused during training under optimize=True!
                        g = Graph(loss, optimize=True)
                        g.forward()
                        g.backward()
                        return {
                            "n": len(rx),
                            "loss": float(asnumpy(loss.value)),
                            "out": asnumpy(out.value).copy(),
                            "rm": asnumpy(bn.running_mean).copy(),
                            "rv": asnumpy(bn.running_var).copy(),
                            "grads": {k: asnumpy(n.grad).copy() for k, n in (("ck", conv.kernel), ("cb", conv.b), ("bg", bn.gamma), ("bb", bn.beta), ("fw", fc.W), ("fb", fc.b))},
                        }

                    with ThreadPoolExecutor(max_workers=world) as pool_exec:
                        rank_res = [f.result(timeout=30) for f in [pool_exec.submit(_rank_job, r) for r in range(world)]]

                    total_n = float(sum(shards))
                    avg_loss = sum(r["n"] * r["loss"] for r in rank_res) / total_n
                    cat_out = np.concatenate([r["out"] for r in rank_res], axis=0)
                    np.testing.assert_allclose(avg_loss, ref["loss"], rtol=1e-4, atol=1e-5)
                    np.testing.assert_allclose(cat_out, ref["out"], rtol=1e-4, atol=1e-4)
                    for r in rank_res:
                        np.testing.assert_allclose(r["rm"], ref["rm"], rtol=1e-5, atol=1e-5)
                        np.testing.assert_allclose(r["rv"], ref["rv"], rtol=1e-4, atol=1e-4)
                    for k in ref["grads"]:
                        avg_grad = sum((r["n"] / total_n) * r["grads"][k] for r in rank_res)
                        np.testing.assert_allclose(avg_grad, ref["grads"][k], rtol=1e-4, atol=1e-4, err_msg=f"grad {k}")

    def test_convert_sync_batchnorm_on_resnet_and_mobilenet_eval_onnx_fold(self):
        rng = np.random.default_rng(7)
        for builder in (
            lambda: ResNet18(in_channels=3, output_dim=2, stem="cifar", base_width=4,
                             initializer=make_initializer("kaiming_normal", seed=3)),
            lambda: MobileNetV2(output_dim=2, dropout=0.0,
                                initializer=make_initializer("kaiming_normal", seed=5)),
        ):
            model = builder()
            dummy_collective = lambda tag, vals: vals
            _, converted = convert_sync_batchnorm(model, dummy_collective)
            self.assertGreater(converted, 5)
            for layer in model.sub_layers:
                if isinstance(layer, BatchNorm2d):
                    self.assertIsInstance(layer, SyncBatchNorm2d)

        # Eval mode must use running stats without touching collective, fold Conv+BN cleanly, and export ONNX
        if not (importlib.util.find_spec("onnx") and importlib.util.find_spec("onnxruntime")):
            self.skipTest("onnx and onnxruntime required")
        import onnxruntime as ort

        model = ResNet18(in_channels=3, output_dim=2, stem="cifar", base_width=4,
                         initializer=make_initializer("kaiming_normal", seed=9))

        def _forbid(tag, vals):
            raise AssertionError(f"eval mode must not call collective: {tag}")

        convert_sync_batchnorm(model, _forbid)
        model.eval()
        x = Variable(rng.normal(size=(1, 3, 32, 32)).astype(np.float32), requires_grad=False)
        folded_root, fold_stats = ms.apply_graph_optimizations(model(x), mode="inference")
        self.assertGreater(fold_stats["bn_folds"], 0)
        g_folded = Graph(folded_root, optimize=False)
        g_folded.forward()
        g = Graph(model(x), optimize=False)
        g.forward()
        with tempfile.TemporaryDirectory() as tmp:
            onnx_path = Path(tmp) / "syncbn_eval.onnx"
            ms.export_onnx(g, onnx_path, input_nodes=[x], output_names=["out"], dynamic_batch=True)
            onnx_folded = Path(tmp) / "syncbn_folded.onnx"
            ms.export_onnx(g_folded, onnx_folded, input_nodes=[x], output_names=["out"], dynamic_batch=True)
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            sess_folded = ort.InferenceSession(str(onnx_folded), providers=["CPUExecutionProvider"])
            sample = rng.normal(size=(3, 3, 32, 32)).astype(np.float32)
            x.value = sample
            g.forward()
            g_folded.forward()
            (got,) = sess.run(None, {sess.get_inputs()[0].name: sample})
            (got_folded,) = sess_folded.run(None, {sess_folded.get_inputs()[0].name: sample})
            np.testing.assert_allclose(got, g.target_node.value, rtol=1e-4, atol=1e-4)
            np.testing.assert_allclose(got_folded, g_folded.target_node.value, rtol=1e-4, atol=1e-4)

    def test_multiprocess_syncrun_ps_sync_batchnorm_matches_single_process(self):
        rng = np.random.default_rng(31)
        x_np = rng.normal(size=(8, 3, 8, 8)).astype(np.float32)
        y_np = rng.normal(size=(8, 2)).astype(np.float32)

        conv = Conv2D(3, 4, kernel_size=3, padding=1, bias=False, name="conv")
        bn = BatchNorm2d(4, momentum=0.1, eps=1e-5, name="bn")
        pool = GlobalAvgPool2d(name="gap")
        fc = Dense(4, 2, activation=None, name="fc")
        bn.beta.value[...] = 0.35  # away from ReLU zero kink
        bn.train()
        params_map = {
            "conv.kernel": conv.kernel,
            "bn.gamma": bn.gamma,
            "bn.beta": bn.beta,
            "fc.W": fc.W,
            "fc.b": fc.b,
        }
        init_arrays = {
            **{k: v.value.copy() for k, v in params_map.items()},
            "bn.running_mean": bn.running_mean.copy(),
            "bn.running_var": bn.running_var.copy(),
        }

        x_ref = Variable(x_np, requires_grad=False)
        y_ref = Variable(y_np, requires_grad=False)
        loss_ref = MSELoss(fc(pool(ReLU(bn(conv(x_ref))))), y_ref)
        g_ref = Graph(loss_ref, optimize=True)
        opt_ref = Adam(g_ref, learning_rate=0.02)
        opt_ref.one_step()
        opt_ref.update()
        ref_loss = float(loss_ref.value)

        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            state_path = work / "init.npz"
            save_state(state_path, init_arrays)
            with SyncRun(
                mode="ps",
                transport="socket_json",
                world_size=2,
                shard_sizes=[3, 5],
                initial_state=state_path,
                timeout=30.0,
            ) as run:
                # Pass full x_np/y_np; wait SyncRun sends same args to all ranks, slice inside or pass shards:
                run.start(_sync_bn_ps_worker_sliced, args=(init_arrays, x_np, y_np, [3, 5], str(work)))
                res = run.wait()
            self.assertEqual(res["status"], "passed", res.get("error"))
            r0 = dict(np.load(work / "rank_0.npz"))
            r1 = dict(np.load(work / "rank_1.npz"))
            np.testing.assert_allclose(float(r0["loss"][0]), ref_loss, rtol=1e-4, atol=1e-5)
            np.testing.assert_allclose(r0["rm"], r1["rm"], rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(r0["rm"], bn.running_mean, rtol=1e-5, atol=1e-5)
            np.testing.assert_allclose(r0["rv"], bn.running_var, rtol=1e-4, atol=1e-4)
            for k, node in params_map.items():
                np.testing.assert_allclose(r0[f"p_{k}"], r1[f"p_{k}"], rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(r0[f"p_{k}"], node.value, rtol=1e-4, atol=1e-4)


def _sync_bn_ps_worker_sliced(rank: int, sync, init_arrays: dict, x_full: np.ndarray, y_full: np.ndarray,
                              shards: list[int], out_dir: str):
    start = sum(shards[:rank])
    stop = start + shards[rank]
    _sync_bn_ps_worker(rank, sync, init_arrays, x_full[start:stop], y_full[start:stop], out_dir)


if __name__ == "__main__":
    unittest.main()
