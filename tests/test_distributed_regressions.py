import unittest
import uuid

import numpy as np

from MyFlows.core.device import set_device
from MyFlows.distributed.constants import PROTOCOL_VERSION, STATUS_DUP, STATUS_OK, STATUS_STALE, STATUS_WAITING
from MyFlows.distributed.engine import PSEngine
from MyFlows.distributed.launcher import run_training
from MyFlows.distributed.protocol import encode_tensors, payload_hash
from MyFlows.distributed.schedule import BatchCursor, resolve_budget
from MyFlows.distributed.session import TrainSession
from MyFlows.distributed.tasks.donkey_cnn import DonkeyCnnTask
from MyFlows.distributed.tasks.mnist_mlp import MnistMlpTask
from MyFlows.distributed.transport_grpc import RingInbox


def _engine(n=1, shards=None, size=8):
    params = {"w": np.ones((size,), np.float32)}
    return PSEngine(
        run_id="run-cache",
        n_workers=n,
        init_parameters=params,
        optimizer_meta={"impl": "MBGD", "learning_rate": 0.01, "t": 0},
        shard_sizes=shards or [size] * n,
        retain_rounds=1,
        digest_rounds=4,
    ), params


def _msg(engine, worker, kind, **fields):
    message = {
        "protocol_version": PROTOCOL_VERSION,
        "run_id": engine.run_id,
        "worker_id": worker,
        "request_id": fields.pop("request_id", uuid.uuid4().hex),
        "type": kind,
        "schema_hash": engine.schema_hash,
    }
    message.update(fields)
    return message


def _tiny_mnist_split(n=8, dim=784):
    x = np.arange(n * dim, dtype=np.float32).reshape(n, dim) / float(n * dim)
    y = np.arange(n, dtype=np.int64).reshape(n, 1) % 10
    return {
        "train": (x, y),
        "val": (x[:2], y[:2]),
        "test": (x[:2], y[:2]),
    }


def _tiny_donkey_split(n=4):
    x = np.zeros((n, 3, 120, 160), dtype=np.float32)
    y = np.zeros((n, 2), dtype=np.float32)
    for i in range(n):
        x[i, :, :4, :4] = i + 1
        y[i] = (0.1 * i, 0.3)
    return {
        "train": (x, y),
        "val": (x[:1], y[:1]),
        "test": (x[:1], y[:1]),
    }


def _complete_round(engine, params, step, version, grads):
    for worker in range(engine.n_workers):
        engine.handle(_msg(
            engine, worker, "push", request_id=f"push-{worker}-{step}",
            global_step=step, parameter_version=version, n_samples=engine.shard_sizes[worker],
            loss=0.5, tensors=encode_tensors(grads), payload_hash=payload_hash(grads),
        ))
    pulls = []
    for worker in range(engine.n_workers):
        pulls.append(engine.handle(_msg(
            engine, worker, "pull", request_id=f"pull-{worker}-{step}",
            global_step=step, parameter_version=version,
        )))
    update_id = pulls[0]["update_id"]
    last = None
    for worker in range(engine.n_workers):
        last = engine.handle(_msg(
            engine, worker, "update_applied", request_id=f"upd-{worker}-{step}",
            global_step=step, parameter_version=version + 1,
            update_id=update_id, optimizer_step=step + 1, digest="same",
        ))
    return last, pulls[0]["request_id"]


class TaskDispatchTest(unittest.TestCase):
    def test_single_uses_mnist_adapter_not_synthetic(self):
        split = _tiny_mnist_split()
        result = run_training(
            mode="single", transport="none", task="mnist_mlp", split_arrays=split,
            hidden=8, global_batch=4, epochs=1, steps=1, seed=0, timeout=30,
            collect_history=True, drop_last=True,
        )
        self.assertEqual(result["status"], "passed", result.get("error"))
        weight = result["history"][0]["fc1.weight"]
        self.assertEqual(tuple(weight.shape), (784, 8))
        self.assertEqual(result["config"]["task"], "mnist_mlp")

    def test_unknown_task_fails_instead_of_passing(self):
        with self.assertRaises(ValueError):
            run_training(mode="single", transport="none", task="no_such_task", steps=1)

    def test_donkey_build_without_conv_backend_uses_numpy_on_cpu(self):
        set_device("cpu")
        built = DonkeyCnnTask().build_model({"seed": 0}, device="cpu")
        self.assertEqual(built["backend"], "numpy")
        self.assertNotIn("conv_backend", {"seed": 0})


class EpochAndBatchCursorTest(unittest.TestCase):
    def test_real_task_epochs_control_update_count(self):
        split = _tiny_mnist_split(n=8)
        result = run_training(
            mode="single", transport="none", task="mnist_mlp", split_arrays=split,
            hidden=8, global_batch=4, epochs=3, steps=100, seed=0, timeout=30,
            collect_history=False, drop_last=True, optimizer="mbgd",
        )
        self.assertEqual(result["status"], "passed", result.get("error"))
        self.assertEqual(len(result["losses"]), 6)
        self.assertEqual(result["config"]["steps_semantics"], "per_epoch_batch_cap")
        self.assertEqual([row["epoch"] for row in result["metrics"]], [0, 0, 1, 1, 2, 2])
        self.assertEqual([row["global_step"] for row in result["metrics"]], list(range(6)))
        self.assertEqual(result["epoch_index_builds"], 3)

    def test_synthetic_keeps_steps_semantics(self):
        budget = resolve_budget({"task": "synthetic", "steps": 20, "epochs": 10}, None)
        self.assertEqual(budget["epochs"], 1)
        self.assertEqual(budget["steps_per_epoch"], 20)
        self.assertEqual(budget["total_steps"], 20)
        self.assertEqual(budget["steps_semantics"], "synthetic_total_steps")
        result = run_training(mode="single", transport="none", task="synthetic", steps=3, epochs=9)
        self.assertEqual(result["status"], "passed", result.get("error"))
        self.assertEqual(len(result["losses"]), 3)

    def test_epoch_index_is_built_once_per_epoch(self):
        task = MnistMlpTask()
        split = _tiny_mnist_split(n=8)
        config = {"task": "mnist_mlp", "global_batch": 4, "seed": 0, "drop_last": True, "steps": 100, "epochs": 2}
        cursor = BatchCursor(task, split, config)
        seen = []
        for epoch in range(2):
            for step in range(cursor.n_batches(epoch)):
                x, y = cursor.get(epoch, step)
                seen.append((epoch, step, x.shape[0]))
        self.assertEqual(cursor.builds, 2)
        self.assertEqual(task.epoch_index_builds, 2)
        self.assertEqual(seen, [(0, 0, 4), (0, 1, 4), (1, 0, 4), (1, 1, 4)])


class MetricsWithoutHistoryTest(unittest.TestCase):
    def test_losses_and_metrics_are_kept_when_history_disabled(self):
        split = _tiny_mnist_split(n=8)
        result = run_training(
            mode="single", transport="none", task="mnist_mlp", split_arrays=split,
            hidden=8, global_batch=4, epochs=1, steps=2, seed=0, timeout=30,
            collect_history=False, drop_last=True,
        )
        self.assertEqual(result["status"], "passed", result.get("error"))
        self.assertFalse(result["history"])
        self.assertEqual(len(result["losses"]), 2)
        self.assertTrue(all(np.isfinite(v) for v in result["losses"]))
        self.assertEqual(len(result["metrics"]), 2)
        self.assertIn("accuracy", result["metrics"][0])
        self.assertIn("n_samples", result["metrics"][0])


class PsCacheCleanupTest(unittest.TestCase):
    def _init(self, engine, params):
        for worker in range(engine.n_workers):
            engine.handle(_msg(engine, worker, "init"))
            engine.handle(_msg(engine, worker, "initial_state_applied",
                               parameter_hash=payload_hash(params)))

    def test_completed_rounds_do_not_keep_full_tensors(self):
        size = 50000
        engine, params = _engine(n=1, shards=[32], size=size)
        self._init(engine, params)
        grads = {"w": np.ones((size,), np.float32)}
        bytes_at = []
        for step in range(5):
            _complete_round(engine, params, step, step, grads)
            bytes_at.append(engine.retained_payload_bytes())
        five = bytes_at[-1]
        for step in range(5, 25):
            last, _pull_id = _complete_round(engine, params, step, step, grads)
            self.assertTrue(last.get("round_ready"))
        twenty_five = engine.retained_payload_bytes()
        self.assertLess(five, 8 * 1024 * 1024)
        self.assertLess(twenty_five, 8 * 1024 * 1024)
        self.assertLess(twenty_five / max(five, 1), 3.0)
        retry = engine.handle(_msg(
            engine, 0, "push", request_id="push-0-0", global_step=0, parameter_version=0,
            n_samples=32, loss=0.5, tensors=encode_tensors(grads), payload_hash=payload_hash(grads),
        ))
        self.assertEqual(retry["status"], STATUS_STALE)
        conflict = engine.handle(_msg(
            engine, 0, "push", request_id="push-0-24", global_step=24, parameter_version=24,
            n_samples=32, loss=9.0, tensors=encode_tensors(grads), payload_hash=payload_hash(grads),
        ))
        self.assertEqual(conflict["status"], STATUS_DUP)

    def test_request_index_stays_bounded_and_window_retries_work(self):
        engine, params = _engine(n=1, shards=[32], size=1)
        self._init(engine, params)
        grads = {"w": np.ones((1,), np.float32)}
        sizes = []
        footprints = []
        for step in range(80):
            last, pull_id = _complete_round(engine, params, step, step, grads)
            self.assertTrue(last.get("round_ready"))
            sizes.append(engine.cache_sizes())
            footprints.append(engine.cache_footprint_bytes())
        last_size = sizes[-1]
        self.assertLessEqual(last_size["request_index"], 1 + 3 * engine.digest_rounds + 3)
        self.assertLessEqual(last_size["rounds"], engine.digest_rounds + 1)
        self.assertLessEqual(last_size["reply_cache"], 1 + 3 * engine.retain_rounds + 3)
        self.assertLess(max(item["request_index"] for item in sizes[10:]), 40)
        self.assertLess(footprints[-1] / max(footprints[10], 1), 3.0)
        in_window = engine.handle(_msg(
            engine, 0, "push", request_id="push-0-79", global_step=79, parameter_version=79,
            n_samples=32, loss=0.5, tensors=encode_tensors(grads), payload_hash=payload_hash(grads),
        ))
        self.assertEqual(in_window["status"], STATUS_OK)
        conflict = engine.handle(_msg(
            engine, 0, "push", request_id="push-0-79", global_step=79, parameter_version=79,
            n_samples=32, loss=9.0, tensors=encode_tensors(grads), payload_hash=payload_hash(grads),
        ))
        self.assertEqual(conflict["status"], STATUS_DUP)
        stale = engine.handle(_msg(
            engine, 0, "push", request_id="push-0-0", global_step=0, parameter_version=0,
            n_samples=32, loss=0.5, tensors=encode_tensors(grads), payload_hash=payload_hash(grads),
        ))
        self.assertEqual(stale["status"], STATUS_STALE)
        self.assertEqual(engine.committed_version, 80)

    def test_multi_worker_release_retry_after_commit_is_not_deadlocked(self):
        engine, params = _engine(n=2, shards=[20, 12], size=4)
        self._init(engine, params)
        grads = {"w": np.ones((4,), np.float32)}
        last, _pull = _complete_round(engine, params, 0, 0, grads)
        self.assertTrue(last.get("round_ready"))
        retry = engine.handle(_msg(
            engine, 0, "update_applied", request_id="upd-0-0", global_step=0,
            parameter_version=1, update_id=last["update_id"], optimizer_step=1, digest="same",
        ))
        self.assertEqual(retry["status"], STATUS_OK)
        self.assertTrue(retry.get("round_ready"))
        for step in range(1, 6):
            last, _pull = _complete_round(engine, params, step, step, grads)
            self.assertTrue(last.get("round_ready"))
        late = engine.handle(_msg(
            engine, 1, "update_applied", request_id="upd-1-0", global_step=0,
            parameter_version=1, update_id="run-cache:0:0", optimizer_step=1, digest="same",
        ))
        self.assertEqual(late["status"], STATUS_STALE)

    def test_pull_retry_after_gc_returns_lightweight_status(self):
        engine, params = _engine(n=1, shards=[32], size=16)
        self._init(engine, params)
        grads = {"w": np.ones((16,), np.float32)}
        last, pull_id = _complete_round(engine, params, 0, 0, grads)
        self.assertTrue(last.get("round_ready"))
        for step in range(1, 4):
            _complete_round(engine, params, step, step, grads)
        retry = engine.handle(_msg(
            engine, 0, "pull", request_id=pull_id, global_step=0, parameter_version=0,
        ))
        self.assertEqual(retry["status"], STATUS_OK)
        self.assertFalse(retry.get("tensors"))

    def _waiting_then_commit(self, engine, params, retain_rounds=None):
        if retain_rounds is not None:
            engine.retain_rounds = int(retain_rounds)
        self._init(engine, params)
        grads = {"w": np.ones((engine.schema[0]["shape"][0],), np.float32)}
        for worker in range(engine.n_workers):
            engine.handle(_msg(
                engine, worker, "push", request_id=f"push-{worker}-0",
                global_step=0, parameter_version=0, n_samples=engine.shard_sizes[worker],
                loss=0.5, tensors=encode_tensors(grads), payload_hash=payload_hash(grads),
            ))
        pulls = [
            engine.handle(_msg(
                engine, worker, "pull", request_id=f"pull-{worker}-0",
                global_step=0, parameter_version=0,
            ))
            for worker in range(engine.n_workers)
        ]
        update_id = pulls[0]["update_id"]
        waiting = engine.handle(_msg(
            engine, 0, "update_applied", request_id="upd-0-0",
            global_step=0, parameter_version=1, update_id=update_id,
            optimizer_step=1, digest="same",
        ))
        self.assertEqual(waiting["status"], STATUS_WAITING, waiting)
        committed = engine.handle(_msg(
            engine, 1, "update_applied", request_id="upd-1-0",
            global_step=0, parameter_version=1, update_id=update_id,
            optimizer_step=1, digest="same",
        ))
        self.assertEqual(committed["status"], STATUS_OK, committed)
        self.assertTrue(committed.get("round_ready"))
        return update_id, grads

    def test_identical_waiting_update_retry_succeeds_after_commit(self):
        engine, params = _engine(n=2, shards=[20, 12], size=4)
        update_id, _grads = self._waiting_then_commit(engine, params)
        retry = engine.handle(_msg(
            engine, 0, "update_applied", request_id="upd-0-0",
            global_step=0, parameter_version=1, update_id=update_id,
            optimizer_step=1, digest="same",
        ))
        self.assertEqual(retry["status"], STATUS_OK, retry)
        self.assertTrue(retry.get("round_ready"))
        self.assertEqual(engine.committed_version, 1)

    def test_waiting_update_conflicts_are_rejected_after_commit(self):
        for field, value in (("digest", "tampered"), ("parameter_version", 99), ("optimizer_step", 99)):
            with self.subTest(field=field):
                engine, params = _engine(n=2, shards=[20, 12], size=4)
                update_id, _grads = self._waiting_then_commit(engine, params)
                payload = dict(
                    request_id="upd-0-0", global_step=0, parameter_version=1,
                    update_id=update_id, optimizer_step=1, digest="same",
                )
                payload[field] = value
                conflict = engine.handle(_msg(engine, 0, "update_applied", **payload))
                self.assertEqual(conflict["status"], STATUS_DUP, conflict)
                self.assertEqual(engine.committed_version, 1)

    def test_new_request_id_cannot_bypass_applied_summary(self):
        engine, params = _engine(n=2, shards=[20, 12], size=4)
        update_id, _grads = self._waiting_then_commit(engine, params)
        conflict = engine.handle(_msg(
            engine, 0, "update_applied", request_id="upd-0-0-other",
            global_step=0, parameter_version=1, update_id=update_id,
            optimizer_step=1, digest="other-digest",
        ))
        self.assertEqual(conflict["status"], STATUS_DUP, conflict)
        self.assertEqual(engine.committed_version, 1)

    def test_retain_zero_still_rejects_conflicts_inside_digest_window(self):
        engine, params = _engine(n=2, shards=[20, 12], size=8)
        engine.retain_rounds = 0
        engine.digest_rounds = 4
        update_id, grads = self._waiting_then_commit(engine, params, retain_rounds=0)
        rnd = engine.rounds[0]
        self.assertIsNone((rnd.avg or {}).get("gradients"))
        conflict = engine.handle(_msg(
            engine, 0, "update_applied", request_id="upd-0-0",
            global_step=0, parameter_version=1, update_id=update_id,
            optimizer_step=1, digest="nope",
        ))
        self.assertEqual(conflict["status"], STATUS_DUP, conflict)
        same = engine.handle(_msg(
            engine, 0, "update_applied", request_id="upd-0-0",
            global_step=0, parameter_version=1, update_id=update_id,
            optimizer_step=1, digest="same",
        ))
        self.assertEqual(same["status"], STATUS_OK, same)
        for step in range(1, 6):
            last, _pull = _complete_round(engine, params, step, step, grads)
            self.assertTrue(last.get("round_ready"))
        stale = engine.handle(_msg(
            engine, 0, "update_applied", request_id="upd-0-0",
            global_step=0, parameter_version=1, update_id=update_id,
            optimizer_step=1, digest="same",
        ))
        self.assertEqual(stale["status"], STATUS_STALE, stale)
        self.assertEqual(engine.committed_version, 6)


class RingInboxCleanupTest(unittest.TestCase):
    def test_consumed_chunks_release_payload_across_steps(self):
        inbox = RingInbox(digest_steps=3)
        payload = np.arange(20000, dtype=np.float32).tobytes()
        sizes = []
        for step in range(12):
            key = (f"c:{step}", "SCATTER", 0, 0, 1, step)
            put = inbox.put({
                "collective_id": f"c:{step}",
                "stage": "SCATTER",
                "round": 0,
                "chunk_id": 0,
                "sender_rank": 1,
                "global_step": step,
                "payload_hash": f"h{step}",
                "data": payload,
                "request_id": f"r{step}",
            })
            self.assertEqual(put["status"], STATUS_OK)
            got = inbox.wait(key, timeout=0.2)
            self.assertEqual(got["data"], payload)
            dup = inbox.put({
                "collective_id": f"c:{step}",
                "stage": "SCATTER",
                "round": 0,
                "chunk_id": 0,
                "sender_rank": 1,
                "global_step": step,
                "payload_hash": f"h{step}",
                "data": payload,
                "request_id": f"r{step}-dup",
            })
            self.assertTrue(dup.get("duplicate"))
            sizes.append(inbox.payload_bytes())
        self.assertEqual(inbox.buffered_count(), 0)
        self.assertLess(max(sizes), 2 * len(payload))
        early = inbox.put({
            "collective_id": "c:early",
            "stage": "GATHER",
            "round": 0,
            "chunk_id": 1,
            "sender_rank": 0,
            "global_step": 99,
            "payload_hash": "early",
            "data": b"abc",
            "request_id": "early",
        })
        self.assertEqual(early["status"], STATUS_OK)
        got = inbox.wait(("c:early", "GATHER", 0, 1, 0, 99), timeout=0.2)
        self.assertEqual(got["data"], b"abc")


class LocalUpdatePathTest(unittest.TestCase):
    def test_single_and_ring_n1_skip_communication_and_cpu_grad_materialize(self):
        single = run_training(
            mode="single", transport="none", task="synthetic", steps=3, seed=1,
            collect_history=True, timeout=20,
        )
        ring = run_training(
            mode="ring", transport="grpc_proto", task="synthetic", train_workers=1,
            steps=3, seed=1, shard_sizes=[32], collect_history=True, timeout=20,
        )
        self.assertEqual(single["status"], "passed", single.get("error"))
        self.assertEqual(ring["status"], "passed", ring.get("error"))
        self.assertFalse(single["communication"])
        self.assertFalse(ring["communication"])
        self.assertEqual(single["cpu_grad_materializations"], 0)
        self.assertEqual(ring["cpu_grad_materializations"], 0)
        for actual, expected in zip(ring["history"], single["history"]):
            for name in actual:
                np.testing.assert_allclose(actual[name], expected[name], atol=1e-5, rtol=1e-4)

    def test_ps_n1_still_uses_push_pull(self):
        result = run_training(
            mode="ps", transport="socket_json", task="synthetic", train_workers=1,
            steps=2, shard_sizes=[32], timeout=20, collect_history=True,
        )
        self.assertEqual(result["status"], "passed", result.get("error"))
        self.assertTrue(result["communication"])
        self.assertGreaterEqual(result["cpu_grad_materializations"], 2)
        acks = [e for e in result["events"] if e.get("phase") == "gradient_acknowledged"]
        self.assertEqual(len(acks), 2)

    def test_session_local_grads_match_cpu_reference(self):
        set_device("cpu")
        local = TrainSession({
            "task": "synthetic", "seed": 0, "optimizer": "adam", "learning_rate": 0.01,
            "device": "cpu", "train_workers": 1, "global_batch": 8, "steps": 2,
        })
        ref = TrainSession({
            "task": "synthetic", "seed": 0, "optimizer": "adam", "learning_rate": 0.01,
            "device": "cpu", "train_workers": 1, "global_batch": 8, "steps": 2,
        })
        rng = np.random.default_rng(4)
        for step in range(3):
            x = rng.normal(size=(8, 4)).astype(np.float32)
            y = rng.normal(size=(8, 2)).astype(np.float32)
            computed = local.forward_backward(x, y, materialize_cpu_gradients=False)
            self.assertIsNone(computed.get("gradients"))
            self.assertTrue(computed["device_gradients"])
            local.apply_global_gradients(computed["device_gradients"], f"local-{step}")
            copied = ref.forward_backward(x, y, materialize_cpu_gradients=True)
            ref.apply_global_gradients(copied["gradients"], f"ref-{step}")
        for name in local.params:
            np.testing.assert_allclose(
                np.asarray(local.params[name].value),
                np.asarray(ref.params[name].value),
                atol=1e-6, rtol=1e-5,
            )

    def test_ps_two_workers_match_single_within_tolerance(self):
        reference = run_training(
            mode="single", transport="none", task="synthetic", steps=4, seed=2,
            collect_history=True, timeout=20,
        )
        ps = run_training(
            mode="ps", transport="socket_json", task="synthetic", train_workers=2,
            steps=4, seed=2, shard_sizes=[16, 16], timeout=20, collect_history=True,
        )
        ring = run_training(
            mode="ring", transport="grpc_proto", task="synthetic", train_workers=2,
            steps=4, seed=2, shard_sizes=[16, 16], timeout=20, collect_history=True,
        )
        self.assertEqual(ps["status"], "passed", ps.get("error"))
        self.assertEqual(ring["status"], "passed", ring.get("error"))
        self.assertTrue(ps["communication"])
        self.assertTrue(ring["communication"])
        for actual in (ps["history"][-1], ring["history"][-1]):
            for name in actual:
                np.testing.assert_allclose(
                    actual[name], reference["history"][-1][name], atol=1e-4, rtol=1e-3)


class DonkeyDispatchSmokeTest(unittest.TestCase):
    def test_single_donkey_with_injected_split(self):
        split = _tiny_donkey_split(n=4)
        result = run_training(
            mode="single", transport="none", task="donkey_cnn", split_arrays=split,
            global_batch=2, epochs=1, steps=1, seed=0, timeout=60,
            collect_history=False, drop_last=True, image_h=120, image_w=160,
        )
        self.assertEqual(result["status"], "passed", result.get("error"))
        self.assertEqual(len(result["losses"]), 1)
        self.assertIn("angle_mae", result["metrics"][0])


class GlobalTrainMetricsTest(unittest.TestCase):
    def test_uneven_shards_match_single_batch_metrics_before_update(self):
        n, dim = 8, 784
        x = np.zeros((n, dim), dtype=np.float32)
        x[5:] = 1.0
        y = np.zeros((n, 1), dtype=np.int64)
        y[:5] = 0
        y[5:] = 1
        split = {
            "train": (x, y),
            "val": (x[:2], y[:2]),
            "test": (x[:2], y[:2]),
        }
        split_meta = {
            "source": "injected-offline", "n_train": 8, "n_val": 2, "n_test": 2,
            "seed": 0, "split_fingerprint": "mnist-offline-8",
        }
        common = dict(
            task="mnist_mlp", split_arrays=split, split_meta=split_meta, hidden=8,
            global_batch=8, epochs=1, steps=1, seed=0, timeout=30,
            collect_history=True, drop_last=True, optimizer="mbgd",
        )
        single = run_training(mode="single", transport="none", **common)
        ps = run_training(mode="ps", transport="socket_json", train_workers=2,
                          shard_sizes=[5, 3], **common)
        self.assertEqual(single["status"], "passed", single.get("error"))
        self.assertEqual(ps["status"], "passed", ps.get("error"))
        self.assertEqual(single["metrics"][0]["scope"], "train_batch")
        self.assertEqual(ps["metrics"][0]["aggregation"], "global")
        self.assertEqual(ps["metrics"][0]["n_samples"], 8)
        self.assertEqual(set(ps["worker_metrics"]), {0, 1})
        self.assertNotEqual(ps["worker_metrics"][0][0]["accuracy"],
                            ps["worker_metrics"][1][0]["accuracy"])
        self.assertAlmostEqual(ps["losses"][0], single["losses"][0], places=5)
        self.assertAlmostEqual(ps["metrics"][0]["accuracy"], single["metrics"][0]["accuracy"], places=5)
        worker_n = sum(row[0]["n_samples"] for row in ps["worker_metrics"].values())
        self.assertEqual(ps["metrics"][0]["n_samples"], worker_n)

    def test_ring_and_ps_global_loss_match_single_on_synthetic(self):
        kwargs = dict(task="synthetic", steps=4, seed=3, optimizer="adam",
                      collect_history=True, timeout=25, global_batch=32)
        single = run_training(mode="single", transport="none", **kwargs)
        ps = run_training(mode="ps", transport="grpc_proto", train_workers=2,
                          shard_sizes=[20, 12], **kwargs)
        ring = run_training(mode="ring", transport="grpc_proto", train_workers=2,
                            shard_sizes=[20, 12], **kwargs)
        self.assertEqual(ps["status"], "passed", ps.get("error"))
        self.assertEqual(ring["status"], "passed", ring.get("error"))
        np.testing.assert_allclose(ps["losses"], single["losses"], atol=1e-4, rtol=1e-3)
        np.testing.assert_allclose(ring["losses"], single["losses"], atol=1e-4, rtol=1e-3)
        self.assertEqual(ps["metrics"][0]["n_samples"], 32)
        self.assertEqual(ring["metrics"][0]["n_samples"], 32)
        self.assertEqual(len(ps["worker_metrics"][0]), 4)
        self.assertEqual(len(ps["worker_metrics"][1]), 4)
        self.assertNotAlmostEqual(ps["worker_metrics"][0][-1]["loss"],
                                  ps["worker_metrics"][1][-1]["loss"])


class DataMetaPropagationTest(unittest.TestCase):
    def test_offline_split_meta_matches_across_modes(self):
        split = _tiny_mnist_split(n=8)
        split_meta = {
            "source": "injected-offline", "n_train": 8, "n_val": 2, "n_test": 2,
            "seed": 7, "split_fingerprint": "fp-offline-7",
        }
        common = dict(
            task="mnist_mlp", split_arrays=split, split_meta=split_meta, hidden=8,
            global_batch=4, epochs=1, steps=1, seed=7, timeout=30,
            collect_history=False, drop_last=True,
        )
        results = {
            "single": run_training(mode="single", transport="none", **common),
            "ps_json": run_training(mode="ps", transport="socket_json", train_workers=2,
                                    shard_sizes=[2, 2], **common),
            "ps_grpc": run_training(mode="ps", transport="grpc_proto", train_workers=2,
                                    shard_sizes=[2, 2], **common),
            "ring": run_training(mode="ring", transport="grpc_proto", train_workers=2,
                                 shard_sizes=[2, 2], **common),
        }
        metas = []
        for name, result in results.items():
            self.assertEqual(result["status"], "passed", f"{name}: {result.get('error')}")
            meta = result.get("data_meta") or {}
            self.assertEqual(meta.get("source"), "injected-offline")
            self.assertEqual(meta.get("n_train"), 8)
            self.assertEqual(meta.get("n_val"), 2)
            self.assertEqual(meta.get("n_test"), 2)
            self.assertEqual(meta.get("seed"), 7)
            self.assertEqual(meta.get("split_fingerprint"), "fp-offline-7")
            metas.append(meta)
        self.assertEqual(metas[0], metas[1])
        self.assertEqual(metas[0], metas[2])
        self.assertEqual(metas[0], metas[3])

    def test_noncontiguous_mnist_split_matches_across_modes(self):
        n, dim = 8, 784
        wide = np.arange(n * dim * 2, dtype=np.float32).reshape(n, dim * 2)
        x = wide[:, ::2]
        y_wide = (np.arange(n * 2, dtype=np.int64) % 10).reshape(n, 2)
        y = y_wide[:, :1]
        self.assertFalse(x.flags.c_contiguous)
        split = {
            "train": (x, y),
            "val": (x[:2], y[:2]),
            "test": (x[:2], y[:2]),
        }
        common = dict(
            task="mnist_mlp", split_arrays=split, hidden=8,
            global_batch=4, epochs=1, steps=1, seed=3, timeout=30,
            collect_history=True, drop_last=True,
        )
        results = {
            "single": run_training(mode="single", transport="none", **common),
            "ps_json": run_training(mode="ps", transport="socket_json", train_workers=2,
                                    shard_sizes=[2, 2], **common),
            "ps_grpc": run_training(mode="ps", transport="grpc_proto", train_workers=2,
                                    shard_sizes=[2, 2], **common),
            "ring": run_training(mode="ring", transport="grpc_proto", train_workers=2,
                                 shard_sizes=[2, 2], **common),
        }
        metas = []
        reference = results["single"]
        self.assertEqual(reference["status"], "passed", reference.get("error"))
        for name, result in results.items():
            self.assertEqual(result["status"], "passed", f"{name}: {result.get('error')}")
            self.assertTrue(result.get("data_meta", {}).get("split_fingerprint"))
            metas.append(result["data_meta"])
            np.testing.assert_allclose(result["losses"], reference["losses"], atol=1e-5, rtol=1e-4)
            np.testing.assert_allclose(
                result["metrics"][0]["accuracy"], reference["metrics"][0]["accuracy"],
                atol=1e-5, rtol=1e-4)
            if name != "single":
                for pname in reference["history"][-1]:
                    np.testing.assert_allclose(
                        result["history"][-1][pname], reference["history"][-1][pname],
                        atol=1e-4, rtol=1e-3)
        self.assertEqual(metas[0], metas[1])
        self.assertEqual(metas[0], metas[2])
        self.assertEqual(metas[0], metas[3])

    def test_conflicting_worker_fingerprints_fail(self):
        split = _tiny_mnist_split(n=8)
        split_meta = {
            "source": "injected-offline", "n_train": 8, "n_val": 2, "n_test": 2,
            "seed": 0, "split_fingerprint": "base-fp",
        }
        result = run_training(
            mode="ps", transport="socket_json", task="mnist_mlp", train_workers=2,
            shard_sizes=[2, 2], split_arrays=split, split_meta=split_meta,
            hidden=8, global_batch=4, epochs=1, steps=1, seed=0, timeout=30,
            collect_history=False, drop_last=True,
            test_data_meta_by_worker={0: "base-fp", 1: "tampered-fp"},
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("data_meta", str(result.get("error") or "").lower())
        self.assertFalse(result.get("data_meta"))

    def test_benchmark_manifest_records_data_split(self):
        import json
        import tempfile
        from pathlib import Path

        from benchmark.distributed_train import main

        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "run")
            main([
                "--mode", "single", "--task", "synthetic", "--steps", "1",
                "--global-batch", "8", "--out-dir", out, "--timeout", "15",
            ])
            manifest = json.loads((Path(out) / "manifest.json").read_text(encoding="utf-8"))
            split = manifest.get("data_split") or {}
            self.assertEqual(split.get("source"), "synthetic")
            self.assertTrue(split.get("split_fingerprint"))
            self.assertIsNotNone(split.get("n_train"))


if __name__ == "__main__":
    unittest.main()
