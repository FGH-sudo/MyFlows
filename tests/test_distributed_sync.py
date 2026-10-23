"""SyncRun / RankSync: a numpy training loop over PS and Ring matches full-batch training."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from MyFlows.distributed.protocol import payload_hash
from MyFlows.distributed.sync import SyncRun, save_state

STEPS = 4
LEARNING_RATE = 0.1


def _data(global_batch):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(STEPS, global_batch, 3)).astype(np.float32)
    y = (x @ np.array([0.5, -1.0, 2.0], np.float32) + 0.25).astype(np.float32)
    return x, y


def _gradients(weight, bias, x, y):
    error = x @ weight + bias - y
    return {"w": 2 * x.T @ error / len(y), "b": np.array([2 * error.mean()], np.float32)}, float(np.mean(error ** 2))


def _initial():
    return {"w": np.array([0.1, 0.2, -0.3], np.float32), "b": np.zeros((1,), np.float32),
            "running": np.zeros((3,), np.float32)}


def _rank(rank, sync, out_dir, shard_sizes, stop_after):
    """Plain SGD on this rank's shard; ``running`` imitates a BN statistic averaged across ranks."""
    state = sync.initialize(_initial())
    x, y = _data(sum(shard_sizes))
    start = sum(shard_sizes[:rank])
    shard = slice(start, start + shard_sizes[rank])
    losses = []
    for step in range(STEPS):
        grads, loss = _gradients(state["w"], state["b"], x[step, shard], y[step, shard])
        local = dict(grads, running=x[step, shard].mean(axis=0))
        means, global_loss = sync.allreduce(local, shard_sizes[rank], loss)
        state["w"] = state["w"] - LEARNING_RATE * means["w"]
        state["b"] = state["b"] - LEARNING_RATE * means["b"]
        state["running"] = means["running"]
        losses.append(global_loss)
        if sync.commit(payload_hash(state), stop=rank == 1 and step == stop_after):
            break
    totals = sync.reduce("sums", {"rank": np.float32(rank), "ones": np.ones(2, np.float32)})
    np.savez(Path(out_dir) / f"rank-{rank}.npz", **state)
    (Path(out_dir) / f"rank-{rank}.json").write_text(json.dumps({
        "losses": losses, "rank_sum": float(totals["rank"][0]), "ones": totals["ones"].tolist()}))


def _reference(global_batch):
    x, y = _data(global_batch)
    state = _initial()
    losses = []
    for step in range(STEPS):
        grads, loss = _gradients(state["w"], state["b"], x[step], y[step])
        state["w"] = state["w"] - LEARNING_RATE * grads["w"]
        state["b"] = state["b"] - LEARNING_RATE * grads["b"]
        state["running"] = x[step].mean(axis=0)
        losses.append(loss)
    return state, losses


class SyncRunTests(unittest.TestCase):
    def _run(self, mode, transport, shard_sizes, stop_after=None):
        with tempfile.TemporaryDirectory() as tmp:
            initial = Path(tmp) / "initial.npz"
            save_state(initial, _initial())
            with SyncRun(mode=mode, transport=transport, world_size=len(shard_sizes), shard_sizes=shard_sizes,
                         initial_state=initial, timeout=30) as run:
                run.start(_rank, (tmp, shard_sizes, -1 if stop_after is None else stop_after))
                outcome = run.wait()
            self.assertEqual(outcome["status"], "passed", outcome.get("traceback"))
            return [(dict(np.load(Path(tmp) / f"rank-{r}.npz")),
                     json.loads((Path(tmp) / f"rank-{r}.json").read_text()))
                    for r in range(len(shard_sizes))]

    def test_ps_and_ring_match_full_batch_training(self):
        expected, expected_losses = _reference(12)
        for mode, transport, shards in (("ps", "grpc_proto", [7, 5]), ("ps", "socket_json", [4, 4, 4]),
                                        ("ring", "grpc_proto", [5, 4, 3]), ("ring", "grpc_proto", [12])):
            with self.subTest(mode=mode, transport=transport, shards=shards):
                for state, info in self._run(mode, transport, shards):
                    for name, value in expected.items():
                        np.testing.assert_allclose(state[name], value, rtol=1e-5, atol=1e-6)
                    np.testing.assert_allclose(info["losses"], expected_losses, rtol=1e-5)
                    self.assertEqual(info["rank_sum"], sum(range(len(shards))))
                    self.assertEqual(info["ones"], [len(shards)] * 2)

    def test_one_rank_stopping_stops_every_rank_after_the_same_step(self):
        for mode in ("ps", "ring"):
            with self.subTest(mode=mode):
                ranks = self._run(mode, "grpc_proto", [6, 6], stop_after=1)
                self.assertEqual([len(info["losses"]) for _, info in ranks], [2, 2])

    def test_ring_barrier_rejects_different_initial_states(self):
        from MyFlows.distributed.ring import MetadataBarrier

        barrier = MetadataBarrier(2)
        self.assertEqual(barrier.handle({"type": "join_init", "worker_id": 0, "digest": "a"})["status"], "WAITING")
        reply = barrier.handle({"type": "join_init", "worker_id": 1, "digest": "b"})
        self.assertEqual(reply["status"], "DUPLICATE_CONFLICT")

    def test_rank_failure_fails_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            initial = Path(tmp) / "initial.npz"
            save_state(initial, {"other": np.zeros(2, np.float32)})
            with SyncRun(mode="ps", transport="grpc_proto", world_size=2, shard_sizes=[6, 6],
                         initial_state=initial, timeout=10) as run:
                run.start(_rank, (tmp, [6, 6], -1))
                outcome = run.wait()
        self.assertEqual(outcome["status"], "failed")
        self.assertIn("differ from the run's initial state", outcome["error"])


if __name__ == "__main__":
    unittest.main()
