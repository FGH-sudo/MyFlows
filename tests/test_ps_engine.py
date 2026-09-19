import threading
import time
import unittest
import uuid

import numpy as np

from MyFlows.distributed.constants import PROTOCOL_VERSION, STATUS_DUP, STATUS_OK, STATUS_WAITING
from MyFlows.distributed.engine import PSEngine
from MyFlows.distributed.protocol import encode_tensors, payload_hash


def _engine(n=2, shards=None):
    params = {"w": np.ones((2, 2), np.float32)}
    return PSEngine(
        run_id="run-a",
        n_workers=n,
        init_parameters=params,
        optimizer_meta={"impl": "MBGD", "learning_rate": 0.01, "t": 0},
        shard_sizes=shards or ([20, 12] if n == 2 else [32] * n),
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


class PSEngineTest(unittest.TestCase):
    def _init_workers(self, engine, params):
        for worker in range(engine.n_workers):
            reply = engine.handle(_msg(engine, worker, "init"))
            self.assertEqual(reply["status"], STATUS_OK, reply)
            applied = engine.handle(_msg(
                engine, worker, "initial_state_applied",
                parameter_hash=payload_hash(params),
            ))
            self.assertEqual(applied["status"], STATUS_OK)

    def test_push_pull_weighted_and_update_barrier(self):
        engine, params = _engine()
        self._init_workers(engine, params)
        g0 = {"w": np.ones((2, 2), np.float32)}
        g1 = {"w": np.full((2, 2), 3.0, np.float32)}
        p0 = engine.handle(_msg(engine, 0, "push", global_step=0, parameter_version=0,
                                n_samples=20, loss=1.0, tensors=encode_tensors(g0),
                                payload_hash=payload_hash(g0)))
        self.assertEqual(p0["status"], STATUS_OK)
        waiting = engine.handle(_msg(engine, 0, "pull", global_step=0, parameter_version=0), wait_s=0)
        self.assertEqual(waiting["status"], STATUS_WAITING)
        engine.handle(_msg(engine, 1, "push", global_step=0, parameter_version=0,
                           n_samples=12, loss=3.0, tensors=encode_tensors(g1),
                           payload_hash=payload_hash(g1)))
        pull0 = engine.handle(_msg(engine, 0, "pull", global_step=0, parameter_version=0), wait_s=0)
        pull1 = engine.handle(_msg(engine, 1, "pull", global_step=0, parameter_version=0), wait_s=0)
        self.assertEqual(pull0["status"], STATUS_OK)
        self.assertEqual(pull0["update_id"], pull1["update_id"])
        self.assertEqual(pull0["payload_hash"], pull1["payload_hash"])
        self.assertAlmostEqual(pull0["loss"], 1.75)
        applied0 = engine.handle(_msg(
            engine, 0, "update_applied", global_step=0, parameter_version=1,
            update_id=pull0["update_id"], optimizer_step=1, digest="abc",
        ), wait_s=0)
        self.assertEqual(applied0["status"], STATUS_WAITING)
        applied1 = engine.handle(_msg(
            engine, 1, "update_applied", global_step=0, parameter_version=1,
            update_id=pull0["update_id"], optimizer_step=1, digest="abc",
        ), wait_s=0)
        self.assertEqual(applied1["status"], STATUS_OK)
        self.assertTrue(applied1["round_ready"])
        retry = engine.handle(_msg(
            engine, 0, "update_applied", global_step=0, parameter_version=1,
            update_id=pull0["update_id"], optimizer_step=1, digest="abc",
            request_id=applied0["request_id"],
        ), wait_s=0)
        self.assertEqual(retry["status"], STATUS_OK)
        self.assertTrue(retry["round_ready"])
        self.assertEqual(engine.committed_version, 1)

    def test_duplicate_request_id_is_idempotent_and_conflict_fails(self):
        engine, params = _engine(n=1, shards=[32])
        self._init_workers(engine, params)
        grads = {"w": np.ones((2, 2), np.float32)}
        req = "same-push"
        first = engine.handle(_msg(engine, 0, "push", request_id=req, global_step=0, parameter_version=0,
                                   n_samples=32, loss=0.5, tensors=encode_tensors(grads),
                                   payload_hash=payload_hash(grads)))
        again = engine.handle(_msg(engine, 0, "push", request_id=req, global_step=0, parameter_version=0,
                                   n_samples=32, loss=0.5, tensors=encode_tensors(grads),
                                   payload_hash=payload_hash(grads)))
        self.assertEqual(first["status"], again["status"], STATUS_OK)
        conflict = engine.handle(_msg(engine, 0, "push", request_id=req, global_step=0, parameter_version=0,
                                      n_samples=32, loss=9.0, tensors=encode_tensors(grads),
                                      payload_hash=payload_hash(grads)))
        self.assertEqual(conflict["status"], STATUS_DUP)

    def test_heartbeat_is_not_blocked_by_waiting_pull(self):
        engine, params = _engine()
        self._init_workers(engine, params)
        started = []

        def blocked_pull():
            started.append(True)
            return engine.handle(_msg(engine, 0, "pull", global_step=0, parameter_version=0), wait_s=0.4)

        thread = threading.Thread(target=blocked_pull)
        thread.start()
        while not started:
            time.sleep(0.01)
        hb = engine.handle(_msg(engine, 1, "heartbeat"))
        self.assertEqual(hb["status"], STATUS_OK)
        thread.join(timeout=2)
        self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
