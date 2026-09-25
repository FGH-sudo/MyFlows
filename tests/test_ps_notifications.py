"""Blocking PS requests must wake on state changes, including cancellation."""

import threading
import unittest
from unittest.mock import patch

import numpy as np

from MyFlows.distributed.constants import PROTOCOL_VERSION, STATUS_ABORTED, STATUS_OK, STATUS_WAITING
from MyFlows.distributed.engine import PSEngine
from MyFlows.distributed.launcher import normalize_config, run_training


class PsNotificationsTest(unittest.TestCase):
    def setUp(self):
        self.engine = PSEngine(run_id='notify-test', n_workers=2,
                               init_parameters={'w': np.ones(2, dtype=np.float32)},
                               optimizer_meta={'t': 0}, shard_sizes=[2, 2], wait_strategy='notify')
        self.next_id = 0
        for worker in range(2):
            self.engine.handle(self.message(worker, 'initial_state_applied',
                                            parameter_hash=self.engine.init_parameter_hash))

    def message(self, worker, kind, **fields):
        self.next_id += 1
        return dict(protocol_version=PROTOCOL_VERSION, run_id=self.engine.run_id,
                    worker_id=worker, request_id=str(self.next_id), type=kind,
                    global_step=0, parameter_version=0, **fields)

    def push(self, worker):
        return self.engine.handle(self.message(worker, 'push', n_samples=2, loss=1.,
                                               tensor_arrays={'w': np.full(2, worker + 1., np.float32)}))

    def confirm(self, worker):
        msg = self.message(worker, 'update_applied', update_id='notify-test:0:0',
                           optimizer_step=1, digest='same-state')
        msg['parameter_version'] = 1
        return msg

    def run_waiting(self, message, transition):
        # Observe entry to the real wait while its lock is held. The transition
        # cannot acquire that lock until wait has atomically released it.
        # A generous join deadline tests notification, not scheduler latency.
        entered = threading.Event()
        replies = []
        original_wait = self.engine.cv.wait

        def observed_wait(timeout=None):
            entered.set()
            return original_wait(timeout)

        with patch.object(self.engine.cv, 'wait', side_effect=observed_wait):
            thread = threading.Thread(target=lambda: replies.append(self.engine.handle(message, wait_s=60)),
                                      daemon=True)
            thread.start()
            try:
                self.assertTrue(entered.wait(5), 'request did not block')
                transition()
                thread.join(5)
                self.assertFalse(thread.is_alive(), 'state change failed to wake the request')
            finally:
                self.engine.abort('test cleanup')
                thread.join(5)
        self.assertEqual(len(replies), 1)
        return replies[0]

    def test_last_push_releases_pull(self):
        self.push(0)
        reply = self.run_waiting(self.message(0, 'pull'), lambda: self.push(1))
        self.assertEqual(reply['status'], STATUS_OK)
        self.assertEqual(reply['tensors'][0]['data'], [1.5, 1.5])

    def test_last_confirmation_releases_peer(self):
        self.push(0)
        self.push(1)
        reply = self.run_waiting(self.confirm(0), lambda: self.engine.handle(self.confirm(1)))
        self.assertEqual(reply['status'], STATUS_OK)
        self.assertTrue(reply['round_ready'])
        self.assertEqual(self.engine.committed_version, 1)

    def test_abort_releases_pull(self):
        reply = self.run_waiting(self.message(0, 'pull'), lambda: self.engine.abort('lost worker'))
        self.assertEqual(reply['status'], STATUS_ABORTED)
        self.assertEqual(reply['error'], 'lost worker')

    def test_stop_releases_confirmation(self):
        self.push(0)
        self.push(1)
        reply = self.run_waiting(self.confirm(0), lambda: self.engine.handle(self.message(1, 'stop')))
        self.assertEqual(reply['status'], STATUS_ABORTED)

    def test_deadline_preserves_waiting_and_request_can_be_retried(self):
        msg = self.message(0, 'pull')
        self.assertEqual(self.engine.handle(msg, wait_s=.01)['status'], STATUS_WAITING)
        self.push(0)
        self.push(1)
        self.assertEqual(self.engine.handle(msg, wait_s=.01)['status'], STATUS_OK)

    def test_strategy_is_validated_and_poll_remains_default(self):
        self.assertEqual(normalize_config()['ps_wait_strategy'], 'poll')
        with self.assertRaisesRegex(ValueError, 'ps_wait_strategy'):
            normalize_config(ps_wait_strategy='unknown')

    def test_notify_matches_poll_with_real_workers_and_retry(self):
        config = dict(mode='ps', transport='grpc_proto', task='synthetic', device='cpu',
                      train_workers=2, shard_sizes=[20, 12], optimizer='adam', steps=4,
                      collect_history=True, timeout=20, fault='retry')
        results = [run_training(**config, ps_wait_strategy=strategy) for strategy in ('poll', 'notify')]
        for result in results:
            self.assertEqual(result['status'], 'passed', result.get('error'))
            self.assertFalse(result['alive_pids'])
            self.assertEqual(len(result['history']), 5)
        np.testing.assert_array_equal(results[0]['losses'], results[1]['losses'])
        for left, right in zip(results[0]['history'], results[1]['history']):
            for name in left:
                np.testing.assert_array_equal(left[name], right[name])


if __name__ == '__main__':
    unittest.main()
