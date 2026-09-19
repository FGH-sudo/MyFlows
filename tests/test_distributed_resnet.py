"""ResNet shard equivalence and immutable BN checkpoint coverage."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from MyFlows.distributed.measurement import load_state, save_state
from MyFlows.distributed.session import TrainSession


class DistributedResNetTests(unittest.TestCase):
    def config(self):
        return dict(task='donkey_resnet18', device='cpu', optimizer='adam', learning_rate=1e-4,
                    base_width=2, image_h=32, image_w=32, seed=3, bn_mode='frozen', profile=False)

    def test_frozen_bn_uneven_shards_match_full_batch(self):
        config = self.config()
        reference = TrainSession(config)
        shard = TrainSession(config)
        rng = np.random.default_rng(9)
        x, y = rng.random((5, 3, 32, 32), dtype=np.float32), rng.random((5, 2), dtype=np.float32)
        frozen = {k: v.copy() for k, v in reference.buffers.items()}
        expected = reference.forward_backward(x, y)['gradients']
        parts = [shard.forward_backward(x[:2], y[:2])['gradients'], shard.forward_backward(x[2:], y[2:])['gradients']]
        for name in expected:
            actual = (2*parts[0][name]+3*parts[1][name])/5
            np.testing.assert_allclose(actual, expected[name], atol=1e-4, rtol=1e-3, err_msg=name)
        for name, value in frozen.items():
            np.testing.assert_array_equal(reference.buffers[name], value)
            np.testing.assert_array_equal(shard.buffers[name], value)

    def test_checkpoint_restores_bn_and_adam_next_update(self):
        config = self.config()
        reference = TrainSession(config)
        rng = np.random.default_rng(7)
        x, y = rng.random((2, 3, 32, 32), dtype=np.float32), rng.random((2, 2), dtype=np.float32)
        gradients = reference.forward_backward(x, y)['gradients']
        reference.apply_global_gradients(gradients, '0')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'state.npz'
            save_state(reference, path)
            restored = TrainSession(config)
            for value in restored.buffers.values():
                value[...] = 12
            load_state(restored, path)
            self.assertEqual(restored.digest(), reference.digest())
            for session in (reference, restored):
                grads = session.forward_backward(x, y)['gradients']
                session.apply_global_gradients(grads, '1')
            self.assertEqual(restored.digest(), reference.digest())


if __name__ == '__main__':
    unittest.main()
