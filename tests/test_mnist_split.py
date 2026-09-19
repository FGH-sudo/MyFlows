import hashlib
import unittest

import numpy as np

from MyFlows.distributed.tasks.mnist_mlp import OFFICIAL_MNIST_TEST, OFFICIAL_MNIST_TRAIN, split_official_mnist


def _ordered_mnist_fixture(n=70000):
    x = np.arange(n, dtype=np.float32).reshape(n, 1)
    y = np.arange(n, dtype=np.int64).reshape(n, 1)
    return x, y


class OfficialMnistSplitTest(unittest.TestCase):
    def test_official_test_set_never_enters_train_or_val(self):
        x, y = _ordered_mnist_fixture()
        split, meta = split_official_mnist(x, y, n_train=50000, n_val=10000, seed=3)
        train_labels = set(split["train"][1].ravel().tolist())
        val_labels = set(split["val"][1].ravel().tolist())
        test_labels = set(split["test"][1].ravel().tolist())
        official_test = set(range(OFFICIAL_MNIST_TRAIN, OFFICIAL_MNIST_TRAIN + OFFICIAL_MNIST_TEST))
        official_train = set(range(OFFICIAL_MNIST_TRAIN))
        self.assertEqual(test_labels, official_test)
        self.assertTrue(train_labels.isdisjoint(official_test))
        self.assertTrue(val_labels.isdisjoint(official_test))
        self.assertTrue(train_labels.issubset(official_train))
        self.assertTrue(val_labels.issubset(official_train))
        self.assertTrue(train_labels.isdisjoint(val_labels))
        self.assertEqual(len(split["train"][0]), 50000)
        self.assertEqual(len(split["val"][0]), 10000)
        self.assertEqual(len(split["test"][0]), 10000)
        self.assertEqual(meta["n_train"], 50000)
        self.assertEqual(meta["n_val"], 10000)
        self.assertEqual(meta["n_test"], 10000)
        self.assertEqual(meta["source"], "mnist_784")
        self.assertEqual(meta["seed"], 3)
        self.assertTrue(meta["split_fingerprint"])
        fingerprint = hashlib.sha256(
            np.asarray(meta["train_indices"], dtype=np.int64).tobytes()
            + np.asarray(meta["val_indices"], dtype=np.int64).tobytes()
            + np.asarray(meta["test_indices"], dtype=np.int64).tobytes()
        ).hexdigest()
        self.assertEqual(meta["split_fingerprint"], fingerprint)
        np.testing.assert_array_equal(meta["test_indices"], np.arange(60000, 70000))

    def test_rejects_splits_larger_than_official_train(self):
        x, y = _ordered_mnist_fixture()
        with self.assertRaises(ValueError):
            split_official_mnist(x, y, n_train=60001, n_val=0, seed=0)
        with self.assertRaises(ValueError):
            split_official_mnist(x, y, n_train=50000, n_val=10001, seed=0)
        with self.assertRaises(ValueError):
            split_official_mnist(x, y, n_train=60000, n_val=1, seed=0)

    def test_shuffling_all_70000_would_fail_this_boundary(self):
        x, y = _ordered_mnist_fixture()
        rng = np.random.default_rng(0)
        order = rng.permutation(70000)
        shuffled = y[order]
        leaked = set(shuffled[:50000].ravel()) & set(range(60000, 70000))
        self.assertTrue(leaked, "fixture assumes a full shuffle mixes the official test set")
        split, _meta = split_official_mnist(x, y, n_train=50000, n_val=10000, seed=0)
        self.assertTrue(set(split["train"][1].ravel()).isdisjoint(range(60000, 70000)))


if __name__ == "__main__":
    unittest.main()
