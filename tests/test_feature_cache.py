from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from MyFlows.transfer.cache import FeatureCache


class FeatureCacheTest(unittest.TestCase):
    def test_cache_hit_and_invalidation_for_weights_preprocessing_and_order(self):
        with TemporaryDirectory() as tmp:
            cache = FeatureCache(tmp)
            samples = {"train": [1, 2, 3], "val": [4], "test": []}
            calls = []
            def encode(batch):
                calls.append(list(batch))
                return np.asarray([[n, -n] for n in batch], np.float32)
            kwargs = dict(sample_manifest=samples, weights_sha256="first", preprocessing={"size": [32, 32]}, batch=2, width=2, encode=encode)
            arrays, initial = cache.materialize(samples, **kwargs)
            np.testing.assert_array_equal(arrays["train"], [[1, -1], [2, -2], [3, -3]])
            self.assertEqual(arrays["test"].shape, (0, 2))
            del arrays
            arrays, cached = cache.materialize(samples, **kwargs)
            self.assertTrue(cached["hit"])
            del arrays
            self.assertEqual(calls, [[1, 2], [3], [4]])
            for weights, prep, order in (("changed", kwargs["preprocessing"], samples), ("first", {"size": [64, 64]}, samples),
                                         ("first", kwargs["preprocessing"], {**samples, "train": [3, 2, 1]})):
                self.assertNotEqual(initial["key"], cache.key(weights, prep, order))

    def test_encoder_failure_does_not_publish_partial_cache(self):
        with TemporaryDirectory() as tmp:
            cache = FeatureCache(tmp)
            with self.assertRaisesRegex(ValueError, "invalid FP32"):
                cache.materialize({"train": [1]}, sample_manifest={"train": [1]}, weights_sha256="x", preprocessing={},
                                  batch=1, width=2, encode=lambda items: np.ones((2, 5)))
            self.assertEqual(list(Path(tmp).iterdir()), [])
