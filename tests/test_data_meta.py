import pickle
import unittest

import numpy as np

from MyFlows.distributed import data_meta
from MyFlows.distributed.data_meta import (
    ensure_split_meta,
    fingerprint_arrays,
    lightweight_data_meta,
    merge_data_meta,
)


def _split(n=32, dim=16):
    x = np.arange(n * dim, dtype=np.float32).reshape(n, dim)
    y = np.arange(n, dtype=np.int64).reshape(n, 1)
    return {
        "train": (x, y),
        "val": (x[:4].copy(), y[:4].copy()),
        "test": (x[4:8].copy(), y[4:8].copy()),
    }


class DataFingerprintTest(unittest.TestCase):
    def test_middle_feature_change_alters_fingerprint(self):
        base = np.arange(64, dtype=np.float32)
        changed = base.copy()
        changed[32] = 999.0
        np.testing.assert_array_equal(base[:8], changed[:8])
        np.testing.assert_array_equal(base[-8:], changed[-8:])
        self.assertNotEqual(fingerprint_arrays(base), fingerprint_arrays(changed))

    def test_middle_label_change_alters_split_fingerprint(self):
        split = _split()
        other = {
            "train": (split["train"][0].copy(), split["train"][1].copy()),
            "val": split["val"],
            "test": split["test"],
        }
        other["train"][1][10, 0] = 77
        left = ensure_split_meta(split, {"seed": 0})
        right = ensure_split_meta(other, {"seed": 0})
        self.assertNotEqual(left["meta"]["split_fingerprint"], right["meta"]["split_fingerprint"])

    def test_middle_sample_swap_alters_fingerprint(self):
        split = _split()
        x, y = split["train"]
        swapped_x, swapped_y = x.copy(), y.copy()
        swapped_x[10], swapped_x[11] = x[11].copy(), x[10].copy()
        swapped_y[10], swapped_y[11] = y[11].copy(), y[10].copy()
        other = {"train": (swapped_x, swapped_y), "val": split["val"], "test": split["test"]}
        left = ensure_split_meta(split, {"seed": 1})
        right = ensure_split_meta(other, {"seed": 1})
        self.assertNotEqual(left["meta"]["split_fingerprint"], right["meta"]["split_fingerprint"])

    def test_train_val_reassignment_alters_fingerprint(self):
        split = _split()
        x, y = split["train"]
        other = {
            "train": (np.concatenate([x[:15], x[16:]]), np.concatenate([y[:15], y[16:]])),
            "val": (x[15:16], y[15:16]),
            "test": split["test"],
        }
        left = ensure_split_meta(split, {"seed": 2})
        right = ensure_split_meta(other, {"seed": 2})
        self.assertNotEqual(left["meta"]["split_fingerprint"], right["meta"]["split_fingerprint"])

    def test_copied_arrays_match_and_chunk_size_is_stable(self):
        array = np.arange(10000, dtype=np.float32).reshape(100, 100)
        copied = array.copy()
        self.assertEqual(fingerprint_arrays(array), fingerprint_arrays(copied))
        self.assertEqual(
            fingerprint_arrays(array, chunk_bytes=32),
            fingerprint_arrays(array, chunk_bytes=4096),
        )

    def test_merge_rejects_auto_fingerprints_from_different_content(self):
        split = _split()
        other = {
            "train": (split["train"][0].copy(), split["train"][1].copy()),
            "val": split["val"],
            "test": split["test"],
        }
        other["train"][0][20, 8] = -3.0
        left = lightweight_data_meta(ensure_split_meta(split, {"seed": 3, "source": "injected"}), {"seed": 3})
        right = lightweight_data_meta(ensure_split_meta(other, {"seed": 3, "source": "injected"}), {"seed": 3})
        self.assertNotEqual(left["split_fingerprint"], right["split_fingerprint"])
        with self.assertRaises(ValueError):
            merge_data_meta([left, right], 2)

    def test_explicit_split_meta_fingerprint_is_kept(self):
        split = _split()
        split["meta"] = {
            "source": "injected-offline",
            "n_train": 32,
            "n_val": 4,
            "n_test": 4,
            "seed": 9,
            "split_fingerprint": "external-fp",
        }
        kept = ensure_split_meta(split, {"seed": 9})
        self.assertEqual(kept["meta"]["split_fingerprint"], "external-fp")
        auto = ensure_split_meta(_split(), {"seed": 9})
        self.assertNotEqual(auto["meta"]["split_fingerprint"], "external-fp")
        self.assertEqual(len(auto["meta"]["split_fingerprint"]), 64)

    def test_layout_variants_match_contiguous_copy(self):
        base = np.arange(16 * 1568, dtype=np.float32).reshape(16, 1568)
        sliced = base[:, ::2]
        self.assertFalse(sliced.flags.c_contiguous)
        np.testing.assert_array_equal(sliced, sliced.copy())
        self.assertEqual(fingerprint_arrays(sliced), fingerprint_arrays(sliced.copy()))

        transposed = base.T
        self.assertEqual(fingerprint_arrays(transposed), fingerprint_arrays(np.ascontiguousarray(transposed)))

        fortran = np.asfortranarray(base)
        self.assertTrue(fortran.flags.f_contiguous)
        self.assertEqual(fingerprint_arrays(fortran), fingerprint_arrays(np.ascontiguousarray(fortran)))

        reversed_view = base[::-1, ::-1]
        self.assertEqual(fingerprint_arrays(reversed_view), fingerprint_arrays(reversed_view.copy()))

    def test_pickle_roundtrip_keeps_fingerprint(self):
        view = np.arange(16 * 1568, dtype=np.float32).reshape(16, 1568)[:, ::2]
        restored = pickle.loads(pickle.dumps(view))
        self.assertEqual(fingerprint_arrays(view), fingerprint_arrays(restored))
        self.assertEqual(fingerprint_arrays(view), fingerprint_arrays(np.ascontiguousarray(view)))

    def test_chunk_size_stable_for_strided_and_contiguous(self):
        view = np.arange(3000, dtype=np.float32).reshape(50, 60)[:, ::2]
        copied = view.copy()
        for array in (view, copied):
            self.assertEqual(
                fingerprint_arrays(array, chunk_bytes=64),
                fingerprint_arrays(array, chunk_bytes=4096),
            )
        self.assertEqual(
            fingerprint_arrays(view, chunk_bytes=64),
            fingerprint_arrays(copied, chunk_bytes=4096),
        )

    def test_merge_accepts_same_content_different_layout(self):
        split = _split()
        x, y = split["train"]
        wide = np.zeros((x.shape[0], x.shape[1] * 2), dtype=x.dtype)
        wide[:, ::2] = x
        strided = {
            "train": (wide[:, ::2], y),
            "val": split["val"],
            "test": split["test"],
        }
        left = lightweight_data_meta(ensure_split_meta(split, {"seed": 4, "source": "injected"}), {"seed": 4})
        right = lightweight_data_meta(ensure_split_meta(strided, {"seed": 4, "source": "injected"}), {"seed": 4})
        self.assertEqual(left, right)
        self.assertEqual(merge_data_meta([left, right], 2), left)

    def test_nchw_strided_view_hashes_without_per_element_recursion(self):
        hwc = np.arange(4 * 120 * 160 * 3, dtype=np.float32).reshape(4, 120, 160, 3)
        nchw = np.transpose(hwc, (0, 3, 1, 2))
        self.assertFalse(nchw.flags.c_contiguous)
        calls = []
        original = data_meta._hash_array

        def wrapped(sha, array, chunk_bytes):
            calls.append(tuple(array.shape))
            return original(sha, array, chunk_bytes)

        data_meta._hash_array = wrapped
        try:
            digest = fingerprint_arrays(nchw, chunk_bytes=4096)
        finally:
            data_meta._hash_array = original
        self.assertEqual(digest, fingerprint_arrays(nchw.copy(), chunk_bytes=4096))
        self.assertEqual(calls, [(4, 3, 120, 160)])


if __name__ == "__main__":
    unittest.main()
