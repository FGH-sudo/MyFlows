import struct
import unittest

import numpy as np

from MyFlows.distributed.gradient_layout import (
    allgather_apply,
    build_layout,
    flatten_gradients,
    padded_chunks,
    restore_gradients,
    scatter_reduce_apply,
    simulate_ring_allreduce,
)
from MyFlows.distributed.protocol import (
    PROTOCOL_VERSION,
    aggregate_weighted,
    decode_tensors,
    encode_tensors,
    pack_frame,
    payload_hash,
    recv_frame_from_buffer,
    schema_from_arrays,
    schema_hash,
    unpack_frame,
)
from MyFlows.distributed.shards import split_global_batch, split_indices


class ProtocolSchemaTest(unittest.TestCase):
    def test_schema_and_payload_hash_are_stable(self):
        arrays = {
            "fc1.weight": np.arange(8, dtype=np.float32).reshape(4, 2),
            "fc1.bias": np.ones(2, dtype=np.float32),
        }
        other = {k: v.copy() for k, v in arrays.items()}
        self.assertEqual(schema_hash(arrays), schema_hash(other))
        self.assertEqual(payload_hash(arrays), payload_hash(other))
        other["fc1.bias"] = other["fc1.bias"] + 1
        self.assertNotEqual(payload_hash(arrays), payload_hash(other))

    def test_encode_rejects_non_finite_and_roundtrips_fp32(self):
        arrays = {"w": np.array([[1.5, -2.0], [0.0, 3.25]], dtype=np.float32)}
        encoded = encode_tensors(arrays)
        restored = decode_tensors(encoded, schema_from_arrays(arrays))
        np.testing.assert_array_equal(restored["w"], arrays["w"])
        with self.assertRaises(ValueError):
            encode_tensors({"w": np.array([np.nan], dtype=np.float32)})
        with self.assertRaises(ValueError):
            encode_tensors({"w": np.array([np.inf], dtype=np.float32)})

    def test_length_prefixed_json_handles_split_and_sticky_frames(self):
        messages = [
            {"protocol_version": PROTOCOL_VERSION, "type": "push", "n": 1},
            {"protocol_version": PROTOCOL_VERSION, "type": "pull", "n": 2},
        ]
        blob = b"".join(pack_frame(m) for m in messages)
        first_len = struct.unpack("!I", blob[:4])[0]
        self.assertEqual(blob[4:4 + first_len], pack_frame(messages[0])[4:])
        pieces = [blob[i:i + 3] for i in range(0, len(blob), 3)]
        buffer = bytearray()
        decoded = []
        for piece in pieces:
            buffer.extend(piece)
            while True:
                frame, buffer = recv_frame_from_buffer(buffer)
                if frame is None:
                    break
                decoded.append(unpack_frame(frame))
        self.assertEqual(decoded, messages)
        framed = pack_frame({"ok": 1})
        self.assertEqual(struct.unpack("!I", framed[:4])[0], len(framed) - 4)
        with self.assertRaises(ValueError):
            pack_frame({"payload": "x" * 100}, max_bytes=8)

    def test_weighted_mean_uses_sample_counts(self):
        g0 = {"w": np.ones((2, 2), np.float32)}
        g1 = {"w": np.full((2, 2), 3.0, np.float32)}
        mean, loss, total = aggregate_weighted(
            [{"n_samples": 20, "loss": 1.0, "gradients": g0},
             {"n_samples": 12, "loss": 3.0, "gradients": g1}]
        )
        self.assertEqual(total, 32)
        self.assertAlmostEqual(loss, 1.75)
        np.testing.assert_allclose(mean["w"], np.full((2, 2), 1.75, np.float32))


class ShardSplitTest(unittest.TestCase):
    def test_even_and_uneven_shards_cover_batch_without_overlap(self):
        even = split_global_batch(32, 2)
        self.assertEqual(even, [16, 16])
        uneven = split_global_batch(32, 2, shard_sizes=[20, 12])
        self.assertEqual(uneven, [20, 12])
        four, _dropped = split_indices(np.arange(10), 4)
        lengths = [len(part) for part in four]
        self.assertEqual(sum(lengths), 10)
        merged = np.concatenate(four)
        np.testing.assert_array_equal(np.sort(merged), np.arange(10))
        self.assertEqual(len(np.unique(merged)), 10)
        dropped, n_drop = split_indices(np.arange(10), 3, global_batch=4, drop_last=True)
        self.assertEqual(sum(len(p) for p in dropped), 8)
        self.assertEqual(n_drop, 2)

    def test_rejects_empty_or_mismatched_shards(self):
        with self.assertRaises(ValueError):
            split_global_batch(32, 2, shard_sizes=[32, 0])
        with self.assertRaises(ValueError):
            split_global_batch(32, 2, shard_sizes=[20, 10])


class GradientLayoutRingTest(unittest.TestCase):
    def test_flatten_pad_restore_covers_uneven_and_small_vectors(self):
        grads = {
            "a": np.arange(3, dtype=np.float32),
            "b": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        }
        layout = build_layout(schema_from_arrays(grads))
        flat = flatten_gradients(grads, layout)
        self.assertEqual(flat.dtype, np.float32)
        restored = restore_gradients(flat, layout)
        np.testing.assert_array_equal(restored["a"], grads["a"])
        np.testing.assert_array_equal(restored["b"], grads["b"])
        chunks, pad = padded_chunks(np.arange(3, dtype=np.float32), 4)
        self.assertEqual(chunks.shape, (4, 1))
        self.assertEqual(pad, 1)
        tiny = padded_chunks(np.array([9.0], dtype=np.float32), 4)
        self.assertEqual(tiny[0].shape, (4, 1))

    def test_scatter_reduce_then_allgather_matches_weighted_sum(self):
        for n in (1, 2, 4, 5):
            with self.subTest(n=n):
                vectors = [np.full(7, float(r + 1), dtype=np.float32) for r in range(n)]
                counts = [3 + r for r in range(n)]
                weighted = [vectors[r] * counts[r] for r in range(n)]
                reduced = simulate_ring_allreduce(weighted)
                expected = np.sum(weighted, axis=0)
                for got in reduced:
                    np.testing.assert_allclose(got, expected, atol=1e-6)
                mean = reduced[0] / sum(counts)
                ref = expected / sum(counts)
                np.testing.assert_allclose(mean, ref, atol=1e-6)

    def test_chunk_flow_for_two_and_four_ranks(self):
        n = 2
        s = 0
        vectors = [np.arange(8, dtype=np.float32) + 10 * r for r in range(n)]
        chunks = [padded_chunks(v, n)[0].copy() for v in vectors]
        after = []
        for r in range(n):
            left = (r - 1) % n
            send_id = (left - s) % n
            after.append(scatter_reduce_apply(chunks[r], chunks[left][send_id], r, s, n))
        self.assertTrue(np.allclose(after[0][1], chunks[0][1] + chunks[1][1]))
        self.assertTrue(np.allclose(after[1][0], chunks[1][0] + chunks[0][0]))
        gathered = []
        for r in range(n):
            left = (r - 1) % n
            send_id = (left + 1 - s) % n
            gathered.append(allgather_apply(after[r], after[left][send_id], r, s, n))
        expected = vectors[0] + vectors[1]
        for got in gathered:
            np.testing.assert_allclose(got.reshape(-1)[:8], expected)

        four = [np.arange(8, dtype=np.float32) * (r + 1) for r in range(4)]
        reduced = simulate_ring_allreduce(four)
        expected_four = np.sum(four, axis=0)
        for got in reduced:
            np.testing.assert_allclose(got, expected_four, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
