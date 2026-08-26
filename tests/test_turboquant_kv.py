import math
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune" / "modeling"))

from turboquant_kv import (GroupwiseValueCodec, TurboQuantKeyCodec,
                           TurboQuantKVCodec, pack_bits, spherical_codebook,
                           tensor_bytes, unpack_bits)
import common


class BitPackingTest(unittest.TestCase):
    def test_round_trip_all_supported_widths_and_padding(self):
        for bits in (1, 2, 4):
            for width in (0, 1, 7, 8, 9, 63, 64, 65):
                values = torch.arange(width, dtype=torch.uint8).remainder(1 << bits)
                actual = unpack_bits(pack_bits(values, bits), bits, width)
                self.assertTrue(torch.equal(actual, values), (bits, width))

    def test_rejects_invalid_codes_and_truncated_storage(self):
        with self.assertRaises(ValueError):
            pack_bits(torch.tensor([0, 4]), 2)
        with self.assertRaises(ValueError):
            pack_bits(torch.tensor([0.0, 0.5]), 2)
        with self.assertRaises(ValueError):
            unpack_bits(torch.zeros(1, dtype=torch.uint8), 2, 5)


class TurboQuantKVTest(unittest.TestCase):
    def test_codebook_is_ordered_and_exactly_symmetric(self):
        centroids, boundaries = spherical_codebook(64, 2)
        self.assertTrue(bool(torch.all(centroids[1:] > centroids[:-1])))
        self.assertTrue(bool(torch.all(boundaries[1:] > boundaries[:-1])))
        self.assertTrue(torch.equal(centroids, -centroids.flip(0)))
        self.assertTrue(torch.equal(boundaries, -boundaries.flip(0)))

    def test_seed_is_deterministic_and_changes_projection(self):
        first = TurboQuantKeyCodec(64, seed=17)
        same = TurboQuantKeyCodec(64, seed=17)
        other = TurboQuantKeyCodec(64, seed=18)
        self.assertTrue(torch.equal(first.rotation, same.rotation))
        self.assertTrue(torch.equal(first.sketch, same.sketch))
        self.assertFalse(torch.equal(first.sketch, other.sketch))

    def test_shapes_zero_vectors_and_bfloat16(self):
        codec = TurboQuantKVCodec(64)
        for dtype in (torch.float32, torch.bfloat16):
            queries = torch.zeros(2, 3, 64, dtype=dtype)
            keys = torch.zeros(2, 7, 64, dtype=dtype)
            values = torch.zeros_like(keys)
            result = codec.evaluate(queries, keys, values)
            self.assertEqual(result["scores"].shape, (2, 3, 7))
            self.assertEqual(result["output"].shape, (2, 3, 64))
            for name in ("keys", "values", "scores", "output"):
                self.assertTrue(bool(torch.isfinite(result[name]).all()), name)

    def test_storage_is_exactly_68_bytes_per_vector(self):
        codec = TurboQuantKVCodec(64, key_bits=3, value_bits=4, group_size=32)
        keys = torch.randn(2, 5, 64)
        values = torch.randn_like(keys)
        packed_keys = codec.keys.pack(keys)
        packed_values = codec.values.pack(values)
        self.assertEqual(tensor_bytes(packed_keys), 28 * 10)
        self.assertEqual(tensor_bytes(packed_values), 40 * 10)

    def test_value_quantization_has_low_random_nmse(self):
        torch.manual_seed(3)
        values = torch.randn(4, 17, 64)
        codec = GroupwiseValueCodec(64, bits=4, group_size=32)
        reconstructed = codec.unpack(codec.pack(values))
        nmse = (values - reconstructed).square().mean() / values.square().mean()
        self.assertLess(float(nmse), 0.02)

    def test_qjl_score_estimator_is_statistically_unbiased(self):
        torch.manual_seed(9)
        queries = torch.randn(1, 8, 64)
        keys = torch.randn(1, 32, 64)
        exact = queries @ keys.transpose(-2, -1)
        errors = []
        for seed in range(48):
            codec = TurboQuantKeyCodec(64, seed=seed)
            estimated = codec.scores(queries, codec.pack(keys))
            errors.append((estimated - exact).mean())
        errors = torch.stack(errors)
        standard_error = errors.std(unbiased=True) / math.sqrt(len(errors))
        self.assertLess(abs(float(errors.mean())), 3.0 * float(standard_error) + 1e-5)

    def test_rejects_malformed_shapes(self):
        codec = TurboQuantKVCodec(64)
        with self.assertRaises(ValueError):
            codec.keys.pack(torch.zeros(3, 63))
        with self.assertRaises(ValueError):
            codec.evaluate(torch.zeros(64), torch.zeros(2, 64), torch.zeros(2, 64))
        with self.assertRaises(ValueError):
            codec.evaluate(torch.zeros(1, 64), torch.zeros(2, 64), torch.zeros(3, 64))

    def test_existing_two_bit_fake_and_packed_paths_match(self):
        torch.manual_seed(31)
        values = torch.randn(2, 7, 64)
        packed, scale = common.kv2_pack(values)
        reconstructed = common.kv2_unpack(packed, scale)
        self.assertTrue(torch.equal(common.kv2(values), reconstructed))
        self.assertLess(float((values - reconstructed).square().mean() /
                              values.square().mean()), 0.65)


if __name__ == "__main__":
    unittest.main()
