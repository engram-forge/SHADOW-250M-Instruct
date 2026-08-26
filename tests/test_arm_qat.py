import os
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "finetune" / "modeling"))
import common
import arm_qat
from weight_formats import (int4_row_pack,int4_row_unpack,ternary_pack,
                            ternary_unpack_compact)


class ArmQATTest(unittest.TestCase):
    def test_int8_power_of_two_contract(self):
        values = torch.tensor([[0.0, 1.0, -2.0, 3.0], [0.125, -0.25, 0.5, -1.0]])
        code, scale, dequantized = common.int8_pot_values(values)
        self.assertEqual(code.dtype, torch.int8)
        self.assertTrue(bool((code.abs() <= 127).all()))
        self.assertTrue(torch.equal(torch.log2(scale), torch.log2(scale).round()))
        self.assertTrue(torch.equal(dequantized, code.float() * scale))

    def test_activation_ste_has_identity_gradient(self):
        values = torch.tensor([[0.2, -0.7, 1.3]], requires_grad=True)
        common.int8_pot_ste(values).sum().backward()
        self.assertTrue(torch.equal(values.grad, torch.ones_like(values)))

    def test_ternary_projection_matches_integer_equation(self):
        weight = torch.tensor([[0.9, -0.2, 0.1, -0.7], [-0.1, 0.4, -0.8, 1.2]])
        values = torch.tensor([[0.3, -0.9, 1.7, -2.1]])
        trits, row_scale, _ = common.ternary_values(weight)
        code, activation_scale, _ = common.int8_pot_values(values)
        integer = code.to(torch.int32) @ trits.to(torch.int32).T
        expected = integer.float() * activation_scale * row_scale.T
        actual = common.int8_pot_values(values)[2] @ common.ternary_values(weight)[2].T
        self.assertTrue(torch.equal(actual, expected))

    def test_int4_row_contract_and_ste(self):
        weight = torch.tensor([[0.9, -0.2, 0.1, -0.7]], requires_grad=True)
        code, scale, reconstructed = common.int4_row_values(weight)
        self.assertTrue(bool((code >= -7).all() and (code <= 7).all()))
        self.assertTrue(torch.equal(reconstructed, code.float() * scale))
        common.int4_row_ste(weight).sum().backward()
        self.assertTrue(torch.equal(weight.grad, torch.ones_like(weight)))

    def test_g32_rvq_uses_ternary_forward(self):
        layer = common.RVQ(4, 2, 32, 1)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[0.9, -0.2, 0.1, -0.7],
                                             [-0.1, 0.4, -0.8, 1.2]]))
        values = torch.tensor([[0.3, -0.9, 1.7, -2.1]])
        expected = values @ common.ternary_values(layer.weight)[2].T
        self.assertTrue(torch.equal(layer(values), expected))

    def test_activation_stats_are_finite(self):
        stats = arm_qat.activation_stats(torch.tensor([[0.0, 0.25, -1.0, 3.0]]))
        self.assertGreaterEqual(stats["saturation_fraction"], 0.0)
        self.assertLessEqual(stats["saturation_fraction"], 1.0)
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in stats.values()))

    def test_attention_mask_sentinel_fits_fp16(self):
        values=torch.zeros(2,2,dtype=torch.float16)
        masked=values.masked_fill(torch.tensor([[False,True],[True,False]]),common.mask_min(values))
        self.assertTrue(torch.isfinite(masked).all())
        self.assertEqual(masked.min(),torch.finfo(torch.float16).min)

    def test_activation_warmup_schedule(self):
        self.assertEqual(arm_qat.activation_qat_strength(0,100),0.0)
        self.assertEqual(arm_qat.activation_qat_strength(50,100),0.5)
        self.assertEqual(arm_qat.activation_qat_strength(200,100),1.0)
        self.assertEqual(arm_qat.activation_qat_strength(200,100,False),0.0)

    def test_weight_alphabet_changes_ffn_forward(self):
        layer=common.RVQ(4,2,32,1)
        with torch.no_grad(): layer.weight.copy_(torch.tensor(
            [[0.9,-0.2,0.1,-0.7],[-0.1,0.4,-0.8,1.2]]))
        values=torch.tensor([[0.3,-0.9,1.7,-2.1]])
        common.set_ffn_qat("ternary",False,0.0); ternary=layer(values)
        common.set_ffn_qat("int4_row",False,0.0); int4=layer(values)
        self.assertFalse(torch.equal(ternary,int4))
        common.set_ffn_qat("ternary",False,1.0)

    def test_compact_ternary_pack_round_trip_with_padding(self):
        weight=torch.tensor([[0.9,-0.2,0.1,-0.7,1.2,-1.4,0.0]])
        packed,scale,dequantized=ternary_pack(weight,compact=True)
        codes,unpacked=ternary_unpack_compact(packed,scale,weight.shape[1])
        self.assertTrue(set(codes.reshape(-1).tolist())<={-1,0,1})
        self.assertTrue(torch.equal(torch.from_numpy(unpacked),torch.from_numpy(dequantized)))

    def test_int4_pack_round_trip_all_codes(self):
        weight=torch.arange(-7,9,dtype=torch.float32).reshape(1,-1)
        packed,scale,dequantized=int4_row_pack(weight)
        codes,unpacked=int4_row_unpack(packed,scale,weight.shape[1])
        self.assertTrue(bool(((codes>=-7)&(codes<=7)).all()))
        self.assertTrue(torch.equal(torch.from_numpy(unpacked),torch.from_numpy(dequantized)))


if __name__ == "__main__":
    unittest.main()
