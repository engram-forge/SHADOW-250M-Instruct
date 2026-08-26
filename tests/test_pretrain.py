import gzip
import json
from pathlib import Path
import tempfile
import unittest

import torch
import torch.nn as nn

from pretrain.data import DolmaPacker, EOS, ShadowTokenizer, split_shards
from pretrain.diagnostics import normalized_participation, quantization_gap
from pretrain.train import training_contract
from shadow_runtime.retriever import enc


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER = ROOT / "tokenizer/tokenizer.model"
REMAP = ROOT / "tokenizer/new2old.u32"


def write_shard(path, records):
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record) + "\n")


class TokenizerTest(unittest.TestCase):
    def test_matches_runtime_remapping(self):
        tokenizer = ShadowTokenizer(TOKENIZER, REMAP)
        for text in ("hello world", "café 中文", "emoji: 🐕"):
            self.assertEqual(tokenizer.encode(text), enc(text))

    def test_split_is_deterministic_and_disjoint(self):
        paths = [Path(f"part-{index}.json.gz") for index in range(5)]
        first = split_shards(paths, 1337)
        second = split_shards(reversed(paths), 1337)
        self.assertEqual(first, second)
        self.assertFalse(set(first[0]) & set(first[1]))
        self.assertEqual(len(first[1]), 1)


class DiagnosticsTest(unittest.TestCase):
    def test_normalized_participation_distinguishes_outlier(self):
        uniform = normalized_participation(torch.tensor([5.0, 5.0, 5.0, 5.0]))
        outlier = normalized_participation(torch.tensor([10.0, 0.1, 0.1, 0.1]))
        self.assertAlmostEqual(uniform, 1.0)
        self.assertAlmostEqual(outlier, 0.2501500, places=6)

    def test_quantization_gap_separates_weight_families(self):
        class FakeRVQ(nn.Module):
            def __init__(self, weight, quantized, group):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(weight, dtype=torch.float32))
                self._q = torch.tensor(quantized, dtype=torch.float32)
                self.g = group

        model = nn.Module()
        model.rvq = FakeRVQ([[1.0, 2.0]], [[1.0, 1.0]], 8)
        model.ternary = FakeRVQ([[0.1, 10.0]], [[0.0, 0.0]], 32)
        gap = quantization_gap(model, FakeRVQ)
        self.assertAlmostEqual(gap["rvq"]["nmse"], 0.2)
        self.assertEqual(gap["rvq"]["worst_name"], "rvq")
        self.assertGreater(gap["ternary"]["nmse"], 0.1)
        self.assertEqual(gap["ternary"]["worst_name"], "ternary")
        int4_gap=quantization_gap(model,FakeRVQ,"int4_row")
        self.assertIn("int4_row",int4_gap)

    def test_training_contract_rejects_quantizer_changes(self):
        class Args:
            ctx=8; micro_batch=1; accum=1; seed=7; ffn_act_qat=True
            ffn_weight_dtype="ternary"; ffn_act_warmup_tokens=100; amp_dtype="bf16"
            mtp_horizon=2; mtp_loss_weight=0.3
        first=training_contract(Args)
        Args.ffn_weight_dtype="int4_row"
        second=training_contract(Args)
        self.assertNotEqual(first,second)

    def test_training_contract_rejects_mtp_changes(self):
        class Args:
            ctx=8; micro_batch=1; accum=1; seed=7; ffn_act_qat=True
            ffn_weight_dtype="ternary"; ffn_act_warmup_tokens=100; amp_dtype="bf16"
            mtp_horizon=2; mtp_loss_weight=0.3
        first=training_contract(Args)
        Args.mtp_horizon=1
        self.assertNotEqual(first,training_contract(Args))
        Args.mtp_horizon=2; Args.mtp_loss_weight=0.1
        self.assertNotEqual(first,training_contract(Args))


class PackerTest(unittest.TestCase):
    def make_shards(self, directory):
        first = Path(directory) / "a.json.gz"
        second = Path(directory) / "b.json.gz"
        write_shard(first, [
            {"text": "alpha beta gamma delta " * 8},
            {"missing": "text"},
            {"text": "epsilon zeta eta theta " * 8},
        ])
        write_shard(second, [
            {"text": "one two three four " * 8},
            {"text": "five six seven eight " * 8},
        ])
        return [first, second]

    def packer(self, shards):
        return DolmaPacker(
            shards, TOKENIZER, REMAP, context=8, workers=0, chunk_docs=2, seed=7
        )

    def test_eos_and_bad_record_accounting(self):
        with tempfile.TemporaryDirectory() as directory:
            shards = self.make_shards(directory)
            packer = self.packer(shards)
            all_tokens = []
            try:
                while True:
                    window = packer.next_window()
                    all_tokens.extend(window[:-1])
            except StopIteration:
                pass
            self.assertGreaterEqual(all_tokens.count(EOS), 3)
            self.assertEqual(packer.bad_records, 1)
            packer.close()

    def test_resume_produces_identical_next_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            shards = self.make_shards(directory)
            uninterrupted = self.packer(shards)
            uninterrupted.next_window()
            state = uninterrupted.state_dict()
            uninterrupted.close()

            reference = self.packer(shards)
            reference.load_state_dict(state)
            expected = [reference.next_window(), reference.next_window()]
            reference.close()

            resumed = self.packer(shards)
            resumed.load_state_dict(state)
            actual = [resumed.next_window(), resumed.next_window()]
            self.assertEqual(actual, expected)
            resumed.close()


if __name__ == "__main__":
    unittest.main()
