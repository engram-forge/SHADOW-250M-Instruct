import random
import json
import tempfile
import unittest
from unittest import mock

import torch

from finetune.finetune import (
    build_ids, recovery_example, repeated_ngram_ratio, repeated_span_start,
    truncate_complete_turns, unlikelihood_pairs,
)
from shadow_runtime import Engine, repetition_metrics


class TrainingUtilitiesTest(unittest.TestCase):
    def test_separate_validation_file_is_not_mixed(self):
        train = {"messages": [{"role": "user", "content": "train question"}, {"role": "assistant", "content": "train answer"}]}
        val = {"messages": [{"role": "user", "content": "validation question"}, {"role": "assistant", "content": "validation answer"}]}
        with tempfile.TemporaryDirectory() as directory:
            train_path = f"{directory}/train.jsonl"; val_path = f"{directory}/val.jsonl"
            with open(train_path, "w") as stream: stream.write((json.dumps(train) + "\n") * 2)
            with open(val_path, "w") as stream: stream.write(json.dumps(val) + "\n")
            from finetune.finetune import Packer
            packer = Packer(train_path, 128, random.Random(0), 0.5, val_path=val_path)
            self.assertEqual(len(packer.train), 2); self.assertEqual(len(packer.val), 1)
    def test_adjacent_loop_detection(self):
        self.assertEqual(repeated_span_start([1, 2, 3] * 3), 0)
        self.assertIsNone(repeated_span_start([1, 2, 3, 1, 2, 4]))
        self.assertGreater(repeated_ngram_ratio([1, 2, 3, 4] * 4), 0.5)

    def test_truncation_ends_on_complete_assistant_turn(self):
        messages = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
            {"role": "assistant", "content": "second answer " * 30},
        ]
        first_turn_length = len(build_ids(messages[:2])[0])
        kept = truncate_complete_turns(messages, first_turn_length + 3)
        self.assertEqual(kept, messages[:2])
        ids, mask = build_ids(kept)
        self.assertEqual(len(ids), first_turn_length)
        self.assertTrue(any(mask))

    def test_recovery_prefix_is_not_supervised(self):
        ids = list(range(30)); mask = [0] * 5 + [1] * 25
        changed_ids, changed_mask = recovery_example(ids, mask, random.Random(0))
        injected = len(changed_ids) - len(ids)
        self.assertGreater(injected, 0)
        self.assertEqual(changed_mask[5:5 + injected], [0] * injected)
        self.assertEqual(changed_ids[5 + injected:], ids[5:])

    def test_ul_negative_completes_prior_trigram(self):
        # At position 4, context ends in 1,2; token 3 previously followed 1,2.
        x = torch.tensor([[1, 2, 3, 1, 2, 4]])
        y = torch.tensor([[-100, -100, -100, -100, 4, -100]])
        self.assertIn((4, 3), unlikelihood_pairs(x, y, window=64, ngram=3))
        # Never penalize the gold token.
        y[0, 4] = 3
        self.assertNotIn((4, 3), unlikelihood_pairs(x, y, window=64, ngram=3))


class RuntimeGuardTest(unittest.TestCase):
    def test_metrics_find_adjacent_loop(self):
        start, ratio = repetition_metrics([10, 11] + [1, 2, 3] * 3)
        self.assertEqual(start, 2)
        self.assertGreater(ratio, 0)

    def test_chat_retries_and_returns_metadata(self):
        engine = Engine.__new__(Engine)
        outputs = ["I have found that it works. " * 4, "a concise final answer"]
        with mock.patch.object(engine, "_gen", side_effect=outputs) as generate:
            result = engine.chat("test", return_metadata=True)
        self.assertEqual(generate.call_count, 2)
        self.assertTrue(result["retried"])
        self.assertEqual(result["text"], "a concise final answer")
        retry_extra = generate.call_args_list[1].args[2]
        self.assertIn("1.25", retry_extra)


if __name__ == "__main__":
    unittest.main()
