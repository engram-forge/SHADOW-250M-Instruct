import sys
import unittest
from pathlib import Path

import torch


ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"finetune"/"modeling"))
from mtp_reference import AcceptanceMetrics,reference_generate,reference_k2_step


class FakeGreedyModel:
    mtp_horizon=2

    def __init__(self,correct_second=True,vocab=8):
        self.mtp=object(); self.correct_second=correct_second; self.vocab=vocab

    def trunk(self,context):
        return context.float().unsqueeze(-1),None

    def logits(self,hidden):
        token=(hidden.squeeze(-1).long()+1)%self.vocab
        logits=torch.full((*token.shape,self.vocab),-100.0)
        return logits.scatter(-1,token.unsqueeze(-1),100.0)

    def mtp_logits(self,hidden,previous):
        offset=1 if self.correct_second else 2
        token=(previous+offset)%self.vocab
        logits=torch.full((*token.shape,self.vocab),-100.0)
        return logits.scatter(-1,token.unsqueeze(-1),100.0)

    def __call__(self,context):
        return self.logits(self.trunk(context)[0])


class ReferenceDecoderTest(unittest.TestCase):
    def test_exact_k2_accepts_both_matching_candidates(self):
        step=reference_k2_step(FakeGreedyModel(True),torch.tensor([[2,3]]))
        self.assertEqual(step.candidate_first.tolist(),[4])
        self.assertEqual(step.candidate_second.tolist(),[5])
        self.assertEqual(step.reference_first.tolist(),[4])
        self.assertEqual(step.reference_second.tolist(),[5])
        self.assertTrue(bool(step.first_accepted.all()))
        self.assertTrue(bool(step.second_accepted.all()))

    def test_second_rejection_does_not_reject_first(self):
        step=reference_k2_step(FakeGreedyModel(False),torch.tensor([[3]]))
        self.assertTrue(bool(step.first_accepted.all()))
        self.assertFalse(bool(step.second_accepted.any()))
        metrics=AcceptanceMetrics(); metrics.update(step)
        result=metrics.as_dict()
        self.assertEqual(result["first_acceptance_rate"],1.0)
        self.assertEqual(result["second_acceptance_rate"],0.0)
        self.assertEqual(result["mean_accepted_draft_tokens"],1.0)

    def test_generation_keeps_ordinary_sequential_oracle_context(self):
        output,metrics,steps=reference_generate(
            FakeGreedyModel(False),torch.tensor([[1]]),cycles=2)
        self.assertEqual(output.tolist(),[[1,2,3,4,5]])
        self.assertEqual(len(steps),2)
        self.assertEqual(metrics["first_acceptance_rate"],1.0)
        self.assertEqual(metrics["second_acceptance_rate"],0.0)

    def test_requires_horizon_two_and_nonempty_context(self):
        model=FakeGreedyModel(); model.mtp_horizon=1
        with self.assertRaises(ValueError):
            reference_k2_step(model,torch.tensor([[1]]))
        model.mtp_horizon=2
        with self.assertRaises(ValueError):
            reference_k2_step(model,torch.empty((1,0),dtype=torch.long))


if __name__=="__main__": unittest.main()
