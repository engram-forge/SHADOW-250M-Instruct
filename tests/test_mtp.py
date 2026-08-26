import sys
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"finetune"/"modeling"))
import model_250m
from model_250m import Shadow250M,mtp_slices


class PassStruct(nn.Module):
    def forward(self,x): return x,None


def tiny_model(horizon):
    cent=torch.zeros(4,512); cent[0,0]=1; cent[1,1]=1; cent[2,2]=1; cent[3,0]=-1
    with mock.patch.multiple(model_250m,D=4,NL=0,
                             StructStep=PassStruct,RMS=lambda _:nn.Identity()):
        return Shadow250M(cent,F.normalize(cent,dim=-1),len(cent),
                          mtp_horizon=horizon)


class MultiTokenPredictionTest(unittest.TestCase):
    def test_horizon_is_limited_to_two(self):
        hidden=torch.zeros(1,2,4); targets=torch.zeros(1,2,dtype=torch.long)
        for horizon in (0,3):
            with self.assertRaises(ValueError): mtp_slices(hidden,targets,horizon)
            with self.assertRaises(ValueError): tiny_model(horizon)

    def test_alignment_for_conditional_second_token(self):
        hidden=torch.arange(20).reshape(1,5,4)
        targets=torch.tensor([[10,11,12,13,14]])
        pairs=mtp_slices(hidden,targets,2)
        self.assertEqual(len(pairs),2)
        self.assertTrue(torch.equal(pairs[0][0],hidden))
        state,previous,target=pairs[1]
        self.assertEqual(state.shape[1],4)
        self.assertEqual(previous.tolist(),[[10,11,12,13]])
        self.assertEqual(target.tolist(),[[11,12,13,14]])

    def test_horizon_two_has_one_residual_mlp_and_shared_head(self):
        model=tiny_model(2)
        self.assertIsNotNone(model.mtp)
        self.assertEqual(tuple(model.mtp.down.weight.shape),(2,4))
        self.assertEqual(tuple(model.mtp.up.weight.shape),(4,2))
        logits=model.mtp_logits(torch.randn(1,1,4),torch.tensor([[1]]))
        self.assertEqual(tuple(logits.shape),(1,1,4))
        self.assertLess(float(model.mtp.up.weight.detach().std()),0.01)

    def test_weighted_loss_matches_offset_losses_and_trains_mtp(self):
        torch.manual_seed(7); model=tiny_model(2)
        hidden=torch.randn(2,5,4,requires_grad=True)
        targets=torch.tensor([[0,1,2,3,0],[3,2,1,0,-1]])
        conditioning=torch.tensor([[1,2,3,0],[2,1,0,3]])
        offset_losses=model.language_model_losses(hidden,targets,conditioning,chunk=3)
        expected=offset_losses[0]+0.3*offset_losses[1]
        actual=model.language_model_loss(hidden,targets,mtp_loss_weight=0.3,chunk=3,
                                         conditioning_ids=conditioning)
        self.assertTrue(torch.allclose(actual,expected))
        actual.backward()
        self.assertIsNotNone(model.mtp.down.weight.grad)
        self.assertIsNotNone(model.mtp.up.weight.grad)
        self.assertTrue(bool((model.mtp.down.weight.grad!=0).any()))
        self.assertTrue(bool((model.mtp.up.weight.grad!=0).any()))

    def test_metrics_report_losses_accuracy_and_token_counts(self):
        model=tiny_model(2); hidden=torch.randn(1,4,4)
        targets=torch.tensor([[0,1,2,-1]]); conditioning=torch.tensor([[1,2,3]])
        metrics=model.language_model_metrics(hidden,targets,0.3,chunk=2,
                                              conditioning_ids=conditioning)
        self.assertEqual(metrics["base_tokens"],3)
        self.assertEqual(metrics["mtp_tokens"],2)
        self.assertTrue(0<=float(metrics["base_accuracy"])<=1)
        self.assertTrue(0<=float(metrics["mtp_accuracy"])<=1)
        self.assertTrue(torch.allclose(metrics["loss"],
                                      metrics["base_loss"]+0.3*metrics["mtp_loss"]))

    def test_horizon_one_matches_main_head_loss(self):
        torch.manual_seed(11); model=tiny_model(1)
        hidden=torch.randn(1,4,4); targets=torch.tensor([[0,1,-1,2]])
        expected=model._vocab_loss(hidden,targets,model.head,chunk=2)
        actual=model.language_model_loss(hidden,targets,mtp_loss_weight=0.3,chunk=2)
        self.assertTrue(torch.equal(actual,expected))

    def test_short_sequences_only_use_available_offsets(self):
        model=tiny_model(2); hidden=torch.randn(1,1,4); targets=torch.tensor([[1]])
        self.assertEqual(len(model.language_model_losses(hidden,targets)),1)

    def test_unmasked_input_conditions_masked_finetune_targets(self):
        model=tiny_model(2); hidden=torch.randn(1,3,4)
        targets=torch.tensor([[-1,1,2]]); conditioning=torch.tensor([[3,1]])
        losses=model.language_model_losses(hidden,targets,conditioning,chunk=2)
        self.assertEqual(len(losses),2)
        self.assertTrue(all(bool(torch.isfinite(loss)) for loss in losses))


if __name__=="__main__": unittest.main()
