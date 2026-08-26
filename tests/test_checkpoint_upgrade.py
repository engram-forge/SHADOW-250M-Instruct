import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"pretrain")); sys.path.insert(0,str(ROOT/"finetune"/"modeling"))
import model_250m
import upgrade_checkpoint


class CheckpointUpgradeTest(unittest.TestCase):
    def test_legacy_checkpoint_becomes_model_only_k2_warm_start(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"old.pt"; output=Path(directory)/"new.pt"
            torch.save({"model":{"backbone.weight":torch.ones(2,2)},
                        "cfg":{"mtp_horizon":1,"ffn_weight_dtype":"ternary"},
                        "optimizer":{"must":"not survive"}},source)
            with mock.patch.object(upgrade_checkpoint,"D",4):
                payload=upgrade_checkpoint.upgrade(source,output,seed=9)
            loaded=torch.load(output,map_location="cpu",weights_only=False)
            self.assertEqual(payload["checkpoint_type"],"model_only_warm_start")
            self.assertEqual(loaded["cfg"]["mtp_horizon"],2)
            self.assertIn("backbone.weight",loaded["model"])
            self.assertIn("mtp.up.weight",loaded["model"]); self.assertIn("mtp.down.weight",loaded["model"])
            self.assertNotIn("optimizer",loaded)
            self.assertFalse(loaded["provenance"]["optimizer_state_reused"])

    def test_refuses_already_mtp_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/"mtp.pt"
            torch.save({"model":{"mtp.up.weight":torch.ones(1)},
                        "cfg":{"mtp_horizon":2}},source)
            with self.assertRaises(ValueError):
                upgrade_checkpoint.upgrade(source,Path(directory)/"out.pt")


if __name__=="__main__": unittest.main()
