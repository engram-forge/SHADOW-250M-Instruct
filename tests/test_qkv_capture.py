import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch


ROOT=Path(__file__).resolve().parents[1]
PYTHON=sys.executable
TINY_ENV={"SHADOW_D":"64","SHADOW_NL":"1","SHADOW_NH":"1",
          "SHADOW_NKV":"1","SHADOW_HD":"64","SHADOW_FFNH":"128",
          "SHADOW_FAST_ATTN":"1","SHADOW_KV_BITS":"4",
          "SHADOW_KV_HOT_TOKENS":"2"}


class QKVCaptureTest(unittest.TestCase):
    def test_checkpoint_capture_and_codec_benchmark(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory); checkpoint=directory/"tiny.pt"; capture=directory/"qkv.npz"
            make="""
import numpy as np,torch
from model_250m import Shadow250M
fp=np.unpackbits(np.load(r'%s'),axis=1)[:,:512]
cent=torch.tensor(fp.astype(np.float32)*2-1)
model=Shadow250M(cent,torch.nn.functional.normalize(cent,dim=-1),len(cent),mtp_horizon=2)
torch.save({'model':model.state_dict(),'cfg':{'V':len(cent),'mtp_horizon':2,
'ffn_weight_dtype':'ternary','ffn_act_qat':True,'kv_format':'int4','kv_hot_tokens':2}},r'%s')
"""%(ROOT/"deployment/fp131072.npy",checkpoint)
            env={**os.environ,**TINY_ENV,"PYTHONPATH":str(ROOT/"finetune"/"modeling")}
            subprocess.run([PYTHON,"-c",make],cwd=ROOT,env=env,check=True,
                           stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            result=subprocess.run([PYTHON,"benchmarks/capture_qkv.py","--checkpoint",str(checkpoint),
                "--out",str(capture),"--prompt","alpha beta gamma","--device","cpu"],
                cwd=ROOT,env=env,check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            self.assertIn("wrote",result.stdout)
            arrays=np.load(capture)
            self.assertEqual(arrays["queries"].shape[-1],64)
            self.assertEqual(arrays["keys"].shape,arrays["values"].shape)
            result=subprocess.run([PYTHON,"benchmarks/turboquant_kv.py","--capture",str(capture),
                "--repeats","1"],cwd=ROOT,env=env,check=True,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            self.assertIn('"a55_int4_kv"',result.stdout)
            self.assertIn('"turboquant_k3_v4"',result.stdout)


if __name__=="__main__": unittest.main()
