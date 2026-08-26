import gzip
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F

import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"finetune"/"modeling"))
from weight_formats import (int4_row_pack,int4_row_unpack,ternary_pack,
                            ternary_unpack_compact)
from validate_export import validate_export


PYTHON=Path(sys.executable)
TINY_ENV={"SHADOW_D":"64","SHADOW_NL":"1","SHADOW_NH":"1",
          "SHADOW_NKV":"1","SHADOW_HD":"64","SHADOW_FFNH":"128",
          "SHADOW_FAST_ATTN":"1","SHADOW_KV_BITS":"1","SHADOW_KV_TWO_TIER":"1"}


def run(command,env=None):
    merged={**os.environ,**TINY_ENV,**(env or {})}
    result=subprocess.run([str(PYTHON),*command],cwd=ROOT,env=merged,text=True,
                          stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    if result.returncode: raise subprocess.CalledProcessError(
        result.returncode,result.args,output=result.stdout)
    return result


def write_conversations(path):
    rows=[
        {"messages":[{"role":"user","content":"Say alpha."},{"role":"assistant","content":"Alpha."}]},
        {"messages":[{"role":"user","content":"Say beta."},{"role":"assistant","content":"Beta."}]},
        {"messages":[{"role":"user","content":"Say gamma."},{"role":"assistant","content":"Gamma."}]},
    ]
    path.write_text("".join(json.dumps(row)+"\n" for row in rows))


def write_dolma(path,text):
    with gzip.open(path,"wt",encoding="utf-8") as stream:
        for index in range(8): stream.write(json.dumps({"text":f"{text} document {index}. "*8})+"\n")


def make_checkpoint(path,weight_dtype,mtp_horizon=2):
    code=f"""
import sys,numpy as np,torch
sys.path[:0]=['{ROOT/'finetune'}','{ROOT/'finetune/modeling'}']
from model_250m import Shadow250M
fp=np.unpackbits(np.load('{ROOT/'deployment/fp131072.npy'}'),axis=1)[:,:512]
cent=torch.tensor(fp.astype(np.float32)*2-1)
m=Shadow250M(cent,torch.nn.functional.normalize(cent,dim=-1),len(cent),mtp_horizon={mtp_horizon})
torch.save({{'model':m.state_dict(),'cfg':{{'V':len(cent),'ffn_weight_dtype':'{weight_dtype}',
'ffn_act_qat':True,'ffn_act_bits':8,'ffn_act_scale':'per_token_power_of_two',
'ffn_weight_scale':'per_output_row_fp32','ffn_accumulator':'int32',
'mtp_horizon':{mtp_horizon},'mtp_loss_weight':0.3}}}},'{path}')
"""
    run(["-c",code])


def records(path):
    result=[]
    with path.open("rb") as stream:
        if stream.read(4)!=b"SHDW": raise ValueError("magic")
        _,count=struct.unpack("<II",stream.read(8))
        for _ in range(count):
            length,=struct.unpack("<I",stream.read(4)); name=stream.read(length).decode()
            kind,=struct.unpack("<I",stream.read(4)); shape=None
            if kind in (0,5):
                rank,=struct.unpack("<I",stream.read(4)); dims=struct.unpack("<"+"I"*rank,stream.read(4*rank))
                shape=dims
                stream.seek(int(np.prod(dims))*(4 if kind==0 else 2),1)
            elif kind==1:
                output,input_size,group,stages=struct.unpack("<IIII",stream.read(16)); groups=input_size//group; padded=(output+63)&~63
                stream.seek(stages*group*16*4+stages*(padded//64)*groups*32+padded*4,1)
            elif kind in (3,4,6):
                output,input_size=struct.unpack("<II",stream.read(8))
                shape=(output,input_size)
                size=output*(input_size//4 if kind==3 else (input_size+4)//5 if kind==4 else (input_size+1)//2)
                stream.seek(size+output*4,1)
            else: raise ValueError(kind)
            result.append((name,kind,shape))
    return result


def exported_mtp_parameters(path):
    result={}
    wanted={"mtp.norm.w","mtp.down","mtp.up"}
    with path.open("rb") as stream:
        if stream.read(4)!=b"SHDW": raise ValueError("magic")
        _,count=struct.unpack("<II",stream.read(8))
        for _ in range(count):
            length,=struct.unpack("<I",stream.read(4)); name=stream.read(length).decode()
            kind,=struct.unpack("<I",stream.read(4))
            if kind in (0,5):
                rank,=struct.unpack("<I",stream.read(4)); shape=struct.unpack("<"+"I"*rank,stream.read(4*rank))
                dtype=np.float32 if kind==0 else np.float16
                value=np.frombuffer(stream.read(int(np.prod(shape))*np.dtype(dtype).itemsize),dtype).reshape(shape).copy()
                if name in wanted: result[name]=value.astype(np.float32)
            elif kind==1:
                output,input_size,group,stages=struct.unpack("<IIII",stream.read(16)); groups=input_size//group; padded=(output+63)&~63
                stream.seek(stages*group*16*4+stages*(padded//64)*groups*32+padded*4,1)
            elif kind in (3,4,6):
                output,input_size=struct.unpack("<II",stream.read(8))
                columns=input_size//4 if kind==3 else (input_size+4)//5 if kind==4 else (input_size+1)//2
                packed=np.frombuffer(stream.read(output*columns),np.uint8).reshape(output,columns).copy()
                scale=np.frombuffer(stream.read(output*4),np.float32).copy()
                if name in wanted:
                    if kind==4: _,value=ternary_unpack_compact(packed,scale,input_size)
                    elif kind==6: _,value=int4_row_unpack(packed,scale,input_size)
                    else: raise AssertionError("MTP export should use compact release packing")
                    result[name]=value
            else: raise ValueError(kind)
    return result


def int8_pot(value):
    maximum=value.abs().amax(-1,keepdim=True).clamp_min(1e-12)
    scale=torch.exp2(torch.ceil(torch.log2(maximum/127.0)))
    return (value/scale).round().clamp(-127,127)*scale


def mtp_mlp(state,embedding,norm,down,up):
    mixed=state+embedding
    mixed=mixed*torch.rsqrt(mixed.square().mean(-1,keepdim=True)+1e-6)*norm
    hidden=F.silu(F.linear(int8_pot(mixed),down))
    return state+F.linear(int8_pot(hidden),up)


class TrainingExportIntegrationTest(unittest.TestCase):
    def test_finetune_and_both_exports_on_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory); data=directory/"chat.jsonl"; write_conversations(data)
            subprocess.run(["make","-C","kernels/a55","clean","all"],cwd=ROOT,check=True,
                           stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            try:
                for dtype,expected_kind in (("ternary",4),("int4_row",6)):
                    source=directory/f"{dtype}.pt"; make_checkpoint(source,dtype)
                    output=directory/f"tuned-{dtype}"
                    run(["finetune/finetune.py","--data",str(data),"--init",str(source),
                         "--out",str(output),"--device","cpu","--steps","1","--ctx","32",
                         "--micro-batch","1","--accum","1","--warmup","1",
                         "--ffn-act-warmup-steps","1","--val-batches","1","--ul-alpha","0",
                         "--recovery-ratio","0","--amp-dtype","fp32"])
                    checkpoint=torch.load(output/"finetuned.pt",map_location="cpu",weights_only=False)
                    self.assertEqual(checkpoint["cfg"]["ffn_weight_dtype"],dtype)
                    self.assertEqual(checkpoint["cfg"]["mtp_horizon"],2)
                    self.assertIn("mtp.down.weight",checkpoint["model"]); self.assertIn("mtp.up.weight",checkpoint["model"])
                    evaluated=run(["finetune/evaluate_loss.py","--checkpoint",str(output/"finetuned.pt"),
                                   "--data",str(data),"--ctx","32","--batches","1",
                                   "--micro-batch","1","--evaluate-mtp"])
                    self.assertIn("mtp_loss=",evaluated.stdout); self.assertIn("mtp_accuracy=",evaluated.stdout)
                    shdw=directory/f"{dtype}.shdw"
                    run(["finetune/export_model.py",str(output/"finetuned.pt"),str(shdw)])
                    exported=records(shdw)
                    self.assertIn(expected_kind,[kind for _,kind,_ in exported])
                    by_name={name:(kind,shape) for name,kind,shape in exported}
                    self.assertEqual(by_name["mtp.down"],(expected_kind,(32,64)))
                    self.assertEqual(by_name["mtp.up"],(expected_kind,(64,32)))
                    self.assertEqual(by_name["mtp.norm.w"],(0,(64,)))
                    manifest=json.loads(Path(str(shdw)+".a55.json").read_text())
                    validate_export(shdw,manifest,hidden_size=64)
                    self.assertEqual(manifest["architecture_version"],2)
                    self.assertEqual(manifest["ffn_weight"]["dtype"],dtype)
                    self.assertEqual(manifest["mtp"]["horizon"],2)
                    self.assertEqual(manifest["mtp"]["auxiliary_heads"],1)
                    self.assertFalse(manifest["mtp"]["deepseek_exact"])
                    self.assertFalse(manifest["compatible_with_bundled_engine"])
                    exported_params=exported_mtp_parameters(shdw)
                    pack=(lambda value:ternary_pack(value,compact=True)[2]) if dtype=="ternary" else (lambda value:int4_row_pack(value)[2])
                    expected_down=pack(checkpoint["model"]["mtp.down.weight"]); expected_up=pack(checkpoint["model"]["mtp.up.weight"])
                    np.testing.assert_array_equal(exported_params["mtp.down"],expected_down)
                    np.testing.assert_array_equal(exported_params["mtp.up"],expected_up)
                    state=torch.randn(2,3,64); embedding=torch.randn(2,3,64)
                    expected=mtp_mlp(state,embedding,checkpoint["model"]["mtp.norm.w"],
                                     torch.from_numpy(expected_down),torch.from_numpy(expected_up))
                    reconstructed=mtp_mlp(state,embedding,torch.from_numpy(exported_params["mtp.norm.w"]),
                                          torch.from_numpy(exported_params["mtp.down"]),
                                          torch.from_numpy(exported_params["mtp.up"]))
                    self.assertTrue(torch.equal(expected,reconstructed))
                    for option in ([],["--nibble"]):
                        result=subprocess.run([str(ROOT/"kernels/a55/ternary_dotprod"),*option,
                            "--shdw",str(shdw),"b.0.up","1"],cwd=ROOT,text=True,
                            stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                        self.assertEqual(result.returncode,0,result.stdout)
            finally:
                subprocess.run(["make","-C","kernels/a55","clean"],cwd=ROOT,check=True,
                               stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)

    def test_cpu_pretrain_checkpoint_and_resume_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory); data=directory/"dolma"; data.mkdir()
            write_dolma(data/"a.json.gz","alpha"); write_dolma(data/"b.json.gz","beta")
            output=directory/"run"
            command=["pretrain/train.py","train","--data",str(data),"--out",str(output),
                     "--device","cpu","--amp-dtype","fp32","--ctx","16","--micro-batch","1",
                     "--accum","1","--workers","0","--chunk-docs","2","--max-tokens","16",
                     "--val-every","1000","--checkpoint-every","16","--diagnostics-every","0",
                     "--ffn-weight-dtype","int4_row","--ffn-act-warmup-tokens","16",
                     "--mtp-horizon","2","--mtp-loss-weight","0.3"]
            run(command)
            final=output/"checkpoints/final.pt"
            saved=torch.load(final,map_location="cpu",weights_only=False)
            self.assertEqual(saved["cfg"]["ffn_weight_dtype"],"int4_row")
            self.assertEqual(saved["cfg"]["mtp_horizon"],2)
            self.assertIn("mtp.down.weight",saved["model"]); self.assertIn("mtp.up.weight",saved["model"])
            self.assertIn("scaler",saved); self.assertEqual(saved["consumed_tokens"],16)
            resumed=command.copy(); resumed[resumed.index("--max-tokens")+1]="32"
            resumed.extend(["--resume",str(final)])
            run(resumed)
            resumed_state=torch.load(final,map_location="cpu",weights_only=False)
            self.assertEqual(resumed_state["consumed_tokens"],32)
            bad=resumed.copy(); bad.extend(["--ffn-weight-dtype","ternary"])
            with self.assertRaises(subprocess.CalledProcessError): run(bad)
            bad_mtp=resumed.copy(); bad_mtp.extend(["--mtp-horizon","1"])
            with self.assertRaises(subprocess.CalledProcessError): run(bad_mtp)

    def test_legacy_horizon_one_checkpoint_exports_without_mtp_records(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory); source=directory/"legacy.pt"
            make_checkpoint(source,"ternary",mtp_horizon=1)
            checkpoint=torch.load(source,map_location="cpu",weights_only=False)
            checkpoint["cfg"].pop("mtp_horizon"); checkpoint["cfg"].pop("mtp_loss_weight")
            torch.save(checkpoint,source)
            shdw=directory/"legacy.shdw"
            run(["finetune/export_model.py",str(source),str(shdw)])
            self.assertFalse(any(name.startswith("mtp.") for name,_,_ in records(shdw)))
            manifest=json.loads(Path(str(shdw)+".a55.json").read_text())
            self.assertEqual(manifest["mtp"]["horizon"],1)
            validate_export(shdw,manifest,hidden_size=64)

    def test_legacy_checkpoint_upgrade_warm_starts_new_pretrain_run(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory); legacy=directory/"legacy.pt"; upgraded=directory/"k2.pt"
            make_checkpoint(legacy,"ternary",mtp_horizon=1)
            run(["pretrain/upgrade_checkpoint.py","--input",str(legacy),"--output",str(upgraded)])
            warm=torch.load(upgraded,map_location="cpu",weights_only=False)
            self.assertEqual(warm["checkpoint_type"],"model_only_warm_start")
            self.assertNotIn("optimizer",warm); self.assertEqual(warm["cfg"]["mtp_horizon"],2)
            data=directory/"dolma"; data.mkdir(); write_dolma(data/"a.json.gz","alpha"); write_dolma(data/"b.json.gz","beta")
            output=directory/"run"
            run(["pretrain/train.py","train","--data",str(data),"--out",str(output),
                 "--init-checkpoint",str(upgraded),"--device","cpu","--amp-dtype","fp32",
                 "--ctx","8","--micro-batch","1","--accum","1","--workers","0",
                 "--chunk-docs","1","--max-tokens","8","--val-every","1000",
                 "--checkpoint-every","8","--diagnostics-every","0","--mtp-horizon","2",
                 "--ffn-weight-dtype","ternary"])
            final=torch.load(output/"checkpoints/final.pt",map_location="cpu",weights_only=False)
            self.assertIn("optimizer",final); self.assertEqual(final["consumed_tokens"],8)


if __name__=="__main__": unittest.main()
