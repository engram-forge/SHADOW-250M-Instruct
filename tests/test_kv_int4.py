import sys
import unittest
from pathlib import Path

import torch
from unittest import mock


ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"finetune"/"modeling"))
import common


class A55Int4KVTest(unittest.TestCase):
    def test_key_fake_quant_matches_fp16_metadata_pack(self):
        torch.manual_seed(1)
        values=torch.randn(2,3,7,64)
        packed,scale=common.kv4_key_pack(values)
        actual=common.kv4_key_unpack(packed,scale,values.dtype)
        # Runtime metadata is FP16, so compare against the same rounded scale.
        codes=common.kv4_key_values(values)[0]
        expected=codes.to(values.dtype)*scale.to(values.dtype)
        self.assertTrue(torch.equal(actual,expected))
        self.assertEqual(packed.shape[-1],32)
        self.assertEqual(scale.shape,values.shape[:-1]+(1,))

    def test_value_fake_quant_matches_fp16_metadata_pack(self):
        torch.manual_seed(2)
        values=torch.randn(2,3,7,64)
        packed,scale,minimum=common.kv4_value_pack(values)
        actual=common.kv4_value_unpack(packed,scale,minimum,dtype=values.dtype)
        codes=common.kv4_value_values(values)[0].reshape(*values.shape[:-1],2,32)
        expected=(codes.float()*scale.float().unsqueeze(-1)+minimum.float().unsqueeze(-1))
        self.assertTrue(torch.equal(actual,expected.reshape_as(values)))
        self.assertEqual(packed.shape[-1],32)
        self.assertEqual(scale.shape,values.shape[:-1]+(2,))
        self.assertEqual(minimum.shape,scale.shape)

    def test_key_codes_are_signed_nibbles_and_value_codes_are_unsigned(self):
        values=torch.tensor([[-100.0,-8.0,-1.0,0.0,1.0,7.0,8.0,100.0]])
        key_code,_,_=common.kv4_key_values(values)
        value_code,_,_,_=common.kv4_value_values(values,group_size=8)
        self.assertTrue(bool(((key_code>=-7)&(key_code<=7)).all()))
        self.assertTrue(bool(((value_code>=0)&(value_code<=15)).all()))

    def test_ste_preserves_gradient(self):
        key=torch.randn(2,64,requires_grad=True)
        value=torch.randn(2,64,requires_grad=True)
        (common.kv4_key(key).sum()+common.kv4_value(value).sum()).backward()
        self.assertTrue(torch.equal(key.grad,torch.ones_like(key)))
        self.assertTrue(torch.equal(value.grad,torch.ones_like(value)))

    def test_zero_and_nonfinite_inputs_reconstruct_finite(self):
        values=torch.zeros(2,64)
        values[0,0]=float("inf"); values[0,1]=float("nan")
        key=common.kv4_key_unpack(*common.kv4_key_pack(values),dtype=torch.float32)
        packed,scale,minimum=common.kv4_value_pack(values)
        value=common.kv4_value_unpack(packed,scale,minimum,dtype=torch.float32)
        self.assertTrue(bool(torch.isfinite(key).all()))
        self.assertTrue(bool(torch.isfinite(value).all()))

    def test_exact_storage_is_74_bytes_per_token_head(self):
        values=torch.randn(2,3,7,64)
        key=common.kv4_key_pack(values)
        value=common.kv4_value_pack(values)
        vectors=2*3*7
        size=sum(x.numel()*x.element_size() for x in (*key,*value))
        self.assertEqual(size/vectors,74)

    def test_set_kv_format_contract(self):
        common.set_kv_format("int4",64)
        self.assertEqual(common.KV_BITS,4)
        self.assertEqual(common.KV_HOT_TOKENS,64)
        with self.assertRaises(ValueError): common.set_kv_format("int3",64)
        with self.assertRaises(ValueError): common.set_kv_format("int4",-1)
        common.set_kv_format("1bit",128)

    def test_cached_int4_keeps_recent_tokens_exact_and_flushes_old_tokens(self):
        common.set_kv_format("int4",2)
        with mock.patch.multiple(common,D=64,NH=1,NKV=1,HD=64,FFNH=128):
            block=common.Block(0)
        def qkv(z,cos,sin):
            shaped=z.reshape(z.shape[0],z.shape[1],1,64).transpose(1,2)
            return shaped,shaped+0.25,shaped-0.5
        block._qkv=qkv
        block._finish_attention=lambda x,q,k,v,causal,exact=True:x
        first=torch.randn(1,4,64)
        expected_first=block.n1(first)
        _,cache=block.prefill_cached(first,None,None,max_ctx=4)
        self.assertEqual(cache["k"].shape[2],2)
        self.assertEqual(cache["hot_k"].shape[2],2)
        self.assertTrue(torch.equal(cache["hot_k"],(expected_first[:,-2:]+0.25).unsqueeze(1)))
        new=torch.randn(1,1,64)
        expected_new=block.n1(new)
        _,cache=block.decode_cached(new,4,cache,max_ctx=4)
        self.assertEqual(cache["k"].shape[2],2)
        self.assertEqual(cache["hot_k"].shape[2],2)
        self.assertTrue(torch.equal(cache["hot_k"][:,:,-1],expected_new[:,0].unsqueeze(1)+0.25))
        common.set_kv_format("1bit",128)


if __name__=="__main__": unittest.main()
