"""Packing contracts shared by QAT export tests and SHDW writers."""

import numpy as np
import torch


def ternary_pack(weight,compact=False):
    values=weight.detach().float()
    scale=values.abs().mean(1,keepdim=True).clamp_min(1e-5)
    symbols=(values/scale).round().clamp(-1,1).to(torch.int8).cpu().numpy()
    row_scale=scale[:,0].cpu().numpy().astype(np.float32)
    codes=(symbols+1).astype(np.uint8); output,input_size=codes.shape
    if compact:
        padding=(-input_size)%5
        padded=np.concatenate([codes,np.ones((output,padding),np.uint8)],1) if padding else codes
        groups=padded.reshape(output,-1,5).astype(np.uint16)
        packed=(groups[:,:,0]+3*groups[:,:,1]+9*groups[:,:,2]
                +27*groups[:,:,3]+81*groups[:,:,4]).astype(np.uint8)
    else:
        if input_size%4: raise ValueError("2-bit ternary packing requires rows divisible by 4")
        groups=codes.reshape(output,-1,4)
        packed=(groups[:,:,0]|(groups[:,:,1]<<2)|(groups[:,:,2]<<4)
                |(groups[:,:,3]<<6)).astype(np.uint8)
    return packed,row_scale,symbols.astype(np.float32)*row_scale[:,None]


def ternary_unpack_compact(packed,row_scale,input_size):
    packed=np.asarray(packed,dtype=np.uint8); output=packed.shape[0]
    codes=np.empty((output,packed.shape[1]*5),dtype=np.uint8)
    value=packed.astype(np.uint16)
    for position in range(5):
        codes[:,position::5]=value%3; value//=3
    symbols=codes[:,:input_size].astype(np.int8)-1
    return symbols,symbols.astype(np.float32)*np.asarray(row_scale,np.float32)[:,None]


def int4_row_pack(weight):
    values=weight.detach().float()
    scale=(values.abs().amax(1,keepdim=True)/7.0).clamp_min(1e-8)
    codes=(values/scale).round().clamp(-7,7).to(torch.int8).cpu().numpy()
    output,input_size=codes.shape
    if input_size%2: raise ValueError("INT4 packing requires even row width")
    unsigned=(codes.astype(np.int16)&15).astype(np.uint8).reshape(output,-1,2)
    packed=(unsigned[:,:,0]|(unsigned[:,:,1]<<4)).astype(np.uint8)
    row_scale=scale[:,0].cpu().numpy().astype(np.float32)
    return packed,row_scale,codes.astype(np.float32)*row_scale[:,None]


def int4_row_unpack(packed,row_scale,input_size):
    packed=np.asarray(packed,dtype=np.uint8)
    codes=np.stack((packed&15,packed>>4),-1).reshape(packed.shape[0],-1)[:,:input_size]
    signed=codes.astype(np.int8); signed[signed>=8]-=16
    return signed,signed.astype(np.float32)*np.asarray(row_scale,np.float32)[:,None]
