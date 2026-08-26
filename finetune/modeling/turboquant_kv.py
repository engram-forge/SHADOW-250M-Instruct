"""Clean-room TurboQuant-style KV experiment for SHADOW's 64-wide heads.

This module implements the published two-stage idea for keys (a spherical
Lloyd-Max quantizer plus a one-bit Gaussian residual sketch) and conventional
groupwise value quantization. It is an experiment, not a dependency on or copy
of the GPL-3.0 third-party implementation at 0xsero/turboquant.
"""

import math
from typing import NamedTuple

import torch
import torch.nn as nn


class QuantizedKeys(NamedTuple):
    polar_codes: torch.Tensor
    residual_signs: torch.Tensor
    norms: torch.Tensor
    residual_norms: torch.Tensor


class QuantizedValues(NamedTuple):
    codes: torch.Tensor
    scales: torch.Tensor
    minima: torch.Tensor


_CODEBOOK_CACHE = {}


def pack_bits(values, bits):
    if bits not in (1, 2, 4):
        raise ValueError("packed width must be 1, 2, or 4 bits")
    if values.ndim == 0:
        raise ValueError("values must have a packed final dimension")
    levels = (1 << bits) - 1
    if torch.is_floating_point(values) or values.is_complex():
        if not bool(torch.all(values == values.round())):
            raise ValueError("packed values must be integers")
    if not bool(torch.all((values >= 0) & (values <= levels))):
        raise ValueError(f"packed values must be in [0, {levels}]")
    per_byte = 8 // bits
    width = values.shape[-1]
    padding = (-width) % per_byte
    values = values.to(torch.uint8)
    if padding:
        values = torch.nn.functional.pad(values, (0, padding))
    grouped=values.to(torch.uint8).reshape(*values.shape[:-1],-1,per_byte)
    shifts=torch.arange(per_byte,device=values.device,dtype=torch.uint8)*bits
    return (grouped<<shifts).sum(-1,dtype=torch.uint8)


def unpack_bits(packed, bits, width):
    if bits not in (1, 2, 4):
        raise ValueError("packed width must be 1, 2, or 4 bits")
    if packed.ndim == 0 or width < 0:
        raise ValueError("packed data and width must describe a final dimension")
    per_byte=8//bits; mask=(1<<bits)-1
    if packed.shape[-1] * per_byte < width:
        raise ValueError("packed tensor is too short for the requested width")
    shifts=torch.arange(per_byte,device=packed.device,dtype=torch.uint8)*bits
    values=((packed.unsqueeze(-1)>>shifts)&mask).reshape(*packed.shape[:-1],-1)
    return values[...,:width]


def spherical_codebook(dim,bits,grid_size=32769,iterations=64):
    """Numerically solve the symmetric Lloyd-Max problem on a sphere coordinate."""
    if dim<3 or bits not in (1,2,3,4): raise ValueError("unsupported codebook geometry")
    key=(int(dim),int(bits),int(grid_size),int(iterations))
    if key in _CODEBOOK_CACHE: return tuple(item.clone() for item in _CODEBOOK_CACHE[key])
    grid=torch.linspace(-1+1e-6,1-1e-6,grid_size,dtype=torch.float64)
    log_weight=((dim-3)/2)*torch.log1p(-grid.square())
    weight=torch.exp(log_weight-log_weight.max())
    cdf=weight.cumsum(0)
    cdf=cdf/cdf[-1]
    levels=1<<bits
    probabilities=(torch.arange(levels,dtype=torch.float64)+0.5)/levels
    centroids=grid[torch.searchsorted(cdf,probabilities)].clone()
    for _ in range(iterations):
        boundaries=(centroids[:-1]+centroids[1:])/2
        assignment=torch.bucketize(grid,boundaries)
        numerator=torch.zeros(levels,dtype=torch.float64).scatter_add_(0,assignment,grid*weight)
        denominator=torch.zeros(levels,dtype=torch.float64).scatter_add_(0,assignment,weight)
        updated=numerator/denominator.clamp_min(1e-30)
        if torch.max(torch.abs(updated-centroids))<1e-12:
            centroids=updated; break
        centroids=updated
    # The coordinate density is exactly symmetric. Remove integration-grid skew so
    # an A55 implementation may store sign plus magnitude tables deterministically.
    centroids=(centroids-centroids.flip(0))/2
    boundaries=(centroids[:-1]+centroids[1:])/2
    result=(centroids.float(),boundaries.float())
    _CODEBOOK_CACHE[key]=result
    return tuple(item.clone() for item in result)


def seeded_matrices(dim, seed):
    generator=torch.Generator(device="cpu").manual_seed(int(seed))
    gaussian=torch.randn(dim,dim,generator=generator,dtype=torch.float32)
    rotation,upper=torch.linalg.qr(gaussian)
    signs=torch.where(torch.diag(upper)>=0,1.0,-1.0)
    rotation=rotation*signs.unsqueeze(0)
    sketch_generator=torch.Generator(device="cpu").manual_seed(int(seed)+1000)
    sketch=torch.randn(dim,dim,generator=sketch_generator,dtype=torch.float32)
    return rotation,sketch


class TurboQuantKeyCodec(nn.Module):
    """Three-bit key codec: two-bit Polar stage plus one-bit QJL correction."""
    def __init__(self,dim=64,total_bits=3,seed=42):
        super().__init__()
        if total_bits!=3: raise ValueError("the SHADOW experiment currently fixes keys at 3 bits")
        self.dim=int(dim); self.polar_bits=total_bits-1
        centroids,boundaries=spherical_codebook(self.dim,self.polar_bits)
        rotation,sketch=seeded_matrices(self.dim,seed)
        self.register_buffer("centroids",centroids)
        self.register_buffer("boundaries",boundaries)
        self.register_buffer("rotation",rotation)
        self.register_buffer("sketch",sketch)
        self.correction=math.sqrt(math.pi/2)/self.dim

    def _check(self, values, name):
        if values.ndim < 1 or values.shape[-1] != self.dim:
            raise ValueError(f"{name} must end in dimension {self.dim}, got {tuple(values.shape)}")

    def _polar_reconstruct(self,codes,norms):
        indices=unpack_bits(codes,self.polar_bits,self.dim).long()
        rotated=self.centroids[indices]
        return (rotated@self.rotation)*norms.float().unsqueeze(-1)

    @torch.no_grad()
    def pack(self,keys):
        self._check(keys, "keys")
        original=keys.float(); norms=original.norm(dim=-1)
        unit=original/norms.unsqueeze(-1).clamp_min(1e-12)
        rotated=unit@self.rotation.T
        indices=torch.bucketize(rotated.contiguous(),self.boundaries)
        codes=pack_bits(indices,self.polar_bits)
        polar=self._polar_reconstruct(codes,norms)
        residual=original-polar; residual_norms=residual.norm(dim=-1)
        signs=pack_bits((residual@self.sketch.T>=0).to(torch.uint8),1)
        return QuantizedKeys(codes,signs,norms.to(torch.float16),
                             residual_norms.to(torch.float16))

    def unpack(self,packed):
        polar=self._polar_reconstruct(packed.polar_codes,packed.norms)
        signs=unpack_bits(packed.residual_signs,1,self.dim).float().mul_(2).sub_(1)
        correction=(signs@self.sketch)*self.correction
        return polar+correction*packed.residual_norms.float().unsqueeze(-1)

    def scores(self,queries,packed):
        self._check(queries, "queries")
        if queries.ndim < 2 or packed.norms.ndim < 1:
            raise ValueError("attention scores require query and key sequence dimensions")
        polar=self._polar_reconstruct(packed.polar_codes,packed.norms)
        base=queries.float()@polar.transpose(-2,-1)
        query_sketch=queries.float()@self.sketch.T
        signs=unpack_bits(packed.residual_signs,1,self.dim).float().mul_(2).sub_(1)
        correction=query_sketch@signs.transpose(-2,-1)
        return base+correction*(self.correction*packed.residual_norms.float().unsqueeze(-2))


class GroupwiseValueCodec(nn.Module):
    def __init__(self,dim=64,bits=4,group_size=32):
        super().__init__()
        if bits not in (2,4) or dim%group_size: raise ValueError("invalid value geometry")
        self.dim=int(dim); self.bits=int(bits); self.group_size=int(group_size)

    @torch.no_grad()
    def pack(self,values):
        if values.ndim < 1 or values.shape[-1] != self.dim:
            raise ValueError(
                f"values must end in dimension {self.dim}, got {tuple(values.shape)}")
        shape=values.shape; groups=values.float().reshape(*shape[:-1],-1,self.group_size)
        minima=groups.amin(-1); maxima=groups.amax(-1); levels=(1<<self.bits)-1
        scales=((maxima-minima)/levels).clamp_min(1e-10)
        codes=((groups-minima.unsqueeze(-1))/scales.unsqueeze(-1)).round().clamp(0,levels)
        codes=pack_bits(codes.reshape(*shape[:-1],self.dim),self.bits)
        return QuantizedValues(codes,scales.to(torch.float16),minima.to(torch.float16))

    def unpack(self,packed):
        codes=unpack_bits(packed.codes,self.bits,self.dim).float()
        groups=codes.reshape(*codes.shape[:-1],-1,self.group_size)
        values=groups*packed.scales.float().unsqueeze(-1)+packed.minima.float().unsqueeze(-1)
        return values.reshape(*codes.shape[:-1],self.dim)


def tensor_bytes(structure):
    return sum(value.numel()*value.element_size() for value in structure if torch.is_tensor(value))


class TurboQuantKVCodec(nn.Module):
    def __init__(self,dim=64,key_bits=3,value_bits=4,group_size=32,seed=42):
        super().__init__(); self.dim=int(dim)
        self.keys=TurboQuantKeyCodec(dim,key_bits,seed)
        self.values=GroupwiseValueCodec(dim,value_bits,group_size)

    @torch.no_grad()
    def evaluate(self,queries,keys,values):
        if queries.ndim < 2 or keys.ndim < 2 or values.ndim < 2:
            raise ValueError("evaluation requires (..., sequence, head_dim) tensors")
        if keys.shape != values.shape:
            raise ValueError(f"key/value shapes differ: {tuple(keys.shape)} and {tuple(values.shape)}")
        if queries.shape[:-2] != keys.shape[:-2]:
            raise ValueError("query and key/value batch dimensions must match")
        packed_keys=self.keys.pack(keys); packed_values=self.values.pack(values)
        reconstructed_keys=self.keys.unpack(packed_keys); reconstructed_values=self.values.unpack(packed_values)
        exact_scores=queries.float()@keys.float().transpose(-2,-1)/math.sqrt(self.dim)
        scores=self.keys.scores(queries,packed_keys)/math.sqrt(self.dim)
        exact_weights=exact_scores.softmax(-1); weights=scores.softmax(-1)
        exact_output=exact_weights@values.float(); output=weights@reconstructed_values
        return {"packed_keys":packed_keys,"packed_values":packed_values,
                "keys":reconstructed_keys,"values":reconstructed_values,
                "scores":scores,"output":output,"exact_scores":exact_scores,
                "exact_output":exact_output,
                "bytes":tensor_bytes(packed_keys)+tensor_bytes(packed_values)}
