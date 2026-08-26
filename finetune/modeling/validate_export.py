"""Structural release checks for A55 SHDW exports and their manifests."""

import json
from pathlib import Path
import struct

import numpy as np


def _read_exact(stream,size,description):
    data=stream.read(size)
    if len(data)!=size:
        raise ValueError(f"truncated SHDW while reading {description}")
    return data


def _unpack_exact(stream,format,description):
    return struct.unpack(format,_read_exact(stream,struct.calcsize(format),description))


def _skip_exact(stream,size,description):
    _read_exact(stream,size,description)


def shdw_records(path):
    records={}
    with Path(path).open("rb") as stream:
        if _read_exact(stream,4,"magic")!=b"SHDW": raise ValueError("invalid SHDW magic")
        version,count=_unpack_exact(stream,"<II","header")
        if version!=1: raise ValueError(f"unsupported SHDW version {version}")
        for _ in range(count):
            length,=_unpack_exact(stream,"<I","record name length")
            try: name=_read_exact(stream,length,"record name").decode()
            except UnicodeDecodeError as error: raise ValueError("invalid SHDW record name") from error
            kind,=_unpack_exact(stream,"<I",f"{name} kind"); shape=None
            if kind in (0,5):
                rank,=_unpack_exact(stream,"<I",f"{name} rank")
                if rank>16: raise ValueError(f"unreasonable rank {rank} for {name}")
                shape=_unpack_exact(stream,"<"+"I"*rank,f"{name} shape") if rank else ()
                _skip_exact(stream,int(np.prod(shape))*(4 if kind==0 else 2),f"{name} payload")
            elif kind==1:
                output,input_size,group,stages=_unpack_exact(stream,"<IIII",f"{name} RVQ header"); shape=(output,input_size)
                if not group or input_size%group: raise ValueError(f"invalid RVQ geometry for {name}")
                groups=input_size//group; padded=(output+63)&~63
                _skip_exact(stream,stages*group*16*4+stages*(padded//64)*groups*32+padded*4,
                            f"{name} RVQ payload")
            elif kind in (3,4,6):
                output,input_size=_unpack_exact(stream,"<II",f"{name} quantized header"); shape=(output,input_size)
                size=output*(input_size//4 if kind==3 else (input_size+4)//5 if kind==4 else (input_size+1)//2)
                _skip_exact(stream,size+output*4,f"{name} quantized payload")
            else: raise ValueError(f"unsupported SHDW record kind {kind}")
            if name in records: raise ValueError(f"duplicate SHDW record {name}")
            records[name]=(kind,shape)
        if stream.read(1): raise ValueError("trailing data after SHDW records")
    return records


def validate_export(path,manifest=None,hidden_size=1536,checkpoint_cfg=None):
    path=Path(path)
    if manifest is None: manifest=json.loads(Path(str(path)+".a55.json").read_text())
    if manifest.get("architecture_version")!=2: raise ValueError("unsupported architecture version")
    records=shdw_records(path); mtp=manifest.get("mtp",{}); horizon=int(mtp.get("horizon",1))
    if checkpoint_cfg is not None:
        checkpoint_horizon=int(checkpoint_cfg.get("mtp_horizon",1))
        if checkpoint_horizon!=horizon:
            raise ValueError(f"checkpoint MTP horizon {checkpoint_horizon} != manifest {horizon}")
        checkpoint_version=int(checkpoint_cfg.get("architecture_version",2))
        if checkpoint_version!=manifest["architecture_version"]:
            raise ValueError("checkpoint and manifest architecture versions differ")
    names={name for name in records if name.startswith("mtp.")}
    if horizon==1:
        if names: raise ValueError("K=1 manifest must not contain MTP records")
    elif horizon==2:
        required={"mtp.norm.w","mtp.down","mtp.up"}
        if not required<=names: raise ValueError(f"K=2 export missing MTP records {sorted(required-names)}")
        width=int(mtp.get("hidden_width",0))
        expected={"mtp.norm.w":(hidden_size,),"mtp.down":(width,hidden_size),
                  "mtp.up":(hidden_size,width)}
        for name,shape in expected.items():
            if records[name][1]!=shape: raise ValueError(f"{name} shape {records[name][1]} != {shape}")
        if manifest.get("compatible_with_bundled_engine"): raise ValueError("K=2 cannot claim bundled-engine compatibility")
        if mtp.get("conditioning")!="previous_token_embedding": raise ValueError("unknown MTP conditioning contract")
    else: raise ValueError(f"unsupported MTP horizon {horizon}")
    return records


def main():
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument("model"); parser.add_argument("--hidden-size",type=int,default=1536)
    args=parser.parse_args(); records=validate_export(args.model,hidden_size=args.hidden_size)
    print(f"validated {len(records)} records in {args.model}")


if __name__=="__main__": main()
