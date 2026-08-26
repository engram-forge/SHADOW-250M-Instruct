"""Structural release checks for A55 SHDW exports and their manifests."""

import json
from pathlib import Path
import struct

import numpy as np


def shdw_records(path):
    records={}
    with Path(path).open("rb") as stream:
        if stream.read(4)!=b"SHDW": raise ValueError("invalid SHDW magic")
        _,count=struct.unpack("<II",stream.read(8))
        for _ in range(count):
            length,=struct.unpack("<I",stream.read(4)); name=stream.read(length).decode()
            kind,=struct.unpack("<I",stream.read(4)); shape=None
            if kind in (0,5):
                rank,=struct.unpack("<I",stream.read(4)); shape=struct.unpack("<"+"I"*rank,stream.read(4*rank))
                stream.seek(int(np.prod(shape))*(4 if kind==0 else 2),1)
            elif kind==1:
                output,input_size,group,stages=struct.unpack("<IIII",stream.read(16)); shape=(output,input_size)
                groups=input_size//group; padded=(output+63)&~63
                stream.seek(stages*group*16*4+stages*(padded//64)*groups*32+padded*4,1)
            elif kind in (3,4,6):
                output,input_size=struct.unpack("<II",stream.read(8)); shape=(output,input_size)
                size=output*(input_size//4 if kind==3 else (input_size+4)//5 if kind==4 else (input_size+1)//2)
                stream.seek(size+output*4,1)
            else: raise ValueError(f"unsupported SHDW record kind {kind}")
            if name in records: raise ValueError(f"duplicate SHDW record {name}")
            records[name]=(kind,shape)
    return records


def validate_export(path,manifest=None,hidden_size=1536):
    path=Path(path)
    if manifest is None: manifest=json.loads(Path(str(path)+".a55.json").read_text())
    if manifest.get("architecture_version")!=2: raise ValueError("unsupported architecture version")
    records=shdw_records(path); mtp=manifest.get("mtp",{}); horizon=int(mtp.get("horizon",1))
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
