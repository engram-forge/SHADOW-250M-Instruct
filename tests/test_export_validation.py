import tempfile
import unittest
from pathlib import Path
import struct
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"finetune"/"modeling"))
from validate_export import validate_export


def dense_record(name,shape):
    encoded=name.encode(); count=1
    for value in shape: count*=value
    return (struct.pack("<I",len(encoded))+encoded+struct.pack("<II",0,len(shape))+
            struct.pack("<"+"I"*len(shape),*shape)+b"\0"*(4*count))


class ExportValidationTest(unittest.TestCase):
    def test_k2_rejects_missing_records_and_false_compatibility(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"bad.shdw"
            records=[dense_record("mtp.norm.w",(4,)),dense_record("mtp.down",(2,4))]
            path.write_bytes(b"SHDW"+struct.pack("<II",1,len(records))+b"".join(records))
            manifest={"architecture_version":2,"mtp":{"horizon":2,"hidden_width":2,
                "conditioning":"previous_token_embedding"},"compatible_with_bundled_engine":False}
            with self.assertRaisesRegex(ValueError,"missing MTP"):
                validate_export(path,manifest,hidden_size=4)
            records.append(dense_record("mtp.up",(4,2)))
            path.write_bytes(b"SHDW"+struct.pack("<II",1,len(records))+b"".join(records))
            manifest["compatible_with_bundled_engine"]=True
            with self.assertRaisesRegex(ValueError,"bundled-engine"):
                validate_export(path,manifest,hidden_size=4)


if __name__=="__main__": unittest.main()
