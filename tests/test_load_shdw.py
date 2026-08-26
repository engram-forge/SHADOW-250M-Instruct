import pathlib
import struct
import tempfile
import unittest

import numpy as np

from finetune.modeling.load_shdw import read_shdw


class ShdwReaderTest(unittest.TestCase):
    def test_reads_dense_and_base3_ternary_records(self):
        dense = np.array([1.5, -2.0], np.float32)
        # Five codes [-1,0,+1,-1,+1] -> base-3 digits [0,1,2,0,2].
        packed = bytes([0 + 3 + 18 + 0 + 162])
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "fixture.shdw"
            with path.open("wb") as stream:
                stream.write(b"SHDW"); stream.write(struct.pack("<II", 1, 2))
                for name, kind in ((b"dense", 0), (b"tern", 4)):
                    stream.write(struct.pack("<I", len(name))); stream.write(name); stream.write(struct.pack("<I", kind))
                    if kind == 0:
                        stream.write(struct.pack("<II", 1, 2)); stream.write(dense.tobytes())
                    else:
                        stream.write(struct.pack("<II", 1, 5)); stream.write(packed); stream.write(np.array([2.0], np.float32).tobytes())
            version, records = read_shdw(path)
            self.assertEqual(version, 1)
            np.testing.assert_array_equal(records["dense"].value, dense)
            np.testing.assert_array_equal(records["tern"].value, np.array([[-2, 0, 2, -2, 2]], np.float32))


if __name__ == "__main__":
    unittest.main()
