import hashlib
import pathlib
import struct
import tempfile
import unittest

import numpy as np
from finetune.build_kv_archive import HEADER, MAGIC, write_archive


class KVArchiveTest(unittest.TestCase):
    def test_layout_and_payload(self):
        keys = np.arange(2 * 2 * 5 * 8, dtype=np.uint8).reshape(2, 2, 5, 8)
        values = keys ^ 0x55
        positions = np.arange(5, dtype=np.uint64)
        tokens = np.arange(100, 105, dtype=np.uint32)
        digest = hashlib.sha256(b"fixture").digest()
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "fixture.shkv"
            write_archive(path, keys, values, positions, tokens, digest, digest)
            raw = path.read_bytes(); fields = HEADER.unpack_from(raw)
            self.assertEqual(fields[0], MAGIC); self.assertEqual(fields[1:7], (1, 256, 2, 2, 8, 256))
            count, po, ko, vo, to = fields[7:12]
            self.assertEqual(count, 5)
            self.assertEqual(np.frombuffer(raw, dtype="<u8", count=5, offset=po).tolist(), positions.tolist())
            self.assertEqual(raw[ko:ko + keys.nbytes], keys.tobytes())
            self.assertEqual(raw[vo:vo + values.nbytes], values.tobytes())
            self.assertEqual(np.frombuffer(raw, dtype="<u4", count=5, offset=to).tolist(), tokens.tolist())


if __name__ == "__main__":
    unittest.main()
