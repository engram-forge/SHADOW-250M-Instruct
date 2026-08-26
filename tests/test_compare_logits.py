import pathlib
import subprocess
import sys
import tempfile
import unittest

import numpy as np


class CompareLogitsTest(unittest.TestCase):
    def test_equal_arrays_pass_a_strict_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            values = np.array([[1.0, 2.0, 3.0]], np.float32)
            np.save(root / "a.npy", values); np.save(root / "b.npy", values)
            result = subprocess.run([sys.executable, "benchmarks/compare_logits.py",
                                     root / "a.npy", root / "b.npy", "--max-abs", "0"],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('"argmax_agreement": 1.0', result.stdout)


if __name__ == "__main__":
    unittest.main()
