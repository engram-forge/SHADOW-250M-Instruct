import unittest
from benchmarks.release_qualification import cases, validate

class QualificationTest(unittest.TestCase):
    def test_suite_is_balanced_and_unique(self):
        suite = cases(); self.assertEqual(len(suite), 100); self.assertEqual(len({x["id"] for x in suite}), 100)

    def test_validators_accept_reference_shapes(self):
        samples = {
            "json": ('{"a": 1, "status": "ok", "items": [1, 2]}', {"keys":["a","status","items"],"array_key":"items","array_length":2,"value_key":"status","value":"ok"}),
            "numbered_list": ("1. One\n2. Two\n3. Three", {"count":3}),
            "python": ("```python\ndef add(a, b):\n    return a + b\n```", {"function":"add"}),
            "markdown_table": ("| A | B | C |\n|---|---|---|\n| 1 | 2 | 3 |\n| 4 | 5 | 6 |", {"columns":3,"rows":2}),
            "constraints": ("Plan marker1. Work carefully.", {"sentences":2,"required":"marker1","forbidden":"pirate"}),
        }
        for kind,(text,expected) in samples.items():
            ok,_=validate({"validator":kind,"expected":expected},text); self.assertTrue(ok,kind)

if __name__ == "__main__": unittest.main()
