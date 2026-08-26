import pathlib
import unittest
from unittest import mock
import tempfile
from shadow_runtime import Engine
from shadow_runtime.platform import bundled_kernel


class PlatformSelectionTest(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path("/repo")

    def test_selects_each_bundled_runtime(self):
        self.assertEqual(bundled_kernel(self.root, "Windows", "AMD64"), self.root / "deployment/bin/windows/shadow.exe")
        self.assertEqual(bundled_kernel(self.root, "Linux", "x86_64"), self.root / "deployment/bin/linux/shadow")
        self.assertEqual(bundled_kernel(self.root, "Linux", "aarch64"), self.root / "deployment/bin/linux-arm64/shadow")
        self.assertEqual(bundled_kernel(self.root, "Darwin", "arm64"), self.root / "deployment/bin/macos/shadow")

    def test_rejects_intel_mac(self):
        with self.assertRaisesRegex(RuntimeError, "Apple Silicon"):
            bundled_kernel(self.root, "Darwin", "x86_64")

    def test_rejects_unknown_linux_architecture(self):
        with self.assertRaisesRegex(RuntimeError, "Linux riscv64"):
            bundled_kernel(self.root, "Linux", "riscv64")

    def test_native_archive_arguments_are_forwarded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            for name in ("model.shdw", "table.npy", "archive.shkv", "shadow"):
                (root / name).write_bytes(b"x")
            engine = Engine(root / "model.shdw", root / "table.npy", kernel=root / "shadow",
                            kv_archive=root / "archive.shkv", archive_backend="cpu", archive_top_k=7)
            completed = mock.Mock(returncode=0, stdout="", stderr="")
            with mock.patch("shadow_runtime.subprocess.run", return_value=completed) as run:
                engine._gen([2], 1)
            command = run.call_args.args[0]
            self.assertEqual(command[-6:], ["--archive", str((root / "archive.shkv").resolve()),
                                            "--archive-backend", "cpu", "--archive-topk", "7"])


if __name__ == "__main__":
    unittest.main()
