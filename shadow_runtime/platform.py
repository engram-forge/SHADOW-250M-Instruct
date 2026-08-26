"""Platform-specific bundled kernel selection."""
import os
import pathlib
import platform


def bundled_kernel(root=None, system=None, machine=None):
    root = pathlib.Path(root or pathlib.Path(__file__).resolve().parent.parent)
    system = system or platform.system()
    machine = (machine or platform.machine()).lower()
    if system == "Windows":
        return root / "deployment" / "bin" / "windows" / "shadow.exe"
    if system == "Linux":
        if machine in {"aarch64", "arm64"}:
            return root / "deployment" / "bin" / "linux-arm64" / "shadow"
        if machine in {"x86_64", "amd64"}:
            return root / "deployment" / "bin" / "linux" / "shadow"
        raise RuntimeError(f"no bundled SHADOW runtime for Linux {machine}")
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        return root / "deployment" / "bin" / "macos" / "shadow"
    if system == "Darwin":
        raise RuntimeError("the bundled macOS runtime supports Apple Silicon only")
    raise RuntimeError(f"no bundled SHADOW runtime for {system} {machine}")


def ensure_executable(path):
    path = pathlib.Path(path)
    if os.name != "nt":
        path.chmod(path.stat().st_mode | 0o111)
    return path
