import pytest


torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")


@pytest.mark.parametrize(
    "shape",
    [(1536, 512), (1536, 1536), (128, 1536), (4224, 1536),
     (1536, 4224), (4224, 3072), (512, 1536)],
)
def test_tilelang_gemv_matches_torch(shape):
    pytest.importorskip("tilelang")
    from shadow_tilelang.kernels import compile_gemv

    torch.manual_seed(sum(shape))
    x = torch.randn(shape[1], device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    actual = compile_gemv(*shape)(x, weight)
    expected = torch.nn.functional.linear(x, weight)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test_tilelang_engine_matches_cuda_reference():
    pytest.importorskip("tilelang")
    from pathlib import Path
    from shadow_tilelang.engine import TileLangEngine

    root = Path(__file__).resolve().parents[1]
    paths = (root / "deployment/shadow250m_instruct.shdw", root / "deployment/fp131072.npy")
    prompt = [2, 8, 925, 1234]
    with torch.inference_mode():
        reference = TileLangEngine(*paths, backend="torch")
        native = TileLangEngine(*paths, backend="tilelang")
        try:
            expected = reference.generate(prompt, 8)
            actual = native.generate(prompt, 8)
        finally:
            reference.close()
            native.close()
    assert actual == expected == [356, 296, 306, 356, 297, 267, 356, 296]


def test_tilelang_ternary_unpack_matches_cpu():
    import numpy as np
    from shadow_tilelang.format import TernaryRecord, unpack_ternary
    from shadow_tilelang.kernels import compile_ternary_unpack

    packed = np.array([[183, 5], [0, 242]], dtype=np.uint8)
    scales = np.array([0.5, 0.25], dtype=np.float32)
    record = TernaryRecord("test", 4, 2, 7, packed, scales)
    actual = compile_ternary_unpack(2, 7)(
        torch.from_numpy(packed).cuda(), torch.from_numpy(scales).cuda()
    )
    expected = torch.from_numpy(unpack_ternary(record)).cuda().bfloat16()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_tilelang_rvq_unpack_matches_cpu():
    import numpy as np
    from shadow_tilelang.format import RVQRecord, unpack_rvq
    from shadow_tilelang.kernels import compile_rvq_unpack

    codebooks = np.zeros((2, 2, 16), dtype=np.float32)
    codebooks[0, 0] = np.arange(16) / 8
    codebooks[0, 1] = np.arange(16) / 16
    codebooks[1] = codebooks[0] / 4
    indices = np.arange(2 * 2 * 32, dtype=np.uint8).reshape(2, 1, 2, 32)
    scales = np.linspace(0.01, 0.64, 64, dtype=np.float32)
    record = RVQRecord("test", 1, 64, 4, 2, 2, codebooks, indices, scales)
    actual = compile_rvq_unpack(64, 4, 2, 2)(
        torch.from_numpy(codebooks).cuda(), torch.from_numpy(indices).cuda(),
        torch.from_numpy(scales).cuda(),
    )
    expected = torch.from_numpy(unpack_rvq(record)).cuda().bfloat16()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_tilelang_ternary_gemv_matches_materialized_weight():
    import numpy as np
    from shadow_tilelang.format import TernaryRecord, unpack_ternary
    from shadow_tilelang.kernels import compile_ternary_gemv

    rng = np.random.default_rng(17)
    packed = rng.integers(0, 243, size=(64, 39), dtype=np.uint8)
    scales = rng.normal(size=64).astype(np.float32)
    record = TernaryRecord("test", 4, 64, 192, packed, scales)
    x = torch.randn(192, device="cuda", dtype=torch.bfloat16)
    actual = compile_ternary_gemv(64, 192)(
        x, torch.from_numpy(packed).cuda(), torch.from_numpy(scales).cuda()
    )
    weight = torch.from_numpy(unpack_ternary(record)).cuda().bfloat16()
    expected = torch.nn.functional.linear(x, weight)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_tilelang_rvq_gemv_matches_materialized_sections():
    import numpy as np
    from shadow_tilelang.format import RVQRecord, unpack_rvq
    from shadow_tilelang.kernels import compile_rvq_gemv

    rng = np.random.default_rng(23)
    codebooks = rng.normal(size=(2, 2, 8, 16)).astype(np.float32)
    indices = rng.integers(0, 256, size=(2, 2, 24, 32), dtype=np.uint8)
    scales = rng.normal(size=128).astype(np.float32)
    x = torch.randn(192, device="cuda", dtype=torch.bfloat16)
    actual = compile_rvq_gemv(128, 192, 8, 2)(
        x, torch.from_numpy(codebooks).cuda(), torch.from_numpy(indices).cuda(),
        torch.from_numpy(scales).cuda(),
    )
    sections = []
    for section in range(2):
        record = RVQRecord(
            "test", 1, 64, 192, 8, 2, codebooks[section],
            indices[:, section : section + 1], scales[section * 64 : (section + 1) * 64],
        )
        sections.append(torch.from_numpy(unpack_rvq(record)).cuda().bfloat16())
    expected = torch.nn.functional.linear(x, torch.cat(sections))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_tilelang_engine_keeps_quantized_weights_packed():
    from pathlib import Path
    from shadow_tilelang.engine import TileLangEngine
    from shadow_tilelang.kernels import PackedRVQWeight, PackedTernaryWeight

    root = Path(__file__).resolve().parents[1]
    paths = (root / "deployment/shadow250m_instruct.shdw", root / "deployment/fp131072.npy")
    engine = TileLangEngine(*paths, backend="tilelang")
    try:
        assert isinstance(engine.weights["b.0.qkv"], PackedRVQWeight)
        assert isinstance(engine.weights["b.0.up_gate"], PackedTernaryWeight)
        assert isinstance(engine.weights["b.0.dn"], PackedTernaryWeight)
        assert isinstance(engine.weights["step.Wq"], PackedRVQWeight)
    finally:
        engine.close()
