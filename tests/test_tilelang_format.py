from pathlib import Path

import numpy as np

from shadow_tilelang.format import (
    DenseRecord, RVQRecord, ShadowModelFile, TernaryRecord, unpack_rvq, unpack_ternary,
)


ROOT = Path(__file__).resolve().parents[1]


def test_release_model_is_consumed_exactly():
    with ShadowModelFile(ROOT / "deployment/shadow250m_instruct.shdw") as model:
        assert len(model.records) == 140
        assert isinstance(model["b.0.q"], RVQRecord)
        assert isinstance(model["b.0.up"], TernaryRecord)
        assert isinstance(model["emb.weight"], DenseRecord)
        assert model["b.0.q"].codebooks.shape == (2, 8, 16)
        assert model["b.0.up"].packed.shape == (4224, 308)
        assert model["emb.weight"].value.shape == (1536, 512)


def test_unpack_base3_is_row_local():
    record = TernaryRecord(
        "test", 4, 1, 7, np.array([[183, 5]], dtype=np.uint8),
        np.array([0.5], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        unpack_ternary(record),
        np.array([[-0.5, 0.0, 0.5, -0.5, 0.5, 0.5, 0.0]], dtype=np.float32),
    )


def test_unpack_rvq_nibble_layout():
    codebooks = np.zeros((1, 2, 16), dtype=np.float32)
    codebooks[0, 0] = np.arange(16)
    codebooks[0, 1] = np.arange(16) + 20
    indices = np.zeros((1, 1, 1, 32), dtype=np.uint8)
    indices[0, 0, 0, 0] = 0xA3
    scales = np.ones(64, dtype=np.float32)
    record = RVQRecord("test", 1, 64, 2, 2, 1, codebooks, indices, scales)
    matrix = unpack_rvq(record)
    np.testing.assert_array_equal(matrix[0], [3, 23])
    np.testing.assert_array_equal(matrix[32], [10, 30])
