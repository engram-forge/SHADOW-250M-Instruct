import importlib.util

import pytest


torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("tilelang") is None,
    reason="CUDA GPU and TileLang required",
)


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


@pytest.mark.parametrize("max_context,token_count", [(32, 12), (8, 12)])
def test_tilelang_async_greedy_generation_matches_reference(
    max_context, token_count
):
    from pathlib import Path
    from shadow_tilelang.engine import TileLangEngine

    root = Path(__file__).resolve().parents[1]
    paths = (root / "deployment/shadow250m_instruct.shdw", root / "deployment/fp131072.npy")
    prompt = [2, 8, 925, 1234]
    with torch.inference_mode():
        reference = TileLangEngine(*paths, backend="torch", max_context=max_context)
        native = TileLangEngine(*paths, backend="tilelang", max_context=max_context)
        try:
            expected = reference.generate(
                prompt, token_count, repetition_penalty=1.0, stop_ids=()
            )
            actual = native.generate(
                prompt, token_count, repetition_penalty=1.0, stop_ids=()
            )
            assert actual == expected
            assert native.position == reference.position == len(prompt) + token_count
        finally:
            reference.close()
            native.close()


def test_tilelang_greedy_generation_crosses_split_attention_boundary():
    from pathlib import Path
    from shadow_tilelang.engine import SPLIT_ATTENTION_POSITION, TileLangEngine

    root = Path(__file__).resolve().parents[1]
    paths = (root / "deployment/shadow250m_instruct.shdw", root / "deployment/fp131072.npy")
    with torch.inference_mode():
        engine = TileLangEngine(*paths, backend="tilelang", max_context=512)
        try:
            engine.position = SPLIT_ATTENTION_POSITION - 2
            engine._position_cuda.fill_(engine.position)
            logits = engine.step(925)
            generated = engine._generate_greedy_cuda(logits, 4)
            assert len(generated) == 4
            assert engine.position == SPLIT_ATTENTION_POSITION + 3
            assert engine._greedy_graph is not None
            assert engine._greedy_graph_split is not None
        finally:
            engine.close()


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
    materialized = unpack_ternary(record)
    trits = (materialized / scales[:, None] + 1).astype(np.uint8)
    packed_2bit = np.zeros((64, 39), dtype=np.uint16)
    for component in range(5):
        values = trits[:, component::5].astype(np.uint16)
        packed_2bit[:, :values.shape[1]] |= values << (component * 2)
    actual = compile_ternary_gemv(64, 192)(
        x, torch.from_numpy(packed_2bit).cuda(), torch.from_numpy(scales).cuda()
    )
    weight = torch.from_numpy(materialized).cuda().bfloat16()
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
    pair_codebooks = (
        codebooks[:, 0, :, None, :] + codebooks[:, 1, :, :, None]
    ).reshape(2, 8, 256)
    low, high = indices & 15, indices >> 4
    pair_indices = np.concatenate(
        (low[0] | (low[1] << 4), high[0] | (high[1] << 4)), axis=2
    ).transpose(0, 2, 1)
    actual = compile_rvq_gemv(128, 192, 8, 2)(
        x, torch.from_numpy(pair_codebooks).cuda(),
        torch.from_numpy(pair_indices.copy()).cuda(),
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


def test_tilelang_engine_uses_decode_optimized_weight_storage():
    from pathlib import Path
    from shadow_tilelang.engine import TileLangEngine
    from shadow_tilelang.kernels import PackedRVQWeight, PackedTernaryWeight

    root = Path(__file__).resolve().parents[1]
    paths = (root / "deployment/shadow250m_instruct.shdw", root / "deployment/fp131072.npy")
    engine = TileLangEngine(*paths, backend="tilelang")
    try:
        assert isinstance(engine.weights["b.0.qkv"], PackedRVQWeight)
        assert isinstance(engine.weights["b.0.up_gate"], PackedTernaryWeight)
        assert isinstance(engine.weights["b.0.dn"], torch.Tensor)
        assert engine.weights["b.0.dn"].dtype == torch.bfloat16
        assert isinstance(engine.weights["step.Wq"], PackedRVQWeight)
    finally:
        engine.close()


@pytest.mark.parametrize(
    "max_context,position",
    [
        (12, 11), (12, 19),
        (512, 511), (512, 519),
        (2048, 2047), (2048, 2055),
    ],
)
def test_tilelang_attention_matches_reference_with_circular_cache(
    max_context, position
):
    from shadow_tilelang.engine import _power_of_two_quantize
    from shadow_tilelang.kernels import compile_attention, compile_attention_cache_update

    query_heads, kv_heads, head_dim = 4, 2, 64
    torch.manual_seed(31 + position)
    query = _power_of_two_quantize(
        torch.randn(query_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    )
    chronological_keys = _power_of_two_quantize(
        torch.randn(kv_heads, max_context, head_dim, device="cuda", dtype=torch.bfloat16)
    )
    chronological_values = _power_of_two_quantize(
        torch.randn(kv_heads, max_context, head_dim, device="cuda", dtype=torch.bfloat16)
    )
    start = 0 if position + 1 <= max_context else (position + 1) % max_context
    slots = (torch.arange(max_context, device="cuda") + start) % max_context
    keys = torch.empty_like(chronological_keys)
    values = torch.empty_like(chronological_values)
    keys[:, slots] = chronological_keys
    values[:, slots] = chronological_values
    alpha = (torch.randn(query_heads, device="cuda") * 4096).round() / 4096
    position_cuda = torch.tensor([position], device="cuda", dtype=torch.int32)
    compile_attention_cache_update(kv_heads, head_dim, max_context)(
        chronological_keys[:, -1].contiguous(),
        chronological_values[:, -1].contiguous(), keys, values, position_cuda,
    )
    actual = compile_attention(query_heads, kv_heads, head_dim, max_context)(
        query, keys, values, alpha, position_cuda,
    )
    repeated_keys = chronological_keys.repeat_interleave(query_heads // kv_heads, dim=0)
    repeated_values = chronological_values.repeat_interleave(query_heads // kv_heads, dim=0)
    scores = torch.einsum("hd,htd->ht", query, repeated_keys)
    scores = torch.floor(scores * alpha[:, None])
    probability = torch.exp2(
        (scores - scores.amax(-1, keepdim=True)).clamp_min(-15)
    )
    probability /= probability.sum(-1, keepdim=True)
    expected = torch.einsum(
        "ht,htd->hd", probability.to(repeated_values.dtype), repeated_values
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_tilelang_attention_quantizes_value_before_cache_write():
    from shadow_tilelang.engine import _power_of_two_quantize
    from shadow_tilelang.kernels import compile_attention_cache_update

    query_heads, kv_heads, head_dim, max_context = 24, 2, 64, 32
    torch.manual_seed(91)
    query = _power_of_two_quantize(
        torch.randn(query_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    )
    key = _power_of_two_quantize(
        torch.randn(kv_heads, head_dim, device="cuda", dtype=torch.bfloat16)
    )
    value = torch.randn(
        kv_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    keys = torch.zeros(
        kv_heads, max_context, head_dim, device="cuda", dtype=torch.bfloat16
    )
    values = torch.zeros_like(keys)
    position = torch.tensor([7], device="cuda", dtype=torch.int32)
    compile_attention_cache_update(kv_heads, head_dim, max_context)(
        key, value, keys, values, position,
    )
    torch.testing.assert_close(
        values[:, 7], _power_of_two_quantize(value), rtol=0, atol=0
    )


def test_tilelang_attention_full_context_equal_scores_are_bit_exact():
    from shadow_tilelang.kernels import compile_attention

    query_heads, kv_heads, head_dim, max_context = 24, 2, 64, 2048
    query = torch.zeros(
        query_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    keys = torch.zeros(
        kv_heads, max_context, head_dim, device="cuda", dtype=torch.bfloat16
    )
    torch.manual_seed(771)
    values = torch.randn_like(keys)
    alpha = torch.ones(query_heads, device="cuda")
    position = torch.tensor([max_context - 1], device="cuda", dtype=torch.int32)
    actual = compile_attention(
        query_heads, kv_heads, head_dim, max_context
    )(query, keys, values, alpha, position)
    expected = values.float().mean(dim=1).bfloat16().repeat_interleave(
        query_heads // kv_heads, dim=0
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("max_context,position", [(12, 19), (512, 519), (2048, 2055)])
def test_tilelang_split_attention_is_bit_exact(max_context, position):
    from shadow_tilelang.kernels import (
        compile_attention, compile_attention_scores, compile_attention_values,
    )

    query_heads, kv_heads, head_dim = 24, 2, 64
    torch.manual_seed(1700 + position)
    query = torch.randn(
        query_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    keys = torch.randn(
        kv_heads, max_context, head_dim, device="cuda", dtype=torch.bfloat16
    )
    values = torch.randn_like(keys)
    alpha = (torch.randn(query_heads, device="cuda") * 4096).round() / 4096
    position_cuda = torch.tensor([position], device="cuda", dtype=torch.int32)
    expected = compile_attention(
        query_heads, kv_heads, head_dim, max_context
    )(query, keys, values, alpha, position_cuda)
    scores = compile_attention_scores(
        query_heads, kv_heads, head_dim, max_context
    )(query, keys, alpha, position_cuda)
    actual = compile_attention_values(
        query_heads, kv_heads, head_dim, max_context
    )(scores, values, position_cuda)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_tilelang_engine_circular_cache_matches_reference_after_wrap():
    from pathlib import Path
    from shadow_tilelang.engine import TileLangEngine

    root = Path(__file__).resolve().parents[1]
    paths = (root / "deployment/shadow250m_instruct.shdw", root / "deployment/fp131072.npy")
    prompt = [2, 8, 925, 1234, 356, 296, 306]
    with torch.inference_mode():
        reference = TileLangEngine(*paths, backend="torch", max_context=4)
        native = TileLangEngine(*paths, backend="tilelang", max_context=4)
        try:
            for token in prompt:
                expected = reference.step(token)
                actual = native.step(token)
                assert int(actual.argmax()) == int(expected.argmax())
        finally:
            reference.close()
            native.close()


def test_tilelang_structural_cache_preserves_chronological_order():
    from shadow_tilelang.kernels import compile_circular_gather, compile_circular_store

    max_context, width = 4, 64
    cache = torch.zeros(
        max_context, width, device="cuda", dtype=torch.bfloat16
    )
    chronological = []
    for position in range(7):
        value = torch.full(
            (width,), position + 0.5, device="cuda", dtype=torch.bfloat16
        )
        position_cuda = torch.tensor(
            [position], device="cuda", dtype=torch.int32
        )
        compile_circular_store(max_context, width)(
            value, cache, position_cuda
        )
        chronological.append(value)
        chronological = chronological[-max_context:]
        actual = compile_circular_gather(max_context, width)(
            cache, position_cuda
        )[:len(chronological)]
        expected = torch.stack(chronological)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("position", [2, 6])
def test_tilelang_structural_attention_is_invariant_to_cache_order(position):
    from shadow_tilelang.kernels import compile_circular_gather

    max_context, width = 4, 64
    torch.manual_seed(37 + position)
    cache = torch.randn(
        max_context, width, device="cuda", dtype=torch.bfloat16
    )
    query = torch.randn(width, device="cuda", dtype=torch.bfloat16)
    position_cuda = torch.tensor([position], device="cuda", dtype=torch.int32)
    chronological = compile_circular_gather(max_context, width)(cache, position_cuda)
    valid_count = min(position + 1, max_context)

    expected_scores = chronological @ query
    expected_scores[valid_count:] = float("-inf")
    expected_probability = torch.softmax(expected_scores.float(), dim=-1).bfloat16()
    expected = expected_probability @ chronological

    physical_scores = cache @ query
    if position + 1 < max_context:
        physical_scores[position + 1:] = float("-inf")
    physical_probability = torch.softmax(physical_scores.float(), dim=-1).bfloat16()
    actual = physical_probability @ cache
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("max_context,position", [(32, 3), (32, 34), (2048, 2050)])
def test_tilelang_structural_softmax_is_bit_exact(max_context, position):
    from shadow_tilelang.kernels import compile_structural_softmax

    torch.manual_seed(max_context + position)
    scores = torch.randn(
        max_context, device="cuda", dtype=torch.bfloat16
    )
    position_cuda = torch.tensor([position], device="cuda", dtype=torch.int32)
    actual = compile_structural_softmax(max_context)(scores, position_cuda)
    expected_scores = scores.clone()
    if position + 1 < max_context:
        expected_scores[position + 1:] = float("-inf")
    expected = torch.softmax(expected_scores.float(), dim=-1).bfloat16()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("length", [2, 4, 7])
def test_tilelang_batched_prefill_matches_reference_and_decode(length):
    from pathlib import Path
    from shadow_tilelang.engine import TileLangEngine

    root = Path(__file__).resolve().parents[1]
    paths = (root / "deployment/shadow250m_instruct.shdw", root / "deployment/fp131072.npy")
    prompt = [2, 8, 925, 1234, 356, 296, 306][:length]
    with torch.inference_mode():
        reference = TileLangEngine(*paths, backend="torch", max_context=16)
        native = TileLangEngine(*paths, backend="tilelang", max_context=16)
        try:
            expected = reference.prefill(prompt)
            actual = native.prefill(prompt)
            assert int(actual.argmax()) == int(expected.argmax())
            token = int(expected.argmax())
            expected = reference.step(token)
            actual = native.step(token)
            assert int(actual.argmax()) == int(expected.argmax())
        finally:
            reference.close()
            native.close()


def test_tilelang_packed_fingerprint_logits_match_dense_reference():
    import numpy as np
    from shadow_tilelang.kernels import (
        compile_fingerprint_gather, compile_fingerprint_logits,
        compile_fingerprint_unpack,
        compile_fingerprint_unpack_batch,
    )

    vocabulary, features = 257, 512
    rng = np.random.default_rng(41)
    packed = rng.integers(0, 256, size=(vocabulary, features // 8), dtype=np.uint8)
    packed_cuda = torch.from_numpy(packed).cuda()
    dense = torch.from_numpy(
        np.unpackbits(packed, axis=1).astype(np.float32) * 2.0 - 1.0
    ).cuda().bfloat16()
    indices = torch.tensor([0, 37, vocabulary - 1], device="cuda")
    torch.testing.assert_close(
        compile_fingerprint_unpack(features)(packed_cuda[37]), dense[37], rtol=0, atol=0
    )
    torch.testing.assert_close(
        compile_fingerprint_gather(vocabulary, features)(
            packed_cuda, torch.tensor([37], device="cuda")
        ),
        dense[37], rtol=0, atol=0,
    )
    torch.testing.assert_close(
        compile_fingerprint_unpack_batch(3, vocabulary, features)(packed_cuda, indices),
        dense[indices], rtol=0, atol=0,
    )
    projected = torch.randn(features, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(vocabulary, device="cuda")
    actual = compile_fingerprint_logits(vocabulary, features)(projected, packed_cuda, bias)
    expected = projected.float() @ (dense.float().T / (features ** 0.5)) + bias
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert int(actual.argmax()) == int(expected.argmax())


@pytest.mark.parametrize("token", [0, 2, 925, 131071])
def test_tilelang_packed_fingerprint_embedding_is_bit_exact(token):
    from pathlib import Path
    from shadow_tilelang.engine import D, FINGERPRINT_DIM, VOCAB_SIZE, TileLangEngine
    from shadow_tilelang.kernels import compile_fingerprint_embedding

    root = Path(__file__).resolve().parents[1]
    paths = (
        root / "deployment/shadow250m_instruct.shdw",
        root / "deployment/fp131072.npy",
    )
    with torch.inference_mode():
        engine = TileLangEngine(*paths, backend="tilelang", max_context=16)
        try:
            engine._token_cuda.fill_(token)
            expected = engine.linear(
                engine._fingerprint_cuda(), engine.weights["emb.weight"]
            )
            actual = compile_fingerprint_embedding(
                VOCAB_SIZE, FINGERPRINT_DIM, D
            )(
                engine.fingerprints, engine._token_cuda,
                engine.weights["emb.weight"],
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            engine.close()


def test_tilelang_decode_preserves_full_logits_and_tokens():
    from pathlib import Path
    from shadow_tilelang.engine import TileLangEngine

    root = Path(__file__).resolve().parents[1]
    paths = (root / "deployment/shadow250m_instruct.shdw", root / "deployment/fp131072.npy")
    prompt = [2, 8, 925, 1234]
    with torch.inference_mode():
        reference = TileLangEngine(*paths, backend="torch", max_context=32)
        native = TileLangEngine(*paths, backend="tilelang", max_context=32)
        try:
            expected = reference.prefill(prompt)
            actual = native.prefill(prompt)
            for _ in range(12):
                torch.testing.assert_close(actual, expected, rtol=0.025, atol=1.0)
                expected_token = int(expected.argmax())
                assert int(actual.argmax()) == expected_token
                expected = reference.step(expected_token)
                actual = native.step(expected_token)
        finally:
            reference.close()
            native.close()


@pytest.mark.parametrize("shape", [(1, 64), (2, 64), (1, 1536), (4, 1536)])
def test_tilelang_rms_norm_is_bit_exact(shape):
    from shadow_tilelang.engine import _rms_norm
    from shadow_tilelang.kernels import compile_rms_norm

    torch.manual_seed(sum(shape))
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(shape[-1], device="cuda")
    actual = compile_rms_norm(*shape)(x, weight)
    expected = _rms_norm(x, weight)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_tilelang_residual_rms_norm_is_bit_exact():
    from shadow_tilelang.engine import _rms_norm
    from shadow_tilelang.kernels import compile_residual_rms_norm

    torch.manual_seed(901)
    residual = torch.randn(1536, device="cuda", dtype=torch.bfloat16)
    projected = torch.randn_like(residual)
    weight = torch.randn(1536, device="cuda")
    actual, actual_normalized = compile_residual_rms_norm(1536)(
        residual, projected, weight
    )
    expected = residual + projected
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_normalized, _rms_norm(expected, weight), rtol=0, atol=0
    )


def test_tilelang_combined_qk_norm_is_bit_exact():
    from shadow_tilelang.kernels import compile_qk_rms_norm, compile_rms_norm

    query_heads, key_heads, head_dim = 24, 2, 64
    torch.manual_seed(317)
    qk = torch.randn(
        query_heads + key_heads, head_dim,
        device="cuda", dtype=torch.bfloat16,
    )
    query_weight = torch.randn(head_dim, device="cuda")
    key_weight = torch.randn(head_dim, device="cuda")
    expected = torch.cat((
        compile_rms_norm(query_heads, head_dim)(
            qk[:query_heads], query_weight
        ),
        compile_rms_norm(key_heads, head_dim)(qk[query_heads:], key_weight),
    ))
    actual = compile_qk_rms_norm(query_heads, key_heads, head_dim)(
        qk, query_weight, key_weight
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_tilelang_fused_qk_norm_rope_cache_is_bit_exact():
    from shadow_tilelang.engine import _power_of_two_quantize
    from shadow_tilelang.kernels import (
        compile_qk_rms_norm, compile_rope_quantize,
        compile_rope_quantize_cache,
    )

    query_heads, key_heads, head_dim, max_context = 24, 2, 64, 32
    torch.manual_seed(881)
    qk = torch.randn(
        query_heads + key_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    value = torch.randn(
        key_heads, head_dim, device="cuda", dtype=torch.bfloat16
    )
    query_weight = torch.randn(head_dim, device="cuda")
    key_weight = torch.randn(head_dim, device="cuda")
    cosine = torch.randn(head_dim // 2, device="cuda", dtype=torch.bfloat16)
    sine = torch.randn_like(cosine)
    keys = torch.zeros(
        key_heads, max_context, head_dim, device="cuda", dtype=torch.bfloat16
    )
    values = torch.zeros_like(keys)
    position = torch.tensor([7], device="cuda", dtype=torch.int32)
    normalized = compile_qk_rms_norm(query_heads, key_heads, head_dim)(
        qk, query_weight, key_weight
    )
    expected = compile_rope_quantize(query_heads + key_heads, head_dim)(
        normalized, cosine, sine
    )
    actual = compile_rope_quantize_cache(
        query_heads, key_heads, head_dim, max_context
    )(
        qk, value, query_weight, key_weight, cosine, sine, keys, values, position
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(keys[:, 7], expected[query_heads:], rtol=0, atol=0)
    torch.testing.assert_close(
        values[:, 7], _power_of_two_quantize(value), rtol=0, atol=0
    )


@pytest.mark.parametrize("shape", [(24, 64), (2, 64), (4, 64)])
def test_tilelang_power_of_two_quantizer_is_bit_exact(shape):
    from shadow_tilelang.engine import _power_of_two_quantize
    from shadow_tilelang.kernels import compile_power_of_two_quantize

    torch.manual_seed(sum(shape))
    x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
    actual = compile_power_of_two_quantize(*shape)(x)
    expected = _power_of_two_quantize(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("heads", [2, 24])
def test_tilelang_rope_is_bit_exact(heads):
    from shadow_tilelang.kernels import compile_rope

    torch.manual_seed(heads)
    x = torch.randn(heads, 64, device="cuda", dtype=torch.bfloat16)
    angle = torch.randn(32, device="cuda")
    cosine, sine = angle.cos().bfloat16(), angle.sin().bfloat16()
    actual = compile_rope(heads, 64)(x, cosine, sine)
    even, odd = x[..., 0::2], x[..., 1::2]
    expected = torch.stack(
        (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
    ).flatten(-2)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("heads", [2, 24])
def test_tilelang_rope_quantize_fusion_is_bit_exact(heads):
    from shadow_tilelang.kernels import (
        compile_power_of_two_quantize, compile_rope, compile_rope_quantize,
    )

    torch.manual_seed(100 + heads)
    x = torch.randn(heads, 64, device="cuda", dtype=torch.bfloat16)
    angle = torch.randn(32, device="cuda")
    cosine, sine = angle.cos().bfloat16(), angle.sin().bfloat16()
    expected = compile_power_of_two_quantize(heads, 64)(
        compile_rope(heads, 64)(x, cosine, sine)
    )
    actual = compile_rope_quantize(heads, 64)(x, cosine, sine)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_tilelang_projection_epilogues_are_bit_exact():
    from pathlib import Path
    from shadow_tilelang.engine import TileLangEngine

    root = Path(__file__).resolve().parents[1]
    paths = (root / "deployment/shadow250m_instruct.shdw", root / "deployment/fp131072.npy")
    with torch.inference_mode():
        engine = TileLangEngine(*paths, backend="tilelang", max_context=16)
        try:
            torch.manual_seed(53)
            attended = torch.randn(1536, device="cuda", dtype=torch.bfloat16)
            gate = torch.randn(1536, device="cuda", dtype=torch.bfloat16)
            residual = torch.randn(1536, device="cuda", dtype=torch.bfloat16)
            actual = engine.linear.gated_residual(
                attended, gate, residual, engine.weights["b.0.o"]
            )
            expected = residual + engine.linear(
                attended * gate, engine.weights["b.0.o"]
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            hidden = torch.randn(4224, device="cuda", dtype=torch.bfloat16)
            actual = engine.linear.residual(
                hidden, residual, engine.weights["b.0.dn"]
            )
            expected = residual + engine.linear(hidden, engine.weights["b.0.dn"])
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            ffn_input = torch.randn(
                1536, device="cuda", dtype=torch.bfloat16
            )
            up_gate = engine.linear(ffn_input, engine.weights["b.0.up_gate"])
            up, gate_projection = up_gate.split(4224)
            expected = torch.nn.functional.silu(gate_projection) * up
            actual = engine.linear.swiglu(
                ffn_input, engine.weights["b.0.up_gate"]
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            recall = torch.randn(
                1536, device="cuda", dtype=torch.bfloat16
            )
            expected = torch.nn.functional.silu(
                engine.linear(
                    torch.cat((residual, recall)), engine.weights["step.cin"]
                )
            )
            actual = engine.linear.split_silu(
                residual, recall, engine.weights["step.cin"]
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            structural_hidden = torch.randn(
                4224, device="cuda", dtype=torch.bfloat16
            )
            expected = residual + engine.linear(
                structural_hidden, engine.weights["step.cout"]
            )
            actual = engine.linear.rvq_residual(
                structural_hidden, residual, engine.weights["step.cout"]
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        finally:
            engine.close()
