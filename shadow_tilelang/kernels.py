"""TileLang CUDA kernels used by the autoregressive engine."""

from dataclasses import dataclass

from functools import lru_cache


@dataclass(frozen=True)
class PackedRVQWeight:
    """CUDA-resident RVQ payload consumed directly by a GEMV kernel."""

    codebooks: object
    indices: object
    scales: object
    out_features: int
    in_features: int
    group_size: int
    stages: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.out_features, self.in_features


@dataclass(frozen=True)
class PackedTernaryWeight:
    """CUDA-resident base-3 payload consumed directly by a GEMV kernel."""

    packed: object
    scales: object
    out_features: int
    in_features: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.out_features, self.in_features


def _imports():
    try:
        import tilelang
        import tilelang.language as T
    except ImportError as exc:
        raise RuntimeError(
            "TileLang is not installed; run `uv sync --extra tilelang`"
        ) from exc
    return tilelang, T


@lru_cache(maxsize=None)
def compile_gemv(out_features: int, in_features: int):
    """Compile one BF16-input/weight, FP32-accumulating CUDA GEMV shape."""

    tilelang, T = _imports()
    # Every SHADOW deployment dimension is divisible by 128. Eight output
    # lanes times sixteen K-reduction lanes fills one 128-thread CUDA block
    # without a slow tail path, including the unusual 4224-wide FFN.
    n_partition = 8
    reduce_threads = 16
    vector = 8  # one 128-bit FP16 transaction
    block_k = reduce_threads * vector

    @tilelang.jit(target="cuda")
    def gemv(
        x: T.Tensor((in_features,), T.bfloat16),
        weight: T.Tensor((out_features, in_features), T.bfloat16),
    ):
        output = T.empty((out_features,), T.bfloat16)
        with T.Kernel(
            T.ceildiv(out_features, n_partition),
            threads=(reduce_threads, n_partition),
        ) as block:
            lane_k = T.get_thread_binding(0)
            lane_n = T.get_thread_binding(1)
            x_local = T.alloc_local((vector,), T.bfloat16)
            w_local = T.alloc_local((vector,), T.bfloat16)
            partial = T.alloc_local((1,), T.float32)
            reduced = T.alloc_local((1,), T.float32)
            T.clear(partial)
            for tile_k in T.serial(T.ceildiv(in_features, block_k)):
                for inner in T.vectorized(vector):
                    k = tile_k * block_k + lane_k * vector + inner
                    x_local[inner] = x[k]
                    w_local[inner] = weight[block * n_partition + lane_n, k]
                for inner in T.serial(vector):
                    partial[0] += (
                        x_local[inner].astype(T.float32)
                        * w_local[inner].astype(T.float32)
                    )
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1), partial[0], True, reduced[0], lane_k, dtype="handle"
                    )
                )
            if lane_k == 0:
                output[block * n_partition + lane_n] = reduced[0]
        return output

    return gemv


@lru_cache(maxsize=None)
def compile_ternary_unpack(out_features: int, in_features: int):
    """Compile CUDA expansion of row-local five-trit weight bytes."""

    tilelang, T = _imports()
    packed_width = (in_features + 4) // 5
    total = out_features * in_features
    threads = 256

    @tilelang.jit(target="cuda")
    def unpack(
        packed: T.Tensor((out_features, packed_width), T.uint8),
        scales: T.Tensor((out_features,), T.float32),
    ):
        output = T.empty((out_features, in_features), T.bfloat16)
        with T.Kernel(T.ceildiv(total, threads), threads=threads) as block:
            thread = T.get_thread_binding(0)
            linear = block * threads + thread
            if linear < total:
                row = linear // in_features
                column = linear % in_features
                component = column % 5
                divisor = T.alloc_local((1,), T.int32)
                divisor[0] = 1
                for _ in T.serial(component):
                    divisor[0] *= 3
                byte = packed[row, column // 5].astype(T.int32)
                trit = (byte // divisor[0]) % 3 - 1
                output[row, column] = trit.astype(T.float32) * scales[row]
        return output

    return unpack


@lru_cache(maxsize=None)
def compile_rvq_unpack(
    out_features: int, in_features: int, group_size: int, stages: int
):
    """Compile CUDA expansion of SHADOW's packed RVQ nibble layout."""

    tilelang, T = _imports()
    padded_out = (out_features + 63) & ~63
    groups = in_features // group_size
    chunks = padded_out // 64
    total = out_features * in_features
    threads = 256

    @tilelang.jit(target="cuda")
    def unpack(
        codebooks: T.Tensor((stages, group_size, 16), T.float32),
        indices: T.Tensor((stages, chunks, groups, 32), T.uint8),
        scales: T.Tensor((padded_out,), T.float32),
    ):
        output = T.empty((out_features, in_features), T.bfloat16)
        with T.Kernel(T.ceildiv(total, threads), threads=threads) as block:
            thread = T.get_thread_binding(0)
            linear = block * threads + thread
            if linear < total:
                row = linear // in_features
                column = linear % in_features
                chunk = row // 64
                lane = row % 64
                packed_lane = T.if_then_else(lane < 32, lane, lane - 32)
                group = column // group_size
                component = column % group_size
                value = T.alloc_local((1,), T.float32)
                value[0] = 0.0
                for stage in T.serial(stages):
                    byte = indices[stage, chunk, group, packed_lane].astype(T.int32)
                    code = T.if_then_else(lane < 32, byte & 15, byte >> 4)
                    value[0] += codebooks[stage, component, code]
                output[row, column] = value[0] * scales[row]
        return output

    return unpack


@lru_cache(maxsize=None)
def compile_rvq_gemv(
    out_features: int, in_features: int, group_size: int, stages: int
):
    """Compile a fused RVQ lookup/dequantization BF16 GEMV.

    The small codebook is repeated per 64-row chunk at load time. This permits
    concatenated Q/K/V payloads to retain distinct codebooks without a branch
    or indirection in the decode kernel.
    """

    if out_features <= 0 or out_features % 64:
        raise ValueError("RVQ GEMV output width must be positive and 64-row aligned")
    if group_size <= 0 or in_features % group_size:
        raise ValueError("RVQ GEMV input width must be divisible by group size")
    tilelang, T = _imports()
    groups = in_features // group_size
    chunks = out_features // 64
    n_partition = 8
    reduce_threads = 16
    group_tiles = T.ceildiv(groups, reduce_threads)

    @tilelang.jit(target="cuda")
    def gemv(
        x: T.Tensor((in_features,), T.bfloat16),
        codebooks: T.Tensor((chunks, stages, group_size, 16), T.float32),
        indices: T.Tensor((stages, chunks, groups, 32), T.uint8),
        scales: T.Tensor((out_features,), T.float32),
    ):
        output = T.empty((out_features,), T.bfloat16)
        with T.Kernel(
            T.ceildiv(out_features, n_partition),
            threads=(reduce_threads, n_partition),
        ) as block:
            lane_k = T.get_thread_binding(0)
            lane_n = T.get_thread_binding(1)
            row = block * n_partition + lane_n
            chunk = row // 64
            row_lane = row % 64
            packed_lane = T.if_then_else(row_lane < 32, row_lane, row_lane - 32)
            partial = T.alloc_local((1,), T.float32)
            reduced = T.alloc_local((1,), T.float32)
            T.clear(partial)
            for group_tile in T.serial(group_tiles):
                group = group_tile * reduce_threads + lane_k
                if group < groups:
                    codes = T.alloc_local((stages,), T.int32)
                    for stage in T.serial(stages):
                        byte = indices[stage, chunk, group, packed_lane].astype(T.int32)
                        codes[stage] = T.if_then_else(
                            row_lane < 32, byte & 15, byte >> 4
                        )
                    for component in T.serial(group_size):
                        value = T.alloc_local((1,), T.float32)
                        value[0] = 0.0
                        for stage in T.serial(stages):
                            value[0] += codebooks[chunk, stage, component, codes[stage]]
                        weight = (value[0] * scales[row]).astype(T.bfloat16)
                        partial[0] += (
                            x[group * group_size + component].astype(T.float32)
                            * weight.astype(T.float32)
                        )
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1), partial[0], True, reduced[0], lane_k,
                        dtype="handle",
                    )
                )
            if lane_k == 0:
                output[row] = reduced[0]
        return output

    return gemv


@lru_cache(maxsize=None)
def compile_ternary_gemv(out_features: int, in_features: int):
    """Compile a fused five-trits-per-byte dequantization BF16 GEMV."""

    tilelang, T = _imports()
    packed_width = (in_features + 4) // 5
    n_partition = 8
    reduce_threads = 32

    @tilelang.jit(target="cuda")
    def gemv(
        x: T.Tensor((in_features,), T.bfloat16),
        packed: T.Tensor((out_features, packed_width), T.uint8),
        scales: T.Tensor((out_features,), T.float32),
    ):
        output = T.empty((out_features,), T.bfloat16)
        with T.Kernel(
            T.ceildiv(out_features, n_partition),
            threads=(reduce_threads, n_partition),
        ) as block:
            lane_k = T.get_thread_binding(0)
            lane_n = T.get_thread_binding(1)
            row = block * n_partition + lane_n
            partial = T.alloc_local((1,), T.float32)
            reduced = T.alloc_local((1,), T.float32)
            T.clear(partial)
            if row < out_features:
                for byte_tile in T.serial(T.ceildiv(packed_width, reduce_threads)):
                    byte_column = byte_tile * reduce_threads + lane_k
                    if byte_column < packed_width:
                        byte = packed[row, byte_column].astype(T.int32)
                        divisor = T.alloc_local((1,), T.int32)
                        divisor[0] = 1
                        for component in T.serial(5):
                            column = byte_column * 5 + component
                            if column < in_features:
                                trit = (byte // divisor[0]) % 3 - 1
                                weight = (
                                    trit.astype(T.float32) * scales[row]
                                ).astype(T.bfloat16)
                                partial[0] += (
                                    x[column].astype(T.float32)
                                    * weight.astype(T.float32)
                                )
                            divisor[0] *= 3
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1), partial[0], True, reduced[0], lane_k,
                        dtype="handle",
                    )
                )
            if lane_k == 0 and row < out_features:
                output[row] = reduced[0]
        return output

    return gemv


@lru_cache(maxsize=None)
def compile_attention(
    query_heads: int, kv_heads: int, head_dim: int, max_context: int
):
    """Compile exact shiftmax decode attention over a circular KV cache."""

    if query_heads % kv_heads:
        raise ValueError("query head count must be divisible by KV head count")
    if head_dim > 1024:
        raise ValueError("attention head width exceeds a CUDA thread block")
    tilelang, T = _imports()
    heads_per_kv = query_heads // kv_heads

    @tilelang.jit(target="cuda")
    def attention(
        query: T.Tensor((query_heads, head_dim), T.bfloat16),
        keys: T.Tensor((kv_heads, max_context, head_dim), T.bfloat16),
        values: T.Tensor((kv_heads, max_context, head_dim), T.bfloat16),
        alpha: T.Tensor((query_heads,), T.float32),
        position: T.Tensor((1,), T.int32),
    ):
        output = T.empty((query_heads, head_dim), T.bfloat16)
        with T.Kernel(query_heads, threads=head_dim) as head:
            lane = T.get_thread_binding(0)
            scores = T.alloc_shared((max_context + 1,), T.float32)
            partial = T.alloc_local((1,), T.float32)
            reduced = T.alloc_local((1,), T.float32)
            total = T.if_then_else(position[0] + 1 < max_context, position[0] + 1, max_context)
            start = T.if_then_else(position[0] + 1 <= max_context, 0, (position[0] + 1) % max_context)
            kv_head = head // heads_per_kv
            for token in T.serial(total):
                slot = (start + token) % max_context
                partial[0] = (
                    query[head, lane].astype(T.float32)
                    * keys[kv_head, slot, lane].astype(T.float32)
                )
                with T.attr(
                    T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
                ):
                    T.evaluate(
                        T.tvm_thread_allreduce(
                            T.uint32(1), partial[0], True, reduced[0], lane,
                            dtype="handle",
                        )
                    )
                if lane == 0:
                    dot = reduced[0].astype(T.bfloat16).astype(T.float32)
                    scores[token] = T.floor(dot * alpha[head])
            T.sync_threads()
            maximum = T.alloc_local((1,), T.float32)
            denominator = T.alloc_local((1,), T.float32)
            if lane == 0:
                maximum[0] = scores[0]
                for token in T.serial(1, total):
                    maximum[0] = T.max(maximum[0], scores[token])
                denominator[0] = 0.0
                for token in T.serial(total):
                    scores[token] = T.exp2(
                        T.max(scores[token] - maximum[0], T.float32(-15.0))
                    )
                    denominator[0] += scores[token]
                scores[max_context] = denominator[0]
            T.sync_threads()
            attended = T.alloc_local((1,), T.float32)
            T.clear(attended)
            for token in T.serial(total):
                slot = (start + token) % max_context
                probability = (
                    scores[token] / scores[max_context]
                ).astype(T.bfloat16)
                attended[0] += (
                    probability.astype(T.float32)
                    * values[kv_head, slot, lane].astype(T.float32)
                )
            output[head, lane] = attended[0]
        return output

    return attention


@lru_cache(maxsize=None)
def compile_gemm(batch_size: int, out_features: int, in_features: int):
    """Compile the batched counterpart of :func:`compile_gemv`."""

    tilelang, T = _imports()
    n_partition, reduce_threads, vector = 8, 16, 8
    block_k = reduce_threads * vector

    @tilelang.jit(target="cuda")
    def gemm(
        x: T.Tensor((batch_size, in_features), T.bfloat16),
        weight: T.Tensor((out_features, in_features), T.bfloat16),
    ):
        output = T.empty((batch_size, out_features), T.bfloat16)
        with T.Kernel(
            T.ceildiv(out_features, n_partition), batch_size,
            threads=(reduce_threads, n_partition),
        ) as (block, token):
            lane_k = T.get_thread_binding(0)
            lane_n = T.get_thread_binding(1)
            x_local = T.alloc_local((vector,), T.bfloat16)
            w_local = T.alloc_local((vector,), T.bfloat16)
            partial = T.alloc_local((1,), T.float32)
            reduced = T.alloc_local((1,), T.float32)
            T.clear(partial)
            for tile_k in T.serial(T.ceildiv(in_features, block_k)):
                for inner in T.vectorized(vector):
                    k = tile_k * block_k + lane_k * vector + inner
                    x_local[inner] = x[token, k]
                    w_local[inner] = weight[block * n_partition + lane_n, k]
                for inner in T.serial(vector):
                    partial[0] += (
                        x_local[inner].astype(T.float32)
                        * w_local[inner].astype(T.float32)
                    )
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), partial[0], True, reduced[0], lane_k, dtype="handle"
                ))
            if lane_k == 0:
                output[token, block * n_partition + lane_n] = reduced[0]
        return output

    return gemm


@lru_cache(maxsize=None)
def compile_rvq_gemm(
    batch_size: int, out_features: int, in_features: int, group_size: int, stages: int
):
    """Compile fused packed RVQ dequantization GEMM for prompt ingestion."""

    if out_features <= 0 or out_features % 64:
        raise ValueError("RVQ GEMM output width must be positive and 64-row aligned")
    tilelang, T = _imports()
    groups, chunks = in_features // group_size, out_features // 64
    n_partition, reduce_threads = 8, 16

    @tilelang.jit(target="cuda")
    def gemm(
        x: T.Tensor((batch_size, in_features), T.bfloat16),
        codebooks: T.Tensor((chunks, stages, group_size, 16), T.float32),
        indices: T.Tensor((stages, chunks, groups, 32), T.uint8),
        scales: T.Tensor((out_features,), T.float32),
    ):
        output = T.empty((batch_size, out_features), T.bfloat16)
        with T.Kernel(
            T.ceildiv(out_features, n_partition), batch_size,
            threads=(reduce_threads, n_partition),
        ) as (block, token):
            lane_k = T.get_thread_binding(0)
            lane_n = T.get_thread_binding(1)
            row = block * n_partition + lane_n
            chunk, row_lane = row // 64, row % 64
            packed_lane = T.if_then_else(row_lane < 32, row_lane, row_lane - 32)
            partial = T.alloc_local((1,), T.float32)
            reduced = T.alloc_local((1,), T.float32)
            T.clear(partial)
            for group_tile in T.serial(T.ceildiv(groups, reduce_threads)):
                group = group_tile * reduce_threads + lane_k
                if group < groups:
                    codes = T.alloc_local((stages,), T.int32)
                    for stage in T.serial(stages):
                        byte = indices[stage, chunk, group, packed_lane].astype(T.int32)
                        codes[stage] = T.if_then_else(row_lane < 32, byte & 15, byte >> 4)
                    for component in T.serial(group_size):
                        value = T.alloc_local((1,), T.float32)
                        value[0] = 0.0
                        for stage in T.serial(stages):
                            value[0] += codebooks[chunk, stage, component, codes[stage]]
                        weight = (value[0] * scales[row]).astype(T.bfloat16)
                        partial[0] += (x[token, group * group_size + component].astype(T.float32)
                                       * weight.astype(T.float32))
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), partial[0], True, reduced[0], lane_k, dtype="handle"
                ))
            if lane_k == 0:
                output[token, row] = reduced[0]
        return output

    return gemm


@lru_cache(maxsize=None)
def compile_ternary_gemm(batch_size: int, out_features: int, in_features: int):
    """Compile fused packed base-3 dequantization GEMM for prompt ingestion."""

    tilelang, T = _imports()
    packed_width = (in_features + 4) // 5
    n_partition, reduce_threads = 8, 32

    @tilelang.jit(target="cuda")
    def gemm(
        x: T.Tensor((batch_size, in_features), T.bfloat16),
        packed: T.Tensor((out_features, packed_width), T.uint8),
        scales: T.Tensor((out_features,), T.float32),
    ):
        output = T.empty((batch_size, out_features), T.bfloat16)
        with T.Kernel(
            T.ceildiv(out_features, n_partition), batch_size,
            threads=(reduce_threads, n_partition),
        ) as (block, token):
            lane_k = T.get_thread_binding(0)
            lane_n = T.get_thread_binding(1)
            row = block * n_partition + lane_n
            partial = T.alloc_local((1,), T.float32)
            reduced = T.alloc_local((1,), T.float32)
            T.clear(partial)
            if row < out_features:
                for byte_tile in T.serial(T.ceildiv(packed_width, reduce_threads)):
                    byte_column = byte_tile * reduce_threads + lane_k
                    if byte_column < packed_width:
                        byte = packed[row, byte_column].astype(T.int32)
                        divisor = T.alloc_local((1,), T.int32)
                        divisor[0] = 1
                        for component in T.serial(5):
                            column = byte_column * 5 + component
                            if column < in_features:
                                trit = (byte // divisor[0]) % 3 - 1
                                weight = (trit.astype(T.float32) * scales[row]).astype(T.bfloat16)
                                partial[0] += (x[token, column].astype(T.float32)
                                               * weight.astype(T.float32))
                            divisor[0] *= 3
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), partial[0], True, reduced[0], lane_k, dtype="handle"
                ))
            if lane_k == 0 and row < out_features:
                output[token, row] = reduced[0]
        return output

    return gemm


@lru_cache(maxsize=None)
def compile_prefill_attention(
    tokens: int, query_heads: int, kv_heads: int, head_dim: int
):
    """Compile causal exact-shiftmax attention for a fresh prompt."""

    tilelang, T = _imports()
    heads_per_kv = query_heads // kv_heads

    @tilelang.jit(target="cuda")
    def attention(
        query: T.Tensor((tokens, query_heads, head_dim), T.bfloat16),
        keys: T.Tensor((tokens, kv_heads, head_dim), T.bfloat16),
        values: T.Tensor((tokens, kv_heads, head_dim), T.bfloat16),
        alpha: T.Tensor((query_heads,), T.float32),
    ):
        output = T.empty((tokens, query_heads, head_dim), T.bfloat16)
        with T.Kernel(query_heads, tokens, threads=head_dim) as (head, query_token):
            lane = T.get_thread_binding(0)
            scores = T.alloc_shared((tokens + 1,), T.float32)
            partial = T.alloc_local((1,), T.float32)
            reduced = T.alloc_local((1,), T.float32)
            kv_head = head // heads_per_kv
            for key_token in T.serial(query_token + 1):
                partial[0] = (query[query_token, head, lane].astype(T.float32)
                              * keys[key_token, kv_head, lane].astype(T.float32))
                with T.attr(
                    T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                    "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
                ):
                    T.evaluate(T.tvm_thread_allreduce(
                        T.uint32(1), partial[0], True, reduced[0], lane, dtype="handle"
                    ))
                if lane == 0:
                    dot = reduced[0].astype(T.bfloat16).astype(T.float32)
                    scores[key_token] = T.floor(dot * alpha[head])
            T.sync_threads()
            if lane == 0:
                maximum = T.alloc_local((1,), T.float32)
                maximum[0] = scores[0]
                for key_token in T.serial(1, query_token + 1):
                    maximum[0] = T.max(maximum[0], scores[key_token])
                denominator = T.alloc_local((1,), T.float32)
                denominator[0] = 0.0
                for key_token in T.serial(query_token + 1):
                    scores[key_token] = T.exp2(T.max(
                        scores[key_token] - maximum[0], T.float32(-15.0)
                    ))
                    denominator[0] += scores[key_token]
                scores[tokens] = denominator[0]
            T.sync_threads()
            attended = T.alloc_local((1,), T.float32)
            T.clear(attended)
            for key_token in T.serial(query_token + 1):
                probability = (scores[key_token] / scores[tokens]).astype(T.bfloat16)
                attended[0] += (probability.astype(T.float32)
                                * values[key_token, kv_head, lane].astype(T.float32))
            output[query_token, head, lane] = attended[0]
        return output

    return attention


@lru_cache(maxsize=None)
def compile_fingerprint_unpack(features: int):
    """Expand one MSB-first bit-packed fingerprint to BF16 signs."""

    if features % 8:
        raise ValueError("fingerprint width must be divisible by eight")
    tilelang, T = _imports()
    packed_width = features // 8
    threads = min(features, 256)

    @tilelang.jit(target="cuda")
    def unpack(packed: T.Tensor((packed_width,), T.uint8)):
        output = T.empty((features,), T.bfloat16)
        with T.Kernel(T.ceildiv(features, threads), threads=threads) as block:
            lane = T.get_thread_binding(0)
            feature = block * threads + lane
            if feature < features:
                byte = packed[feature // 8].astype(T.int32)
                bit = (byte >> (7 - feature % 8)) & 1
                output[feature] = (bit * 2 - 1).astype(T.bfloat16)
        return output

    return unpack


@lru_cache(maxsize=None)
def compile_fingerprint_unpack_batch(batch_size: int, vocabulary: int, features: int):
    """Gather and expand a batch of MSB-first packed fingerprints."""

    if features % 8:
        raise ValueError("fingerprint width must be divisible by eight")
    tilelang, T = _imports()
    packed_width = features // 8
    threads = min(features, 256)

    @tilelang.jit(target="cuda")
    def unpack(
        packed: T.Tensor((vocabulary, packed_width), T.uint8),
        indices: T.Tensor((batch_size,), T.int64),
    ):
        output = T.empty((batch_size, features), T.bfloat16)
        with T.Kernel(
            T.ceildiv(features, threads), batch_size, threads=threads
        ) as (block, token):
            lane = T.get_thread_binding(0)
            feature = block * threads + lane
            if feature < features:
                byte = packed[indices[token], feature // 8].astype(T.int32)
                bit = (byte >> (7 - feature % 8)) & 1
                output[token, feature] = (bit * 2 - 1).astype(T.bfloat16)
        return output

    return unpack


@lru_cache(maxsize=None)
def compile_fingerprint_logits(vocabulary: int, features: int):
    """Project a BF16 fingerprint vector directly against packed signs."""

    if features % 8:
        raise ValueError("fingerprint width must be divisible by eight")
    tilelang, T = _imports()
    packed_width = features // 8
    threads = 32
    bytes_per_lane = T.ceildiv(packed_width, threads)
    normalization = features ** -0.5

    @tilelang.jit(target="cuda")
    def logits(
        projected: T.Tensor((features,), T.bfloat16),
        packed: T.Tensor((vocabulary, packed_width), T.uint8),
        bias: T.Tensor((vocabulary,), T.float32),
    ):
        output = T.empty((vocabulary,), T.float32)
        with T.Kernel(vocabulary, threads=threads) as token:
            lane = T.get_thread_binding(0)
            partial = T.alloc_local((1,), T.float32)
            reduced = T.alloc_local((1,), T.float32)
            T.clear(partial)
            for byte_tile in T.serial(bytes_per_lane):
                byte_index = byte_tile * threads + lane
                if byte_index < packed_width:
                    byte = packed[token, byte_index].astype(T.int32)
                    for component in T.serial(8):
                        bit = (byte >> (7 - component)) & 1
                        sign = bit * 2 - 1
                        partial[0] += (
                            projected[byte_index * 8 + component].astype(T.float32)
                            * sign.astype(T.float32)
                        )
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), partial[0], True, reduced[0], lane, dtype="handle"
                ))
            if lane == 0:
                output[token] = reduced[0] * normalization + bias[token]
        return output

    return logits


@lru_cache(maxsize=None)
def compile_rms_norm(rows: int, width: int, epsilon: float = 1e-6):
    """Compile FP32-accumulating BF16 RMSNorm for flattened rows."""

    if rows < 1 or width < 1:
        raise ValueError("RMSNorm dimensions must be positive")
    tilelang, T = _imports()
    threads = min(256, 1 << (width - 1).bit_length())

    @tilelang.jit(target="cuda")
    def rms_norm(
        x: T.Tensor((rows, width), T.bfloat16),
        weight: T.Tensor((width,), T.float32),
    ):
        output = T.empty((rows, width), T.bfloat16)
        with T.Kernel(rows, threads=threads) as row:
            lane = T.get_thread_binding(0)
            partial = T.alloc_local((1,), T.float32)
            reduced = T.alloc_local((1,), T.float32)
            T.clear(partial)
            for tile in T.serial(T.ceildiv(width, threads)):
                column = tile * threads + lane
                if column < width:
                    value = x[row, column].astype(T.float32)
                    partial[0] += value * value
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), partial[0], True, reduced[0], lane, dtype="handle"
                ))
            scale = T.rsqrt(reduced[0] / width + epsilon)
            for tile in T.serial(T.ceildiv(width, threads)):
                column = tile * threads + lane
                if column < width:
                    normalized = (
                        x[row, column].astype(T.float32) * scale
                    ).astype(T.bfloat16)
                    output[row, column] = normalized * weight[column].astype(T.bfloat16)
        return output

    return rms_norm


@lru_cache(maxsize=None)
def compile_power_of_two_quantize(rows: int, width: int):
    """Compile the exact per-row BF16 power-of-two activation quantizer."""

    tilelang, T = _imports()
    threads = min(256, 1 << (width - 1).bit_length())

    @tilelang.jit(target="cuda")
    def quantize(x: T.Tensor((rows, width), T.bfloat16)):
        output = T.empty((rows, width), T.bfloat16)
        with T.Kernel(rows, threads=threads) as row:
            lane = T.get_thread_binding(0)
            partial = T.alloc_local((1,), T.float32)
            reduced = T.alloc_local((1,), T.float32)
            if lane < width:
                partial[0] = T.abs(x[row, lane]).astype(T.float32)
            else:
                partial[0] = 0.0
            with T.attr(
                T.comm_reducer(lambda a, b: T.max(a, b), [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), partial[0], True, reduced[0], lane, dtype="handle"
                ))
            maximum = T.max(reduced[0], T.float32(1e-6)).astype(T.bfloat16)
            ratio = (maximum / T.bfloat16(127.0)).astype(T.float32)
            logarithm = T.log2(ratio).astype(T.bfloat16).astype(T.float32)
            exponent = T.ceil(logarithm).astype(T.bfloat16).astype(T.float32)
            scale = T.exp2(exponent).astype(T.bfloat16)
            if lane < width:
                rounded = T.round((x[row, lane] / scale).astype(T.float32))
                clipped = T.min(
                    T.max(rounded, T.float32(-127.0)), T.float32(127.0)
                ).astype(T.bfloat16)
                output[row, lane] = clipped * scale
        return output

    return quantize


@lru_cache(maxsize=None)
def compile_rope(heads: int, head_dim: int):
    """Compile BF16 RoPE while preserving the reference operation order."""

    tilelang, T = _imports()
    threads = min(256, 1 << (head_dim - 1).bit_length())

    @tilelang.jit(target="cuda")
    def rope(
        x: T.Tensor((heads, head_dim), T.bfloat16),
        cosine: T.Tensor((head_dim // 2,), T.bfloat16),
        sine: T.Tensor((head_dim // 2,), T.bfloat16),
    ):
        output = T.empty((heads, head_dim), T.bfloat16)
        with T.Kernel(heads, threads=threads) as head:
            lane = T.get_thread_binding(0)
            if lane < head_dim:
                pair = lane // 2
                even = x[head, pair * 2]
                odd = x[head, pair * 2 + 1]
                even_cosine = even * cosine[pair]
                odd_sine = odd * sine[pair]
                even_sine = even * sine[pair]
                odd_cosine = odd * cosine[pair]
                output[head, lane] = T.if_then_else(
                    lane % 2 == 0,
                    even_cosine - odd_sine,
                    even_sine + odd_cosine,
                )
        return output

    return rope




class TileLangLinear:
    """Shape-cached callable used for every decode-time projection."""

    def __call__(self, x, weight):
        if isinstance(weight, PackedRVQWeight):
            kernel = compile_rvq_gemv(
                weight.out_features, weight.in_features, weight.group_size,
                weight.stages,
            )
            return kernel(
                x.contiguous(), weight.codebooks, weight.indices, weight.scales
            )
        if isinstance(weight, PackedTernaryWeight):
            kernel = compile_ternary_gemv(weight.out_features, weight.in_features)
            return kernel(x.contiguous(), weight.packed, weight.scales)
        kernel = compile_gemv(weight.shape[0], weight.shape[1])
        return kernel(x.contiguous(), weight)

    def batch(self, x, weight):
        batch_size = x.shape[0]
        if isinstance(weight, PackedRVQWeight):
            kernel = compile_rvq_gemm(
                batch_size, weight.out_features, weight.in_features,
                weight.group_size, weight.stages,
            )
            return kernel(x.contiguous(), weight.codebooks, weight.indices, weight.scales)
        if isinstance(weight, PackedTernaryWeight):
            kernel = compile_ternary_gemm(
                batch_size, weight.out_features, weight.in_features
            )
            return kernel(x.contiguous(), weight.packed, weight.scales)
        return compile_gemm(batch_size, weight.shape[0], weight.shape[1])(
            x.contiguous(), weight
        )


class TorchLinear:
    """Reference backend used by parity tests and kernel bring-up."""

    def __call__(self, x, weight):
        import torch.nn.functional as functional

        return functional.linear(x, weight)
