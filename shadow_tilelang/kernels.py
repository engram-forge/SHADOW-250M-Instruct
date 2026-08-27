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
class DenseDecodeRVQWeight(PackedRVQWeight):
    """Dense exact-decode view paired with packed RVQ prefill storage."""

    dense: object


@dataclass(frozen=True)
class PackedTernaryWeight:
    """CUDA-resident 2-bit ternary payload consumed directly by GEMV kernels."""

    packed: object
    scales: object
    out_features: int
    in_features: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.out_features, self.in_features


@dataclass(frozen=True)
class InterleavedTernaryWeight(PackedTernaryWeight):
    """Packed ternary pair layout for exact fused SwiGLU decode."""

    paired: object


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
    # Every SHADOW deployment dimension is divisible by 128. Four output
    # lanes times 32 K-reduction lanes fills one 128-thread CUDA block and
    # shortens each thread's serial loop for the wide BF16 FFN projections.
    n_partition = 4
    reduce_threads = 32
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
def compile_rvq_dense_gemv(out_features: int, in_features: int):
    """Compile a dense BF16 GEMV with RVQ's exact reduction order."""

    tilelang, T = _imports()
    n_partition, reduce_threads, vector = 8, 16, 8
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
            row = block * n_partition + lane_n
            partial = T.alloc_local((1,), T.float32)
            reduced = T.alloc_local((1,), T.float32)
            T.clear(partial)
            for tile_k in T.serial(T.ceildiv(in_features, block_k)):
                for inner in T.serial(vector):
                    column = tile_k * block_k + lane_k * vector + inner
                    partial[0] += (
                        x[column].astype(T.float32)
                        * weight[row, column].astype(T.float32)
                    )
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), partial[0], True, reduced[0], lane_k,
                    dtype="handle",
                ))
            if lane_k == 0:
                output[row] = reduced[0]
        return output

    return gemv


@lru_cache(maxsize=None)
def compile_rvq_dense_gemv_split_silu(
    out_features: int, input_features: int
):
    """Dense exact-order structural projection with a BF16 SiLU epilogue."""

    tilelang, T = _imports()
    in_features = input_features * 2
    n_partition, reduce_threads, vector = 8, 16, 8
    block_k = reduce_threads * vector

    @tilelang.jit(target="cuda")
    def gemv(
        left: T.Tensor((input_features,), T.bfloat16),
        right: T.Tensor((input_features,), T.bfloat16),
        weight: T.Tensor((out_features, in_features), T.bfloat16),
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
            for tile_k in T.serial(T.ceildiv(in_features, block_k)):
                for inner in T.serial(vector):
                    column = tile_k * block_k + lane_k * vector + inner
                    activation = T.if_then_else(
                        column < input_features,
                        left[column], right[column - input_features],
                    )
                    partial[0] += (
                        activation.astype(T.float32)
                        * weight[row, column].astype(T.float32)
                    )
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), partial[0], True, reduced[0], lane_k,
                    dtype="handle",
                ))
            if lane_k == 0:
                value = reduced[0].astype(T.bfloat16).astype(T.float32)
                output[row] = value / (T.float32(1) + T.exp(-value))
        return output

    return gemv


@lru_cache(maxsize=None)
def compile_rvq_dense_gemv_residual(out_features: int, in_features: int):
    """Dense exact-order RVQ projection with a BF16 residual epilogue."""

    tilelang, T = _imports()
    n_partition, reduce_threads, vector = 8, 16, 8
    block_k = reduce_threads * vector

    @tilelang.jit(target="cuda")
    def gemv(
        x: T.Tensor((in_features,), T.bfloat16),
        residual: T.Tensor((out_features,), T.bfloat16),
        weight: T.Tensor((out_features, in_features), T.bfloat16),
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
            for tile_k in T.serial(T.ceildiv(in_features, block_k)):
                for inner in T.serial(vector):
                    column = tile_k * block_k + lane_k * vector + inner
                    partial[0] += (
                        x[column].astype(T.float32)
                        * weight[row, column].astype(T.float32)
                    )
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), partial[0], True, reduced[0], lane_k,
                    dtype="handle",
                ))
            if lane_k == 0:
                projected = reduced[0].astype(T.bfloat16)
                output[row] = residual[row] + projected
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
        codebooks: T.Tensor((chunks, group_size, 256), T.float32),
        indices: T.Tensor((chunks, 64, groups), T.uint8),
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
            partial = T.alloc_local((1,), T.float32)
            reduced = T.alloc_local((1,), T.float32)
            T.clear(partial)
            row_scale = scales[row]
            for group_tile in T.serial(group_tiles):
                group = group_tile * reduce_threads + lane_k
                if group < groups:
                    pair_code = indices[chunk, row_lane, group].astype(T.int32)
                    for component in T.serial(group_size):
                        value = codebooks[chunk, component, pair_code]
                        weight = (value * row_scale).astype(T.bfloat16)
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
def compile_rvq_gemv_split_silu(
    out_features: int, input_features: int, group_size: int, stages: int
):
    """Project two contiguous logical inputs and apply exact BF16 SiLU."""

    in_features = input_features * 2
    if out_features <= 0 or out_features % 64:
        raise ValueError("RVQ GEMV output width must be positive and 64-row aligned")
    tilelang, T = _imports()
    groups, chunks = in_features // group_size, out_features // 64
    n_partition, reduce_threads = 8, 16

    @tilelang.jit(target="cuda")
    def gemv(
        left: T.Tensor((input_features,), T.bfloat16),
        right: T.Tensor((input_features,), T.bfloat16),
        codebooks: T.Tensor((chunks, group_size, 256), T.float32),
        indices: T.Tensor((chunks, 64, groups), T.uint8),
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
            chunk, row_lane = row // 64, row % 64
            partial = T.alloc_local((1,), T.float32)
            reduced = T.alloc_local((1,), T.float32)
            T.clear(partial)
            row_scale = scales[row]
            for group_tile in T.serial(T.ceildiv(groups, reduce_threads)):
                group = group_tile * reduce_threads + lane_k
                if group < groups:
                    pair_code = indices[chunk, row_lane, group].astype(T.int32)
                    for component in T.serial(group_size):
                        column = group * group_size + component
                        activation = T.if_then_else(
                            column < input_features,
                            left[column],
                            right[column - input_features],
                        )
                        value = codebooks[chunk, component, pair_code]
                        weight = (value * row_scale).astype(T.bfloat16)
                        partial[0] += (
                            activation.astype(T.float32) * weight.astype(T.float32)
                        )
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), partial[0], True, reduced[0], lane_k,
                    dtype="handle",
                ))
            if lane_k == 0:
                value = reduced[0].astype(T.bfloat16).astype(T.float32)
                output[row] = value / (T.float32(1) + T.exp(-value))
        return output

    return gemv


@lru_cache(maxsize=None)
def compile_rvq_gemv_residual(
    out_features: int, in_features: int, group_size: int, stages: int
):
    """Compile RVQ GEMV with an exact BF16 residual epilogue."""

    if out_features <= 0 or out_features % 64:
        raise ValueError("RVQ GEMV output width must be positive and 64-row aligned")
    tilelang, T = _imports()
    groups, chunks = in_features // group_size, out_features // 64
    n_partition, reduce_threads = 8, 16

    @tilelang.jit(target="cuda")
    def gemv(
        x: T.Tensor((in_features,), T.bfloat16),
        residual: T.Tensor((out_features,), T.bfloat16),
        codebooks: T.Tensor((chunks, group_size, 256), T.float32),
        indices: T.Tensor((chunks, 64, groups), T.uint8),
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
            chunk, row_lane = row // 64, row % 64
            partial = T.alloc_local((1,), T.float32)
            reduced = T.alloc_local((1,), T.float32)
            T.clear(partial)
            row_scale = scales[row]
            for group_tile in T.serial(T.ceildiv(groups, reduce_threads)):
                group = group_tile * reduce_threads + lane_k
                if group < groups:
                    pair_code = indices[chunk, row_lane, group].astype(T.int32)
                    for component in T.serial(group_size):
                        column = group * group_size + component
                        value = codebooks[chunk, component, pair_code]
                        weight = (value * row_scale).astype(T.bfloat16)
                        partial[0] += (
                            x[column].astype(T.float32) * weight.astype(T.float32)
                        )
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), partial[0], True, reduced[0], lane_k,
                    dtype="handle",
                ))
            if lane_k == 0:
                projected = reduced[0].astype(T.bfloat16)
                output[row] = residual[row] + projected
        return output

    return gemv


@lru_cache(maxsize=None)
def compile_rvq_gemv_gated_residual(
    out_features: int, in_features: int, group_size: int, stages: int
):
    """Compile RVQ GEMV with BF16 input gate and residual epilogues."""

    if out_features <= 0 or out_features % 64:
        raise ValueError("RVQ GEMV output width must be positive and 64-row aligned")
    tilelang, T = _imports()
    groups, chunks = in_features // group_size, out_features // 64
    n_partition, reduce_threads = 8, 16

    @tilelang.jit(target="cuda")
    def gemv(
        x: T.Tensor((in_features,), T.bfloat16),
        gate: T.Tensor((in_features,), T.bfloat16),
        residual: T.Tensor((out_features,), T.bfloat16),
        codebooks: T.Tensor((chunks, group_size, 256), T.float32),
        indices: T.Tensor((chunks, 64, groups), T.uint8),
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
            chunk, row_lane = row // 64, row % 64
            partial = T.alloc_local((1,), T.float32)
            reduced = T.alloc_local((1,), T.float32)
            T.clear(partial)
            row_scale = scales[row]
            for group_tile in T.serial(T.ceildiv(groups, reduce_threads)):
                group = group_tile * reduce_threads + lane_k
                if group < groups:
                    pair_code = indices[chunk, row_lane, group].astype(T.int32)
                    for component in T.serial(group_size):
                        column = group * group_size + component
                        value = codebooks[chunk, component, pair_code]
                        weight = (value * row_scale).astype(T.bfloat16)
                        gated = (x[column] * gate[column]).astype(T.bfloat16)
                        partial[0] += gated.astype(T.float32) * weight.astype(T.float32)
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), partial[0], True, reduced[0], lane_k, dtype="handle"
                ))
            if lane_k == 0:
                projected = reduced[0].astype(T.bfloat16)
                output[row] = residual[row] + projected
        return output

    return gemv


@lru_cache(maxsize=None)
def compile_ternary_gemv(out_features: int, in_features: int):
    """Compile fused five-2-bit-trits-per-word BF16 GEMV."""

    tilelang, T = _imports()
    reduction_width = (in_features + 4) // 5
    packed_width = reduction_width
    n_partition = 8
    reduce_threads = 32

    @tilelang.jit(target="cuda")
    def gemv(
        x: T.Tensor((in_features,), T.bfloat16),
        packed: T.Tensor((out_features, packed_width), T.uint16),
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
                row_scale = scales[row].astype(T.bfloat16)
                for group_tile in T.serial(T.ceildiv(reduction_width, reduce_threads)):
                    group = group_tile * reduce_threads + lane_k
                    if group < reduction_width:
                        word = packed[row, group].astype(T.int32)
                        for component in T.unroll(5):
                            column = group * 5 + component
                            if column < in_features:
                                trit = ((word >> (component * 2)) & 3) - 1
                                partial[0] += (
                                    x[column].astype(T.float32)
                                    * trit.astype(T.float32)
                                )
                partial[0] *= row_scale.astype(T.float32)
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
def compile_ternary_swiglu(out_features: int, in_features: int):
    """Project paired up/gate ternary rows and emit exact BF16 SwiGLU."""

    if out_features % 2:
        raise ValueError("SwiGLU projection output width must be even")
    tilelang, T = _imports()
    hidden_features = out_features // 2
    packed_width = (in_features + 4) // 5
    n_partition, reduce_threads = 8, 32

    @tilelang.jit(target="cuda")
    def swiglu(
        x: T.Tensor((in_features,), T.bfloat16),
        packed: T.Tensor((out_features, packed_width), T.uint16),
        scales: T.Tensor((out_features,), T.float32),
    ):
        output = T.empty((hidden_features,), T.bfloat16)
        with T.Kernel(
            T.ceildiv(hidden_features, n_partition),
            threads=(reduce_threads, n_partition),
        ) as block:
            lane_k = T.get_thread_binding(0)
            lane_n = T.get_thread_binding(1)
            row = block * n_partition + lane_n
            up_partial = T.alloc_local((1,), T.float32)
            gate_partial = T.alloc_local((1,), T.float32)
            up_reduced = T.alloc_local((1,), T.float32)
            gate_reduced = T.alloc_local((1,), T.float32)
            T.clear(up_partial)
            T.clear(gate_partial)
            if row < hidden_features:
                up_scale = scales[row].astype(T.bfloat16)
                gate_scale = scales[hidden_features + row].astype(T.bfloat16)
                for group_tile in T.serial(T.ceildiv(packed_width, reduce_threads)):
                    group = group_tile * reduce_threads + lane_k
                    if group < packed_width:
                        up_word = packed[row, group].astype(T.int32)
                        gate_word = packed[hidden_features + row, group].astype(T.int32)
                        for component in T.unroll(5):
                            column = group * 5 + component
                            if column < in_features:
                                activation = x[column].astype(T.float32)
                                up_trit = ((up_word >> (component * 2)) & 3) - 1
                                gate_trit = ((gate_word >> (component * 2)) & 3) - 1
                                up_partial[0] += activation * up_trit.astype(T.float32)
                                gate_partial[0] += activation * gate_trit.astype(T.float32)
                up_partial[0] *= up_scale.astype(T.float32)
                gate_partial[0] *= gate_scale.astype(T.float32)
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), up_partial[0], True, up_reduced[0], lane_k,
                    dtype="handle",
                ))
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), gate_partial[0], True, gate_reduced[0], lane_k,
                    dtype="handle",
                ))
            if lane_k == 0 and row < hidden_features:
                up = up_reduced[0].astype(T.bfloat16)
                gate = gate_reduced[0].astype(T.bfloat16).astype(T.float32)
                silu = (
                    gate / (T.float32(1) + T.exp(-gate))
                ).astype(T.bfloat16)
                output[row] = up * silu
        return output

    return swiglu


@lru_cache(maxsize=None)
def compile_interleaved_ternary_swiglu(out_features: int, in_features: int):
    """Project paired up/gate words from one interleaved 32-bit load."""

    if out_features % 2:
        raise ValueError("SwiGLU projection output width must be even")
    tilelang, T = _imports()
    hidden_features = out_features // 2
    packed_width = (in_features + 4) // 5
    n_partition, reduce_threads = 8, 32

    @tilelang.jit(target="cuda")
    def swiglu(
        x: T.Tensor((in_features,), T.bfloat16),
        paired: T.Tensor((hidden_features, packed_width), T.uint32),
        scales: T.Tensor((out_features,), T.float32),
    ):
        output = T.empty((hidden_features,), T.bfloat16)
        with T.Kernel(
            T.ceildiv(hidden_features, n_partition),
            threads=(reduce_threads, n_partition),
        ) as block:
            lane_k = T.get_thread_binding(0)
            lane_n = T.get_thread_binding(1)
            row = block * n_partition + lane_n
            up_partial = T.alloc_local((1,), T.float32)
            gate_partial = T.alloc_local((1,), T.float32)
            up_reduced = T.alloc_local((1,), T.float32)
            gate_reduced = T.alloc_local((1,), T.float32)
            T.clear(up_partial)
            T.clear(gate_partial)
            if row < hidden_features:
                up_scale = scales[row].astype(T.bfloat16)
                gate_scale = scales[hidden_features + row].astype(T.bfloat16)
                for group_tile in T.serial(T.ceildiv(packed_width, reduce_threads)):
                    group = group_tile * reduce_threads + lane_k
                    if group < packed_width:
                        pair = paired[row, group]
                        up_word = (pair & T.uint32(65535)).astype(T.int32)
                        gate_word = (pair >> T.uint32(16)).astype(T.int32)
                        for component in T.unroll(5):
                            column = group * 5 + component
                            if column < in_features:
                                activation = x[column].astype(T.float32)
                                up_trit = ((up_word >> (component * 2)) & 3) - 1
                                gate_trit = ((gate_word >> (component * 2)) & 3) - 1
                                up_partial[0] += activation * up_trit.astype(T.float32)
                                gate_partial[0] += activation * gate_trit.astype(T.float32)
                up_partial[0] *= up_scale.astype(T.float32)
                gate_partial[0] *= gate_scale.astype(T.float32)
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), up_partial[0], True, up_reduced[0], lane_k,
                    dtype="handle",
                ))
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), gate_partial[0], True, gate_reduced[0], lane_k,
                    dtype="handle",
                ))
            if lane_k == 0 and row < hidden_features:
                up = up_reduced[0].astype(T.bfloat16)
                gate = gate_reduced[0].astype(T.bfloat16).astype(T.float32)
                silu = (
                    gate / (T.float32(1) + T.exp(-gate))
                ).astype(T.bfloat16)
                output[row] = up * silu
        return output

    return swiglu


@lru_cache(maxsize=None)
def compile_ternary_gemv_residual(out_features: int, in_features: int):
    """Compile 2-bit ternary GEMV with a BF16 residual output epilogue."""

    tilelang, T = _imports()
    reduction_width = (in_features + 4) // 5
    packed_width = reduction_width
    n_partition, reduce_threads = 8, 32

    @tilelang.jit(target="cuda")
    def gemv(
        x: T.Tensor((in_features,), T.bfloat16),
        residual: T.Tensor((out_features,), T.bfloat16),
        packed: T.Tensor((out_features, packed_width), T.uint16),
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
                row_scale = scales[row].astype(T.bfloat16)
                for group_tile in T.serial(T.ceildiv(reduction_width, reduce_threads)):
                    group = group_tile * reduce_threads + lane_k
                    if group < reduction_width:
                        word = packed[row, group].astype(T.int32)
                        for component in T.unroll(5):
                            column = group * 5 + component
                            if column < in_features:
                                trit = ((word >> (component * 2)) & 3) - 1
                                partial[0] += (x[column].astype(T.float32)
                                               * trit.astype(T.float32))
                partial[0] *= row_scale.astype(T.float32)
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), partial[0], True, reduced[0], lane_k, dtype="handle"
                ))
            if lane_k == 0 and row < out_features:
                projected = reduced[0].astype(T.bfloat16)
                output[row] = residual[row] + projected
        return output

    return gemv


@lru_cache(maxsize=None)
def compile_attention_cache_update(
    kv_heads: int, head_dim: int, max_context: int
):
    """Quantize and store the current K/V pair once per KV head."""

    tilelang, T = _imports()

    @tilelang.jit(target="cuda")
    def update(
        current_key: T.Tensor((kv_heads, head_dim), T.bfloat16),
        current_value_unquantized: T.Tensor((kv_heads, head_dim), T.bfloat16),
        keys: T.Tensor((kv_heads, max_context, head_dim), T.bfloat16),
        values: T.Tensor((kv_heads, max_context, head_dim), T.bfloat16),
        position: T.Tensor((1,), T.int32),
    ):
        with T.Kernel(kv_heads, threads=head_dim) as kv_head:
            lane = T.get_thread_binding(0)
            value_maximum = T.alloc_local((1,), T.float32)
            value_reduced = T.alloc_local((1,), T.float32)
            value_maximum[0] = T.abs(
                current_value_unquantized[kv_head, lane]
            ).astype(T.float32)
            with T.attr(
                T.comm_reducer(lambda a, b: T.max(a, b), [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), value_maximum[0], True, value_reduced[0], lane,
                    dtype="handle",
                ))
            maximum = T.max(value_reduced[0], T.float32(1e-6)).astype(T.bfloat16)
            ratio = (maximum / T.bfloat16(127.0)).astype(T.float32)
            logarithm = T.log2(ratio).astype(T.bfloat16).astype(T.float32)
            exponent = T.ceil(logarithm).astype(T.bfloat16).astype(T.float32)
            value_scale = T.exp2(exponent).astype(T.bfloat16)
            rounded = T.round((
                current_value_unquantized[kv_head, lane] / value_scale
            ).astype(T.float32))
            clipped = T.min(
                T.max(rounded, T.float32(-127.0)), T.float32(127.0)
            ).astype(T.bfloat16)
            current_slot = position[0] % max_context
            keys[kv_head, current_slot, lane] = current_key[kv_head, lane]
            values[kv_head, current_slot, lane] = clipped * value_scale

    return update


@lru_cache(maxsize=None)
def compile_attention_scores(
    query_heads: int, kv_heads: int, head_dim: int, max_context: int
):
    """Compute exact decode attention scores with token-parallel blocks."""

    if query_heads % kv_heads:
        raise ValueError("query head count must be divisible by KV head count")
    tilelang, T = _imports()
    heads_per_kv = query_heads // kv_heads
    tokens_per_block = 4
    query_heads_per_block = 6

    @tilelang.jit(target="cuda")
    def attention_scores(
        query: T.Tensor((query_heads, head_dim), T.bfloat16),
        keys: T.Tensor((kv_heads, max_context, head_dim), T.bfloat16),
        alpha: T.Tensor((query_heads,), T.float32),
        position: T.Tensor((1,), T.int32),
    ):
        scores = T.empty((query_heads, max_context), T.float32)
        with T.Kernel(
            kv_heads, T.ceildiv(heads_per_kv, query_heads_per_block),
            T.ceildiv(max_context, tokens_per_block),
            threads=(head_dim, tokens_per_block),
        ) as (kv_head, head_block, token_block):
            lane = T.get_thread_binding(0)
            token_lane = T.get_thread_binding(1)
            token = token_block * tokens_per_block + token_lane
            total = T.if_then_else(
                position[0] + 1 < max_context, position[0] + 1, max_context
            )
            start = T.if_then_else(
                position[0] + 1 <= max_context,
                0, (position[0] + 1) % max_context,
            )
            partial_0 = T.alloc_local((1,), T.float32)
            partial_1 = T.alloc_local((1,), T.float32)
            partial_2 = T.alloc_local((1,), T.float32)
            partial_3 = T.alloc_local((1,), T.float32)
            partial_4 = T.alloc_local((1,), T.float32)
            partial_5 = T.alloc_local((1,), T.float32)
            reduced_0 = T.alloc_local((1,), T.float32)
            reduced_1 = T.alloc_local((1,), T.float32)
            reduced_2 = T.alloc_local((1,), T.float32)
            reduced_3 = T.alloc_local((1,), T.float32)
            reduced_4 = T.alloc_local((1,), T.float32)
            reduced_5 = T.alloc_local((1,), T.float32)
            T.clear(partial_0)
            T.clear(partial_1)
            T.clear(partial_2)
            T.clear(partial_3)
            T.clear(partial_4)
            T.clear(partial_5)
            head_0 = kv_head * heads_per_kv + head_block * query_heads_per_block
            head_1 = head_0 + 1
            head_2 = head_0 + 2
            head_3 = head_0 + 3
            head_4 = head_0 + 4
            head_5 = head_0 + 5
            if token < total:
                slot = (start + token) % max_context
                key = keys[kv_head, slot, lane].astype(T.float32)
                partial_0[0] = query[head_0, lane].astype(T.float32) * key
                partial_1[0] = query[head_1, lane].astype(T.float32) * key
                partial_2[0] = query[head_2, lane].astype(T.float32) * key
                partial_3[0] = query[head_3, lane].astype(T.float32) * key
                partial_4[0] = query[head_4, lane].astype(T.float32) * key
                partial_5[0] = query[head_5, lane].astype(T.float32) * key
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), partial_0[0], True, reduced_0[0], lane,
                    dtype="handle",
                ))
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), partial_1[0], True, reduced_1[0], lane,
                    dtype="handle",
                ))
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), partial_2[0], True, reduced_2[0], lane,
                    dtype="handle",
                ))
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), partial_3[0], True, reduced_3[0], lane,
                    dtype="handle",
                ))
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), partial_4[0], True, reduced_4[0], lane,
                    dtype="handle",
                ))
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), partial_5[0], True, reduced_5[0], lane,
                    dtype="handle",
                ))
            if lane == 0 and token < total:
                dot_0 = reduced_0[0].astype(T.bfloat16).astype(T.float32)
                dot_1 = reduced_1[0].astype(T.bfloat16).astype(T.float32)
                dot_2 = reduced_2[0].astype(T.bfloat16).astype(T.float32)
                dot_3 = reduced_3[0].astype(T.bfloat16).astype(T.float32)
                dot_4 = reduced_4[0].astype(T.bfloat16).astype(T.float32)
                dot_5 = reduced_5[0].astype(T.bfloat16).astype(T.float32)
                scores[head_0, token] = T.floor(dot_0 * alpha[head_0])
                scores[head_1, token] = T.floor(dot_1 * alpha[head_1])
                scores[head_2, token] = T.floor(dot_2 * alpha[head_2])
                scores[head_3, token] = T.floor(dot_3 * alpha[head_3])
                scores[head_4, token] = T.floor(dot_4 * alpha[head_4])
                scores[head_5, token] = T.floor(dot_5 * alpha[head_5])
        return scores

    return attention_scores


@lru_cache(maxsize=None)
def compile_attention_values(
    query_heads: int, kv_heads: int, head_dim: int, max_context: int
):
    """Normalize precomputed scores and accumulate exact decode values."""

    if query_heads % kv_heads:
        raise ValueError("query head count must be divisible by KV head count")
    tilelang, T = _imports()
    heads_per_kv = query_heads // kv_heads
    token_parallel = 16

    @tilelang.jit(target="cuda")
    def attention_values(
        input_scores: T.Tensor((query_heads, max_context), T.float32),
        values: T.Tensor((kv_heads, max_context, head_dim), T.bfloat16),
        position: T.Tensor((1,), T.int32),
    ):
        output = T.empty((query_heads, head_dim), T.bfloat16)
        with T.Kernel(
            query_heads, threads=(head_dim, token_parallel)
        ) as head:
            lane = T.get_thread_binding(0)
            token_lane = T.get_thread_binding(1)
            scores = T.alloc_shared((max_context + 1,), T.float32)
            total = T.if_then_else(
                position[0] + 1 < max_context, position[0] + 1, max_context
            )
            start = T.if_then_else(
                position[0] + 1 <= max_context,
                0, (position[0] + 1) % max_context,
            )
            score_threads = head_dim * token_parallel
            for token_tile in T.serial(T.ceildiv(total, score_threads)):
                token = (
                    token_tile * score_threads + token_lane * head_dim + lane
                )
                if token < total:
                    scores[token] = input_scores[head, token]
            T.sync_threads()
            maximum = T.alloc_local((1,), T.float32)
            denominator = T.alloc_local((1,), T.float32)
            if lane == 0 and token_lane == 0:
                maximum[0] = scores[0]
                for token in T.serial(1, total):
                    maximum[0] = T.max(maximum[0], scores[token])
                scores[max_context] = maximum[0]
            T.sync_threads()
            for token_tile in T.serial(T.ceildiv(total, score_threads)):
                token = (
                    token_tile * score_threads + token_lane * head_dim + lane
                )
                if token < total:
                    scores[token] = T.exp2(T.max(
                        scores[token] - scores[max_context], T.float32(-15.0)
                    ))
            T.sync_threads()
            T.clear(denominator)
            for token_tile in T.serial(T.ceildiv(total, score_threads)):
                token = (
                    token_tile * score_threads + token_lane * head_dim + lane
                )
                if token < total:
                    denominator[0] += scores[token]
            denominator_reduced = T.alloc_local((1,), T.float32)
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), denominator[0], True, denominator_reduced[0],
                    lane, token_lane, dtype="handle",
                ))
            if lane == 0 and token_lane == 0:
                scores[max_context] = denominator_reduced[0]
            T.sync_threads()
            attended_partials = T.alloc_shared(
                (token_parallel, head_dim), T.float32
            )
            attended = T.alloc_local((1,), T.float32)
            T.clear(attended)
            kv_head = head // heads_per_kv
            for token_tile in T.serial(T.ceildiv(total, token_parallel)):
                token = token_tile * token_parallel + token_lane
                if token < total:
                    slot = (start + token) % max_context
                    probability = (
                        scores[token] / scores[max_context]
                    ).astype(T.bfloat16)
                    attended[0] += (
                        probability.astype(T.float32)
                        * values[kv_head, slot, lane].astype(T.float32)
                    )
            attended_partials[token_lane, lane] = attended[0]
            T.sync_threads()
            if token_lane == 0:
                T.clear(attended)
                for segment in T.serial(token_parallel):
                    attended[0] += attended_partials[segment, lane]
                output[head, lane] = attended[0]
        return output

    return attention_values


@lru_cache(maxsize=None)
def compile_attention_probabilities(
    query_heads: int, max_context: int, token_parallel: int = 16
):
    """Normalize split attention scores into exact BF16 probabilities."""

    tilelang, T = _imports()
    head_dim = 64
    score_threads = head_dim * token_parallel

    @tilelang.jit(target="cuda")
    def probabilities(
        input_scores: T.Tensor((query_heads, max_context), T.float32),
        position: T.Tensor((1,), T.int32),
    ):
        output = T.empty((query_heads, max_context), T.bfloat16)
        with T.Kernel(
            query_heads, threads=(head_dim, token_parallel)
        ) as head:
            lane = T.get_thread_binding(0)
            token_lane = T.get_thread_binding(1)
            scores = T.alloc_shared((max_context + 1,), T.float32)
            total = T.if_then_else(
                position[0] + 1 < max_context, position[0] + 1, max_context
            )
            for token_tile in T.serial(T.ceildiv(total, score_threads)):
                token = (
                    token_tile * score_threads + token_lane * head_dim + lane
                )
                if token < total:
                    scores[token] = input_scores[head, token]
            T.sync_threads()
            maximum = T.alloc_local((1,), T.float32)
            maximum[0] = T.float32(-1e30)
            for token_tile in T.serial(T.ceildiv(total, score_threads)):
                token = (
                    token_tile * score_threads + token_lane * head_dim + lane
                )
                if token < total:
                    maximum[0] = T.max(maximum[0], scores[token])
            maximum_reduced = T.alloc_local((1,), T.float32)
            with T.attr(
                T.comm_reducer(lambda a, b: T.max(a, b), [T.float32(-1e30)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), maximum[0], True, maximum_reduced[0],
                    lane, token_lane, dtype="handle",
                ))
            if lane == 0 and token_lane == 0:
                scores[max_context] = maximum_reduced[0]
            T.sync_threads()
            for token_tile in T.serial(T.ceildiv(total, score_threads)):
                token = (
                    token_tile * score_threads + token_lane * head_dim + lane
                )
                if token < total:
                    scores[token] = T.exp2(T.max(
                        scores[token] - scores[max_context], T.float32(-15.0)
                    ))
            T.sync_threads()
            denominator = T.alloc_local((1,), T.float32)
            T.clear(denominator)
            for token_tile in T.serial(T.ceildiv(total, score_threads)):
                token = (
                    token_tile * score_threads + token_lane * head_dim + lane
                )
                if token < total:
                    denominator[0] += scores[token]
            denominator_reduced = T.alloc_local((1,), T.float32)
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), denominator[0], True, denominator_reduced[0],
                    lane, token_lane, dtype="handle",
                ))
            if lane == 0 and token_lane == 0:
                scores[max_context] = denominator_reduced[0]
            T.sync_threads()
            for token_tile in T.serial(T.ceildiv(total, score_threads)):
                token = (
                    token_tile * score_threads + token_lane * head_dim + lane
                )
                if token < total:
                    output[head, token] = (
                        scores[token] / scores[max_context]
                    ).astype(T.bfloat16)
        return output

    return probabilities


@lru_cache(maxsize=None)
def compile_attention_value_partials(
    query_heads: int, kv_heads: int, head_dim: int, max_context: int,
    token_parallel: int = 64,
):
    """Accumulate the proven 16 decode-value token segments in parallel."""

    if query_heads % kv_heads:
        raise ValueError("query head count must be divisible by KV head count")
    tilelang, T = _imports()
    heads_per_kv = query_heads // kv_heads
    query_heads_per_block = 4
    if token_parallel < 1:
        raise ValueError("attention value parallelism must be positive")

    @tilelang.jit(target="cuda")
    def value_partials(
        probability: T.Tensor((query_heads, max_context), T.bfloat16),
        values: T.Tensor((kv_heads, max_context, head_dim), T.bfloat16),
        position: T.Tensor((1,), T.int32),
    ):
        output = T.empty(
            (query_heads, token_parallel, head_dim), T.float32
        )
        with T.Kernel(
            kv_heads, T.ceildiv(heads_per_kv, query_heads_per_block),
            token_parallel, threads=head_dim
        ) as (kv_head, head_block, segment):
            lane = T.get_thread_binding(0)
            head_0 = kv_head * heads_per_kv + head_block * query_heads_per_block
            head_1 = head_0 + 1
            head_2 = head_0 + 2
            head_3 = head_0 + 3
            total = T.if_then_else(
                position[0] + 1 < max_context, position[0] + 1, max_context
            )
            start = T.if_then_else(
                position[0] + 1 <= max_context,
                0, (position[0] + 1) % max_context,
            )
            attended_0 = T.alloc_local((1,), T.float32)
            attended_1 = T.alloc_local((1,), T.float32)
            attended_2 = T.alloc_local((1,), T.float32)
            attended_3 = T.alloc_local((1,), T.float32)
            T.clear(attended_0)
            T.clear(attended_1)
            T.clear(attended_2)
            T.clear(attended_3)
            for token_tile in T.serial(T.ceildiv(total, token_parallel)):
                token = token_tile * token_parallel + segment
                if token < total:
                    slot = (start + token) % max_context
                    value = values[kv_head, slot, lane].astype(T.float32)
                    attended_0[0] += (
                        probability[head_0, token].astype(T.float32) * value
                    )
                    attended_1[0] += (
                        probability[head_1, token].astype(T.float32) * value
                    )
                    attended_2[0] += (
                        probability[head_2, token].astype(T.float32) * value
                    )
                    attended_3[0] += (
                        probability[head_3, token].astype(T.float32) * value
                    )
            output[head_0, segment, lane] = attended_0[0]
            output[head_1, segment, lane] = attended_1[0]
            output[head_2, segment, lane] = attended_2[0]
            output[head_3, segment, lane] = attended_3[0]
        return output

    return value_partials


@lru_cache(maxsize=None)
def compile_attention_value_reduce(
    query_heads: int, head_dim: int, token_parallel: int = 64
):
    """Reduce decode-value segments in the reference segment order."""

    tilelang, T = _imports()
    if token_parallel < 1:
        raise ValueError("attention value parallelism must be positive")

    @tilelang.jit(target="cuda")
    def value_reduce(
        partials: T.Tensor(
            (query_heads, token_parallel, head_dim), T.float32
        ),
    ):
        output = T.empty((query_heads, head_dim), T.bfloat16)
        with T.Kernel(query_heads, threads=head_dim) as head:
            lane = T.get_thread_binding(0)
            attended = T.alloc_local((1,), T.float32)
            T.clear(attended)
            for segment in T.serial(token_parallel):
                attended[0] += partials[head, segment, lane]
            output[head, lane] = attended[0]
        return output

    return value_reduce


@lru_cache(maxsize=None)
def compile_attention_value_reduce_gate(
    query_heads: int, head_dim: int, token_parallel: int = 64
):
    """Reduce value segments and apply the exact BF16 output gate once."""

    tilelang, T = _imports()
    if token_parallel < 1:
        raise ValueError("attention value parallelism must be positive")

    @tilelang.jit(target="cuda")
    def value_reduce_gate(
        partials: T.Tensor(
            (query_heads, token_parallel, head_dim), T.float32
        ),
        gate: T.Tensor((query_heads * head_dim,), T.bfloat16),
    ):
        output = T.empty((query_heads, head_dim), T.bfloat16)
        with T.Kernel(query_heads, threads=head_dim) as head:
            lane = T.get_thread_binding(0)
            attended = T.alloc_local((1,), T.float32)
            T.clear(attended)
            for segment in T.serial(token_parallel):
                attended[0] += partials[head, segment, lane]
            rounded = attended[0].astype(T.bfloat16)
            output[head, lane] = rounded * gate[head * head_dim + lane]
        return output

    return value_reduce_gate


@lru_cache(maxsize=None)
def compile_attention(
    query_heads: int, kv_heads: int, head_dim: int, max_context: int,
    token_parallel: int = 16,
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
        gate: T.Tensor((query_heads * head_dim,), T.bfloat16),
        position: T.Tensor((1,), T.int32),
    ):
        output = T.empty((query_heads, head_dim), T.bfloat16)
        with T.Kernel(
            query_heads, threads=(head_dim, token_parallel)
        ) as head:
            lane = T.get_thread_binding(0)
            token_lane = T.get_thread_binding(1)
            scores = T.alloc_shared((max_context + 1,), T.float32)
            partial = T.alloc_local((1,), T.float32)
            reduced = T.alloc_local((1,), T.float32)
            total = T.if_then_else(position[0] + 1 < max_context, position[0] + 1, max_context)
            start = T.if_then_else(position[0] + 1 <= max_context, 0, (position[0] + 1) % max_context)
            kv_head = head // heads_per_kv
            for token_tile in T.serial(T.ceildiv(total, token_parallel)):
                token = token_tile * token_parallel + token_lane
                T.clear(partial)
                if token < total:
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
                if token < total:
                    if lane == 0:
                        dot = reduced[0].astype(T.bfloat16).astype(T.float32)
                        scores[token] = T.floor(dot * alpha[head])
            T.sync_threads()
            maximum = T.alloc_local((1,), T.float32)
            denominator = T.alloc_local((1,), T.float32)
            if lane == 0 and token_lane == 0:
                maximum[0] = scores[0]
                for token in T.serial(1, total):
                    maximum[0] = T.max(maximum[0], scores[token])
                scores[max_context] = maximum[0]
            T.sync_threads()
            score_threads = head_dim * token_parallel
            for token_tile in T.serial(T.ceildiv(total, score_threads)):
                token = (
                    token_tile * score_threads + token_lane * head_dim + lane
                )
                if token < total:
                    scores[token] = T.exp2(
                        T.max(
                            scores[token] - scores[max_context],
                            T.float32(-15.0),
                        )
                    )
            T.sync_threads()
            if lane == 0 and token_lane == 0:
                denominator[0] = T.float32(0)
            else:
                denominator[0] = T.float32(0)
            for token_tile in T.serial(T.ceildiv(total, score_threads)):
                token = (
                    token_tile * score_threads + token_lane * head_dim + lane
                )
                if token < total:
                    denominator[0] += scores[token]
            denominator_reduced = T.alloc_local((1,), T.float32)
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), denominator[0], True, denominator_reduced[0],
                    lane, token_lane, dtype="handle",
                ))
            if lane == 0 and token_lane == 0:
                scores[max_context] = denominator_reduced[0]
            T.sync_threads()
            attended_partials = T.alloc_shared(
                (token_parallel, head_dim), T.float32
            )
            attended = T.alloc_local((1,), T.float32)
            T.clear(attended)
            for token_tile in T.serial(T.ceildiv(total, token_parallel)):
                token = token_tile * token_parallel + token_lane
                if token < total:
                    slot = (start + token) % max_context
                    probability = (
                        scores[token] / scores[max_context]
                    ).astype(T.bfloat16)
                    attended[0] += (
                        probability.astype(T.float32)
                        * values[kv_head, slot, lane].astype(T.float32)
                    )
            attended_partials[token_lane, lane] = attended[0]
            T.sync_threads()
            if token_lane == 0:
                T.clear(attended)
                for segment in T.serial(token_parallel):
                    attended[0] += attended_partials[segment, lane]
                rounded = attended[0].astype(T.bfloat16)
                output[head, lane] = rounded * gate[head * head_dim + lane]
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
        codebooks: T.Tensor((chunks, group_size, 256), T.float32),
        indices: T.Tensor((chunks, 64, groups), T.uint8),
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
            partial = T.alloc_local((1,), T.float32)
            reduced = T.alloc_local((1,), T.float32)
            T.clear(partial)
            row_scale = scales[row]
            for group_tile in T.serial(T.ceildiv(groups, reduce_threads)):
                group = group_tile * reduce_threads + lane_k
                if group < groups:
                    pair_code = indices[chunk, row_lane, group].astype(T.int32)
                    for component in T.serial(group_size):
                        value = codebooks[chunk, component, pair_code]
                        weight = (value * row_scale).astype(T.bfloat16)
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
    """Compile fused packed 2-bit ternary GEMM for prompt ingestion."""

    tilelang, T = _imports()
    reduction_width = (in_features + 4) // 5
    packed_width = reduction_width
    n_partition, reduce_threads = 8, 32

    @tilelang.jit(target="cuda")
    def gemm(
        x: T.Tensor((batch_size, in_features), T.bfloat16),
        packed: T.Tensor((out_features, packed_width), T.uint16),
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
                row_scale = scales[row].astype(T.bfloat16)
                for group_tile in T.serial(T.ceildiv(reduction_width, reduce_threads)):
                    group = group_tile * reduce_threads + lane_k
                    if group < reduction_width:
                        word = packed[row, group].astype(T.int32)
                        for component in T.unroll(5):
                            column = group * 5 + component
                            if column < in_features:
                                trit = ((word >> (component * 2)) & 3) - 1
                                partial[0] += (x[token, column].astype(T.float32)
                                               * trit.astype(T.float32))
                partial[0] *= row_scale.astype(T.float32)
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
def compile_fingerprint_gather(vocabulary: int, features: int):
    """Gather a CUDA-selected packed fingerprint and expand it to BF16 signs."""

    if features % 8:
        raise ValueError("fingerprint width must be divisible by eight")
    tilelang, T = _imports()
    packed_width = features // 8
    threads = min(features, 256)

    @tilelang.jit(target="cuda")
    def gather(
        packed: T.Tensor((vocabulary, packed_width), T.uint8),
        index: T.Tensor((1,), T.int64),
    ):
        output = T.empty((features,), T.bfloat16)
        with T.Kernel(T.ceildiv(features, threads), threads=threads) as block:
            lane = T.get_thread_binding(0)
            feature = block * threads + lane
            if feature < features:
                byte = packed[index[0], feature // 8].astype(T.int32)
                bit = (byte >> (7 - feature % 8)) & 1
                output[feature] = (bit * 2 - 1).astype(T.bfloat16)
        return output

    return gather


@lru_cache(maxsize=None)
def compile_fingerprint_embedding(
    vocabulary: int, features: int, out_features: int
):
    """Project one packed token fingerprint with dense BF16 weights."""

    if features % 8:
        raise ValueError("fingerprint width must be divisible by eight")
    tilelang, T = _imports()
    packed_width = features // 8
    n_partition, reduce_threads, vector = 8, 16, 8
    block_k = reduce_threads * vector

    @tilelang.jit(target="cuda")
    def embedding(
        packed: T.Tensor((vocabulary, packed_width), T.uint8),
        token: T.Tensor((1,), T.int64),
        weight: T.Tensor((out_features, features), T.bfloat16),
    ):
        output = T.empty((out_features,), T.bfloat16)
        with T.Kernel(
            T.ceildiv(out_features, n_partition),
            threads=(reduce_threads, n_partition),
        ) as block:
            lane_k = T.get_thread_binding(0)
            lane_n = T.get_thread_binding(1)
            partial = T.alloc_local((1,), T.float32)
            reduced = T.alloc_local((1,), T.float32)
            T.clear(partial)
            for tile_k in T.serial(T.ceildiv(features, block_k)):
                for inner in T.serial(vector):
                    feature = tile_k * block_k + lane_k * vector + inner
                    byte = packed[token[0], feature // 8].astype(T.int32)
                    bit = (byte >> (7 - feature % 8)) & 1
                    sign = (bit * 2 - 1).astype(T.bfloat16)
                    value = weight[
                        block * n_partition + lane_n, feature
                    ]
                    partial[0] += sign.astype(T.float32) * value.astype(T.float32)
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), partial[0], True, reduced[0], lane_k,
                    dtype="handle",
                ))
            if lane_k == 0:
                output[block * n_partition + lane_n] = reduced[0]
        return output

    return embedding


@lru_cache(maxsize=None)
def compile_circular_store(max_context: int, width: int):
    """Store one vector at a CUDA-selected circular-cache position."""

    tilelang, T = _imports()
    threads = min(width, 256)

    @tilelang.jit(target="cuda")
    def store(
        value: T.Tensor((width,), T.bfloat16),
        cache: T.Tensor((max_context, width), T.bfloat16),
        position: T.Tensor((1,), T.int32),
    ):
        output = T.empty((1,), T.int32)
        with T.Kernel(T.ceildiv(width, threads), threads=threads) as block:
            lane = T.get_thread_binding(0)
            column = block * threads + lane
            if column < width:
                cache[position[0] % max_context, column] = value[column]
            if block == 0 and lane == 0:
                output[0] = position[0]
        return output

    return store


@lru_cache(maxsize=None)
def compile_residual_circular_store(max_context: int, width: int):
    """Add exact BF16 residuals and store the result in a circular cache."""

    tilelang, T = _imports()
    threads = min(width, 256)

    @tilelang.jit(target="cuda")
    def residual_store(
        residual: T.Tensor((width,), T.bfloat16),
        projected: T.Tensor((width,), T.bfloat16),
        cache: T.Tensor((max_context, width), T.bfloat16),
        position: T.Tensor((1,), T.int32),
    ):
        output = T.empty((width,), T.bfloat16)
        with T.Kernel(T.ceildiv(width, threads), threads=threads) as block:
            lane = T.get_thread_binding(0)
            column = block * threads + lane
            if column < width:
                value = residual[column] + projected[column]
                output[column] = value
                cache[position[0] % max_context, column] = value
        return output

    return residual_store


@lru_cache(maxsize=None)
def compile_token_store(max_context: int):
    """Record one dynamic token in a circular device-side history."""

    tilelang, T = _imports()

    @tilelang.jit(target="cuda")
    def store(
        token: T.Tensor((1,), T.int64),
        history: T.Tensor((max_context,), T.int64),
        position: T.Tensor((1,), T.int32),
    ):
        output = T.empty((1,), T.int32)
        with T.Kernel(1, threads=1):
            history[position[0] % max_context] = token[0]
            output[0] = position[0]
        return output

    return store


@lru_cache(maxsize=None)
def compile_circular_gather(max_context: int, width: int):
    """Read a circular cache in chronological order on CUDA."""

    tilelang, T = _imports()
    threads = 256
    total = max_context * width

    @tilelang.jit(target="cuda")
    def gather(
        cache: T.Tensor((max_context, width), T.bfloat16),
        position: T.Tensor((1,), T.int32),
    ):
        output = T.empty((max_context, width), T.bfloat16)
        with T.Kernel(T.ceildiv(total, threads), threads=threads) as block:
            lane = T.get_thread_binding(0)
            linear = block * threads + lane
            if linear < total:
                row = linear // width
                column = linear % width
                start = T.if_then_else(
                    position[0] + 1 <= max_context,
                    0, (position[0] + 1) % max_context,
                )
                output[row, column] = cache[(start + row) % max_context, column]
        return output

    return gather


@lru_cache(maxsize=None)
def compile_structural_softmax(max_context: int, input_scale: float = 1.0):
    """Scale BF16 scores, mask physical slots, and compute BF16 softmax."""

    tilelang, T = _imports()
    threads = 256

    @tilelang.jit(target="cuda")
    def softmax(
        scores: T.Tensor((max_context,), T.bfloat16),
        position: T.Tensor((1,), T.int32),
    ):
        output = T.empty((max_context,), T.bfloat16)
        with T.Kernel(1, threads=threads):
            lane = T.get_thread_binding(0)
            partial_max = T.alloc_local((1,), T.float32)
            maximum = T.alloc_local((1,), T.float32)
            partial_sum = T.alloc_local((1,), T.float32)
            total = T.alloc_local((1,), T.float32)
            partial_max[0] = T.float32(float("-inf"))
            for tile in T.serial(T.ceildiv(max_context, threads)):
                slot = tile * threads + lane
                if slot < max_context:
                    valid = T.if_then_else(
                        position[0] + 1 < max_context,
                        slot <= position[0],
                        True,
                    )
                    if valid:
                        scaled = (
                            scores[slot].astype(T.float32) * input_scale
                        ).astype(T.bfloat16)
                        partial_max[0] = T.max(
                            partial_max[0], scaled.astype(T.float32)
                        )
            with T.attr(
                T.comm_reducer(lambda a, b: T.max(a, b), [T.float32(float("-inf"))]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), partial_max[0], True, maximum[0], lane,
                    dtype="handle",
                ))
            T.clear(partial_sum)
            for tile in T.serial(T.ceildiv(max_context, threads)):
                slot = tile * threads + lane
                if slot < max_context:
                    valid = T.if_then_else(
                        position[0] + 1 < max_context,
                        slot <= position[0],
                        True,
                    )
                    if valid:
                        scaled = (
                            scores[slot].astype(T.float32) * input_scale
                        ).astype(T.bfloat16)
                        partial_sum[0] += T.exp(
                            scaled.astype(T.float32) - maximum[0]
                        )
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), partial_sum[0], True, total[0], lane,
                    dtype="handle",
                ))
            for tile in T.serial(T.ceildiv(max_context, threads)):
                slot = tile * threads + lane
                if slot < max_context:
                    valid = T.if_then_else(
                        position[0] + 1 < max_context,
                        slot <= position[0],
                        True,
                    )
                    scaled = (
                        scores[slot].astype(T.float32) * input_scale
                    ).astype(T.bfloat16)
                    output[slot] = T.if_then_else(
                        valid,
                        T.exp(scaled.astype(T.float32) - maximum[0]) / total[0],
                        T.float32(0),
                    )
        return output

    return softmax


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
    lanes = 32
    rows_per_block = 8
    normalization = features ** -0.5

    @tilelang.jit(target="cuda")
    def logits(
        projected: T.Tensor((features,), T.bfloat16),
        packed: T.Tensor((vocabulary, packed_width), T.uint8),
        bias: T.Tensor((vocabulary,), T.float32),
    ):
        output = T.empty((vocabulary,), T.float32)
        with T.Kernel(
            T.ceildiv(vocabulary, rows_per_block),
            threads=(lanes, rows_per_block),
        ) as block:
            lane = T.get_thread_binding(0)
            row_lane = T.get_thread_binding(1)
            token = block * rows_per_block + row_lane
            shared_projected = T.alloc_shared((features,), T.bfloat16)
            for tile in T.serial(T.ceildiv(features, lanes * rows_per_block)):
                feature = tile * lanes * rows_per_block + row_lane * lanes + lane
                if feature < features:
                    shared_projected[feature] = projected[feature]
            T.sync_threads()
            partial = T.alloc_local((1,), T.float32)
            reduced = T.alloc_local((1,), T.float32)
            T.clear(partial)
            if token < vocabulary:
                for byte_tile in T.serial(T.ceildiv(packed_width, lanes)):
                    byte_index = byte_tile * lanes + lane
                    if byte_index < packed_width:
                        byte = packed[token, byte_index].astype(T.int32)
                        for component in T.serial(8):
                            bit = (byte >> (7 - component)) & 1
                            value = shared_projected[
                                byte_index * 8 + component
                            ].astype(T.float32)
                            partial[0] += T.if_then_else(
                                bit == 0, -value, value
                            )
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), partial[0], True, reduced[0], lane, dtype="handle"
                ))
            if lane == 0 and token < vocabulary:
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
def compile_double_rms_norm(width: int, epsilon: float = 1e-6):
    """Apply two exact BF16 RMSNorms without materializing the first."""

    if width < 1:
        raise ValueError("RMSNorm width must be positive")
    tilelang, T = _imports()
    threads = min(256, 1 << (width - 1).bit_length())
    values_per_thread = T.ceildiv(width, threads)

    @tilelang.jit(target="cuda")
    def double_rms_norm(
        x: T.Tensor((width,), T.bfloat16),
        first_weight: T.Tensor((width,), T.float32),
        second_weight: T.Tensor((width,), T.float32),
    ):
        output = T.empty((width,), T.bfloat16)
        with T.Kernel(1, threads=threads):
            lane = T.get_thread_binding(0)
            values = T.alloc_local((values_per_thread,), T.bfloat16)
            partial = T.alloc_local((1,), T.float32)
            first_reduced = T.alloc_local((1,), T.float32)
            second_reduced = T.alloc_local((1,), T.float32)
            T.clear(partial)
            for tile in T.serial(values_per_thread):
                column = tile * threads + lane
                if column < width:
                    value = x[column].astype(T.float32)
                    partial[0] += value * value
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), partial[0], True, first_reduced[0], lane,
                    dtype="handle",
                ))
            first_scale = T.rsqrt(first_reduced[0] / width + epsilon)
            T.clear(partial)
            for tile in T.serial(values_per_thread):
                column = tile * threads + lane
                if column < width:
                    normalized = (
                        x[column].astype(T.float32) * first_scale
                    ).astype(T.bfloat16)
                    value = (
                        normalized * first_weight[column].astype(T.bfloat16)
                    ).astype(T.bfloat16)
                    values[tile] = value
                    value_float = value.astype(T.float32)
                    partial[0] += value_float * value_float
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), partial[0], True, second_reduced[0], lane,
                    dtype="handle",
                ))
            second_scale = T.rsqrt(second_reduced[0] / width + epsilon)
            for tile in T.serial(values_per_thread):
                column = tile * threads + lane
                if column < width:
                    normalized = (
                        values[tile].astype(T.float32) * second_scale
                    ).astype(T.bfloat16)
                    output[column] = (
                        normalized * second_weight[column].astype(T.bfloat16)
                    )
        return output

    return double_rms_norm


@lru_cache(maxsize=None)
def compile_residual_rms_norm(width: int, epsilon: float = 1e-6):
    """Add a BF16 residual and return it together with its exact RMSNorm."""

    if width < 1:
        raise ValueError("RMSNorm width must be positive")
    tilelang, T = _imports()
    threads = min(256, 1 << (width - 1).bit_length())
    values_per_thread = T.ceildiv(width, threads)

    @tilelang.jit(target="cuda")
    def residual_rms_norm(
        residual: T.Tensor((width,), T.bfloat16),
        projected: T.Tensor((width,), T.bfloat16),
        weight: T.Tensor((width,), T.float32),
    ):
        output = T.empty((width,), T.bfloat16)
        normalized_output = T.empty((width,), T.bfloat16)
        with T.Kernel(1, threads=threads):
            lane = T.get_thread_binding(0)
            values = T.alloc_local((values_per_thread,), T.bfloat16)
            partial = T.alloc_local((1,), T.float32)
            reduced = T.alloc_local((1,), T.float32)
            T.clear(partial)
            for tile in T.serial(values_per_thread):
                column = tile * threads + lane
                if column < width:
                    value = (residual[column] + projected[column]).astype(T.bfloat16)
                    values[tile] = value
                    output[column] = value
                    value_float = value.astype(T.float32)
                    partial[0] += value_float * value_float
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), partial[0], True, reduced[0], lane, dtype="handle"
                ))
            scale = T.rsqrt(reduced[0] / width + epsilon)
            for tile in T.serial(values_per_thread):
                column = tile * threads + lane
                if column < width:
                    normalized = (
                        values[tile].astype(T.float32) * scale
                    ).astype(T.bfloat16)
                    normalized_output[column] = (
                        normalized * weight[column].astype(T.bfloat16)
                    )
        return output, normalized_output

    return residual_rms_norm


@lru_cache(maxsize=None)
def compile_qk_rms_norm(
    query_heads: int, key_heads: int, head_dim: int, epsilon: float = 1e-6
):
    """Normalize contiguous Q/K heads with their respective shared weights."""

    tilelang, T = _imports()
    total_heads = query_heads + key_heads
    threads = min(256, 1 << (head_dim - 1).bit_length())

    @tilelang.jit(target="cuda")
    def rms_norm(
        qk: T.Tensor((total_heads, head_dim), T.bfloat16),
        query_weight: T.Tensor((head_dim,), T.float32),
        key_weight: T.Tensor((head_dim,), T.float32),
    ):
        output = T.empty((total_heads, head_dim), T.bfloat16)
        with T.Kernel(total_heads, threads=threads) as head:
            lane = T.get_thread_binding(0)
            partial = T.alloc_local((1,), T.float32)
            reduced = T.alloc_local((1,), T.float32)
            value = qk[head, lane].astype(T.float32)
            partial[0] = value * value
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), partial[0], True, reduced[0], lane,
                    dtype="handle",
                ))
            scale = T.rsqrt(reduced[0] / head_dim + epsilon)
            normalized = (value * scale).astype(T.bfloat16)
            weight = T.if_then_else(
                head < query_heads,
                query_weight[lane],
                key_weight[lane],
            ).astype(T.bfloat16)
            output[head, lane] = normalized * weight
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
def compile_rope_angles(half_dim: int):
    """Generate decode RoPE cosine and sine from a device position."""

    tilelang, T = _imports()

    @tilelang.jit(target="cuda")
    def angles(
        position: T.Tensor((1,), T.int32),
        inv_frequency: T.Tensor((half_dim,), T.float32),
    ):
        cosine = T.empty((half_dim,), T.bfloat16)
        sine = T.empty((half_dim,), T.bfloat16)
        with T.Kernel(1, threads=half_dim):
            lane = T.get_thread_binding(0)
            angle = position[0].astype(T.float32) * inv_frequency[lane]
            cosine[lane] = T.cos(angle)
            sine[lane] = T.sin(angle)
        return cosine, sine

    return angles


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


@lru_cache(maxsize=None)
def compile_rope_quantize(heads: int, head_dim: int):
    """Apply BF16 RoPE then exact per-head power-of-two quantization."""

    tilelang, T = _imports()
    threads = min(256, 1 << (head_dim - 1).bit_length())

    @tilelang.jit(target="cuda")
    def rope_quantize(
        x: T.Tensor((heads, head_dim), T.bfloat16),
        cosine: T.Tensor((head_dim // 2,), T.bfloat16),
        sine: T.Tensor((head_dim // 2,), T.bfloat16),
    ):
        output = T.empty((heads, head_dim), T.bfloat16)
        with T.Kernel(heads, threads=threads) as head:
            lane = T.get_thread_binding(0)
            rotated = T.alloc_shared((head_dim,), T.bfloat16)
            if lane < head_dim:
                pair = lane // 2
                even = x[head, pair * 2]
                odd = x[head, pair * 2 + 1]
                even_cosine = even * cosine[pair]
                odd_sine = odd * sine[pair]
                even_sine = even * sine[pair]
                odd_cosine = odd * cosine[pair]
                rotated[lane] = T.if_then_else(
                    lane % 2 == 0,
                    even_cosine - odd_sine,
                    even_sine + odd_cosine,
                )
            T.sync_threads()
            partial = T.alloc_local((1,), T.float32)
            reduced = T.alloc_local((1,), T.float32)
            if lane < head_dim:
                partial[0] = T.abs(rotated[lane]).astype(T.float32)
            else:
                partial[0] = 0.0
            with T.attr(
                T.comm_reducer(lambda a, b: T.max(a, b), [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), partial[0], True, reduced[0], lane,
                    dtype="handle",
                ))
            maximum = T.max(reduced[0], T.float32(1e-6)).astype(T.bfloat16)
            ratio = (maximum / T.bfloat16(127.0)).astype(T.float32)
            logarithm = T.log2(ratio).astype(T.bfloat16).astype(T.float32)
            exponent = T.ceil(logarithm).astype(T.bfloat16).astype(T.float32)
            scale = T.exp2(exponent).astype(T.bfloat16)
            if lane < head_dim:
                rounded = T.round((rotated[lane] / scale).astype(T.float32))
                clipped = T.min(
                    T.max(rounded, T.float32(-127.0)), T.float32(127.0)
                ).astype(T.bfloat16)
                output[head, lane] = clipped * scale
        return output

    return rope_quantize


@lru_cache(maxsize=None)
def compile_rope_quantize_cache(
    query_heads: int, kv_heads: int, head_dim: int, max_context: int,
    epsilon: float = 1e-6,
):
    """Normalize/quantize Q/K and store the current K/V cache entry."""

    tilelang, T = _imports()
    heads = query_heads + kv_heads
    threads = min(256, 1 << (head_dim - 1).bit_length())

    @tilelang.jit(target="cuda")
    def rope_quantize_cache(
        qk: T.Tensor((heads, head_dim), T.bfloat16),
        value: T.Tensor((kv_heads, head_dim), T.bfloat16),
        query_weight: T.Tensor((head_dim,), T.float32),
        key_weight: T.Tensor((head_dim,), T.float32),
        cosine: T.Tensor((head_dim // 2,), T.bfloat16),
        sine: T.Tensor((head_dim // 2,), T.bfloat16),
        keys: T.Tensor((kv_heads, max_context, head_dim), T.bfloat16),
        values: T.Tensor((kv_heads, max_context, head_dim), T.bfloat16),
        position: T.Tensor((1,), T.int32),
    ):
        output = T.empty((heads, head_dim), T.bfloat16)
        with T.Kernel(heads, threads=threads) as head:
            lane = T.get_thread_binding(0)
            normalized_qk = T.alloc_shared((head_dim,), T.bfloat16)
            rotated = T.alloc_shared((head_dim,), T.bfloat16)
            norm_partial = T.alloc_local((1,), T.float32)
            norm_reduced = T.alloc_local((1,), T.float32)
            qk_value = qk[head, lane].astype(T.float32)
            norm_partial[0] = qk_value * qk_value
            with T.attr(
                T.comm_reducer(lambda a, b: a + b, [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), norm_partial[0], True, norm_reduced[0], lane,
                    dtype="handle",
                ))
            norm_scale = T.rsqrt(norm_reduced[0] / head_dim + epsilon)
            normalized = (qk_value * norm_scale).astype(T.bfloat16)
            norm_weight = T.if_then_else(
                head < query_heads, query_weight[lane], key_weight[lane]
            ).astype(T.bfloat16)
            normalized_qk[lane] = normalized * norm_weight
            T.sync_threads()
            pair = lane // 2
            even = normalized_qk[pair * 2]
            odd = normalized_qk[pair * 2 + 1]
            even_cosine = even * cosine[pair]
            odd_sine = odd * sine[pair]
            even_sine = even * sine[pair]
            odd_cosine = odd * cosine[pair]
            rotated[lane] = T.if_then_else(
                lane % 2 == 0,
                even_cosine - odd_sine,
                even_sine + odd_cosine,
            )
            T.sync_threads()
            partial = T.alloc_local((1,), T.float32)
            reduced = T.alloc_local((1,), T.float32)
            partial[0] = T.abs(rotated[lane]).astype(T.float32)
            with T.attr(
                T.comm_reducer(lambda a, b: T.max(a, b), [T.float32(0)]),
                "reduce_scope", T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(T.tvm_thread_allreduce(
                    T.uint32(1), partial[0], True, reduced[0], lane,
                    dtype="handle",
                ))
            maximum = T.max(reduced[0], T.float32(1e-6)).astype(T.bfloat16)
            ratio = (maximum / T.bfloat16(127.0)).astype(T.float32)
            logarithm = T.log2(ratio).astype(T.bfloat16).astype(T.float32)
            exponent = T.ceil(logarithm).astype(T.bfloat16).astype(T.float32)
            scale = T.exp2(exponent).astype(T.bfloat16)
            rounded = T.round((rotated[lane] / scale).astype(T.float32))
            clipped = T.min(
                T.max(rounded, T.float32(-127.0)), T.float32(127.0)
            ).astype(T.bfloat16)
            quantized = clipped * scale
            output[head, lane] = quantized
            if head >= query_heads:
                kv_head = head - query_heads
                value_partial = T.alloc_local((1,), T.float32)
                value_reduced = T.alloc_local((1,), T.float32)
                value_partial[0] = T.abs(value[kv_head, lane]).astype(T.float32)
                with T.attr(
                    T.comm_reducer(lambda a, b: T.max(a, b), [T.float32(0)]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
                ):
                    T.evaluate(T.tvm_thread_allreduce(
                        T.uint32(1), value_partial[0], True, value_reduced[0], lane,
                        dtype="handle",
                    ))
                value_maximum = T.max(
                    value_reduced[0], T.float32(1e-6)
                ).astype(T.bfloat16)
                value_ratio = (
                    value_maximum / T.bfloat16(127.0)
                ).astype(T.float32)
                value_logarithm = (
                    T.log2(value_ratio).astype(T.bfloat16).astype(T.float32)
                )
                value_exponent = (
                    T.ceil(value_logarithm).astype(T.bfloat16).astype(T.float32)
                )
                value_scale = T.exp2(value_exponent).astype(T.bfloat16)
                value_rounded = T.round((
                    value[kv_head, lane] / value_scale
                ).astype(T.float32))
                value_clipped = T.min(
                    T.max(value_rounded, T.float32(-127.0)), T.float32(127.0)
                ).astype(T.bfloat16)
                slot = position[0] % max_context
                keys[kv_head, slot, lane] = quantized
                values[kv_head, slot, lane] = value_clipped * value_scale
        return output

    return rope_quantize_cache






class TileLangLinear:
    """Shape-cached callable used for every decode-time projection."""

    def __call__(self, x, weight):
        if isinstance(weight, DenseDecodeRVQWeight):
            return compile_rvq_dense_gemv(*weight.shape)(
                x.contiguous(), weight.dense
            )
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

    def swiglu(self, x, weight):
        if not isinstance(weight, PackedTernaryWeight):
            raise TypeError("SwiGLU projection requires packed ternary weights")
        if isinstance(weight, InterleavedTernaryWeight):
            return compile_interleaved_ternary_swiglu(
                weight.out_features, weight.in_features
            )(x.contiguous(), weight.paired, weight.scales)
        return compile_ternary_swiglu(
            weight.out_features, weight.in_features
        )(x.contiguous(), weight.packed, weight.scales)

    def gated_residual(self, x, gate, residual, weight):
        if not isinstance(weight, PackedRVQWeight):
            raise TypeError("gated residual projection requires packed RVQ weights")
        return compile_rvq_gemv_gated_residual(
            weight.out_features, weight.in_features, weight.group_size, weight.stages
        )(
            x.contiguous(), gate, residual, weight.codebooks, weight.indices,
            weight.scales,
        )

    def split_silu(self, left, right, weight):
        if not isinstance(weight, PackedRVQWeight):
            raise TypeError("split SiLU projection requires packed RVQ weights")
        if weight.in_features != left.numel() + right.numel():
            raise ValueError("split SiLU inputs do not match projection width")
        if isinstance(weight, DenseDecodeRVQWeight):
            return compile_rvq_dense_gemv_split_silu(
                weight.out_features, left.numel()
            )(left.contiguous(), right.contiguous(), weight.dense)
        return compile_rvq_gemv_split_silu(
            weight.out_features, left.numel(), weight.group_size, weight.stages
        )(
            left.contiguous(), right.contiguous(), weight.codebooks,
            weight.indices, weight.scales,
        )

    def rvq_residual(self, x, residual, weight):
        if not isinstance(weight, PackedRVQWeight):
            raise TypeError("RVQ residual projection requires packed RVQ weights")
        if isinstance(weight, DenseDecodeRVQWeight):
            return compile_rvq_dense_gemv_residual(*weight.shape)(
                x.contiguous(), residual, weight.dense
            )
        return compile_rvq_gemv_residual(
            weight.out_features, weight.in_features, weight.group_size,
            weight.stages,
        )(
            x.contiguous(), residual, weight.codebooks, weight.indices,
            weight.scales,
        )

    def residual(self, x, residual, weight):
        if not isinstance(weight, PackedTernaryWeight):
            import torch.nn.functional as functional

            projected = functional.linear(x, weight)
            return residual + projected
        return compile_ternary_gemv_residual(
            weight.out_features, weight.in_features
        )(x.contiguous(), residual, weight.packed, weight.scales)


class TorchLinear:
    """Reference backend used by parity tests and kernel bring-up."""

    def __call__(self, x, weight):
        import torch.nn.functional as functional

        return functional.linear(x, weight)
