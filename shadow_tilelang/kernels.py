"""TileLang CUDA kernels used by the autoregressive engine."""

from functools import lru_cache


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


class TileLangLinear:
    """Shape-cached callable used for every decode-time projection."""

    def __call__(self, x, weight):
        kernel = compile_gemv(weight.shape[0], weight.shape[1])
        return kernel(x.contiguous(), weight)


class TorchLinear:
    """Reference backend used by parity tests and kernel bring-up."""

    def __call__(self, x, weight):
        import torch.nn.functional as functional

        return functional.linear(x, weight)
