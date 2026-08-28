# PyTorch CUDA versus TileLang

This report compares the two CUDA backends in `TileLangEngine`: the eager
PyTorch reference and the optimized TileLang runtime. It covers fresh-prompt
prefill and fixed-length greedy decode at 32, 128, 1,024, and 2,048 tokens.

## Results

TileLang is faster at every measured size. Its advantage is largest for long
prefill because TileLang batches a fresh prompt while the reference backend
consumes it token by token. Decode remains more than 23x faster through 1,024
generated tokens, then falls to 14.2x at 2,048 tokens as the attention context
reaches the full 2,048-token circular-cache capacity.

### Prefill

| Prompt tokens | PyTorch CUDA | TileLang | TileLang speedup | PyTorch median | TileLang median |
|---:|---:|---:|---:|---:|---:|
| 32 | 95.2 tok/s | 3,320.2 tok/s | 34.9x | 336.05 ms | 9.64 ms |
| 128 | 93.8 tok/s | 6,422.9 tok/s | 68.5x | 1,365.23 ms | 19.93 ms |
| 1,024 | 77.1 tok/s | 6,299.7 tok/s | 81.7x | 13,281.54 ms | 162.55 ms |
| 2,048 | 63.3 tok/s | 5,429.5 tok/s | 85.8x | 32,357.92 ms | 377.20 ms |

The next-token argmax matched between backends at every prompt length.

### Decode

| Generated tokens | PyTorch CUDA | TileLang | TileLang speedup | PyTorch median | TileLang median |
|---:|---:|---:|---:|---:|---:|
| 32 | 87.3 tok/s | 2,157.8 tok/s | 24.7x | 366.51 ms | 14.83 ms |
| 128 | 85.7 tok/s | 2,031.3 tok/s | 23.7x | 1,493.80 ms | 63.01 ms |
| 1,024 | 71.8 tok/s | 1,866.7 tok/s | 26.0x | 14,252.09 ms | 548.56 ms |
| 2,048 | 59.4 tok/s | 842.6 tok/s | 14.2x | 34,470.57 ms | 2,430.50 ms |

The complete greedy token sequence matched between backends at every decode
length, including cache wrap after the initial four-token prompt plus 2,048
generated tokens.

## Method

The benchmark ran on 2026-08-28 with the following environment:

- NVIDIA H100 NVL
- Python 3.10.12
- PyTorch 2.13.0+cu130
- CUDA runtime 13.0
- TileLang 0.1.13
- Git commit `b338e7399e69b89da2351f7016441cae4ca0f7e0`
- `max_context=2048`

Each result is the median of three runs. CUDA synchronization brackets every
measured interval. Model loading, lazy TileLang compilation, CUDA graph capture,
allocator warm-up, and decode prompt prefill are excluded. Transfers performed
by the normal runtime path are included. In particular, eager PyTorch decode
selects each token on the host, while TileLang greedy decode returns generated
tokens to the host in chunks.

Prefill uses the deterministic token pattern `2 8 925 1234`, repeated to the
requested length. Decode starts from the four-token prompt `2 8 925 1234`; its
reported size is the number of newly generated tokens. Both paths use greedy
selection without repetition penalty or stop tokens.

The working tree was dirty during measurement. The only pre-existing runtime
change was an uncommitted `rows_per_block` parameter on the interleaved ternary
SwiGLU kernel; its default remains eight. The benchmark harness itself was also
uncommitted.

This is an end-to-end comparison of the runtime implementations users actually
invoke, not an isolated-kernel comparison. The PyTorch backend is deliberately
a simple correctness reference: it materializes quantized weights, uses eager
operations, processes prefill token by token, and does not use CUDA graphs. The
TileLang backend uses packed weights, batched prefill, fused kernels, and
CUDA-graph greedy decode. The speedups therefore measure the complete optimized
runtime, not only the TileLang compiler's contribution.

## Raw timing ranges

| Phase | Tokens | PyTorch range | TileLang range |
|---|---:|---:|---:|
| Prefill | 32 | 335.69–336.14 ms | 9.63–10.36 ms |
| Prefill | 128 | 1,364.11–1,365.41 ms | 19.91–19.97 ms |
| Prefill | 1,024 | 13,238.18–13,292.85 ms | 162.53–162.59 ms |
| Prefill | 2,048 | 32,126.95–32,455.67 ms | 377.16–377.27 ms |
| Decode | 32 | 365.69–366.55 ms | 14.83–14.89 ms |
| Decode | 128 | 1,486.48–1,494.94 ms | 63.00–63.04 ms |
| Decode | 1,024 | 14,211.86–14,402.86 ms | 548.49–549.28 ms |
| Decode | 2,048 | 34,070.93–35,762.24 ms | 2,425.61–2,431.86 ms |

## Reproduce

Install the TileLang environment, then run:

```bash
uv sync --extra tilelang
uv run python benchmarks/tilelang_bench.py \
  --sizes 32 128 1024 2048 \
  --repetitions 3 \
  --out tilelang-benchmark.json
```

The optional JSON file contains every individual duration, medians, throughput,
speedups, environment metadata, and parity results.
