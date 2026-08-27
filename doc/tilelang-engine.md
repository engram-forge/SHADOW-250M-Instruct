# SHADOW CUDA engine with TileLang

This worktree contains an in-process GPU runtime for the released `.shdw`
model. It is intentionally organized as a learning path: the container reader,
reference graph, and TileLang kernels are separate, so generated CUDA can be
inspected and each optimization can be checked independently.

## Setup

The current engine targets NVIDIA CUDA and uses PyTorch only for CUDA tensors,
allocation, and the reference backend. TileLang owns decode-time GEMV kernels.

```bash
uv sync --extra tilelang
uv run python -c "import torch, tilelang; print(torch.cuda.get_device_name(), tilelang.__version__)"
```

TileLang needs CUDA compiler components. The `tilelang` extra pins matching
CUDA 13.0 compiler and CCCL packages on Linux, so `uv sync` also works on a
machine without a host `nvcc`. Do not mix compiler and header minor versions;
CCCL deliberately rejects (for example) CUDA 13.3 `nvcc` with 13.0 headers.

## Run

The CLI mirrors the native CPU binary's low-level contract: model, vocabulary
table, space-separated prompt IDs, then the number of tokens to generate.

```bash
uv run python -m shadow_tilelang \
  deployment/shadow250m_instruct.shdw deployment/fp131072.npy \
  "2 8 925 1234" 16 --backend tilelang
```

For interactive text chat, use the repository tokenizer and the same chat
template as the CPU runtime:

```bash
uv run python -m shadow_tilelang.chat
```

Use `--backend torch` to validate the graph without TileLang kernels. Both
backends run on CUDA and use the same unpacked weights and caches.

## What is native today

- The 52 MB deployment model is read directly; no training checkpoint or
  conversion step is required.
- RVQ and base-3 ternary records are validated and materialized on the GPU.
- Prompt processing and generation stay in one process; the bundled CPU binary
  is not invoked.
- All autoregressive linear projections use a shape-specialized TileLang CUDA
  BF16 GEMV with FP32 accumulation.
- Q/K/V and SwiGLU up/gate rows are concatenated at load time, reducing each
  transformer block from seven projection launches to four.
- The exact power-of-two Q/K/V quantizer and floor/exp2 shiftmax attention are
  represented in the runtime graph.

The first implementation processes prefill one token at a time and materializes
weights as BF16. That is the correctness-oriented baseline. The next measured
optimizations are fused RVQ/base-3 dequantization GEMV, a tiled exact-shiftmax
attention kernel, and a batched prefill path.

On the development H100 NVL, the warm stateful decode fixture currently runs
at about 72 tokens/s and peaks at 1.19 GiB allocated. This is a bring-up number,
not a release claim: the GPU was shared, and the baseline deliberately expands
the 52 MB packed weights to BF16. Packed dequantization GEMV is the main memory
and bandwidth milestone.

## Inspect generated CUDA

TileLang keeps one compiled kernel per matrix shape. After a projection has
compiled, its CUDA source is available from the cached kernel object:

```python
from shadow_tilelang.kernels import compile_gemv

kernel = compile_gemv(1536, 1536)
print(kernel.get_kernel_source())
```

## Validation

The format tests consume the complete release model and check both packed
layouts with small hand-computable examples:

```bash
uv run pytest tests/test_tilelang_format.py
uv run --with pytest pytest tests/test_tilelang_cuda.py
```

For graph parity, generate with `--backend torch`, then switch to `tilelang`;
greedy token IDs must match. The CUDA test covers every matrix shape in SHADOW
and an eight-token stateful generation. The bundled CPU engine shares the first
four greedy tokens in that fixture but can diverge later because its deployment
path uses different activation/accumulation policies (including an optional
INT8 FFN); the authoritative kernel-parity oracle is the CUDA reference graph.
