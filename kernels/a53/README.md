# Cortex-A53 ternary FFN kernel experiment

This harness tests the first uncompressed execution reference: signed INT8 activations times
signed INT8 ternary weights, accumulated exactly into INT32. It uses only baseline ARMv8-A NEON
widening multiply-accumulate operations, not DotProd or I8MM.

Weights are tile-major: `[output_tile][input][16 output lanes]`. Compact base-3 SHDW weights must
be transposed and expanded into this layout once at model load. The 128-input partial interval is
safe for INT16 because `128 * 127 = 16256`.

On the Orange Pi:

```bash
make -C kernels/a53 clean all
taskset -c 3 kernels/a53/ternary_gemv 1536 4224 200
taskset -c 3 kernels/a53/ternary_gemv 4224 1536 200
```

Benchmark actual compact base-3 tensors after one-time expansion to the execution layout:

```bash
taskset -c 3 kernels/a53/ternary_gemv --shdw deployment/shadow250m_instruct.shdw b.0.up 200
taskset -c 3 kernels/a53/ternary_gemv --shdw deployment/shadow250m_instruct.shdw b.0.gt 200
taskset -c 3 kernels/a53/ternary_gemv --shdw deployment/shadow250m_instruct.shdw b.0.dn 200
```

Then collect counters, if permitted by the kernel configuration:

```bash
perf stat -r 5 -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,cache-misses \
  taskset -c 3 kernels/a53/ternary_gemv 1536 4224 1000
```

The harness validates every output against the scalar integer equation before timing. It does not
yet include row-scale application, SwiGLU, signed-nibble decoding, or fused `up`/`gate`; those are
fallback/reference candidates in `DEVPLAN-A55.md`.
