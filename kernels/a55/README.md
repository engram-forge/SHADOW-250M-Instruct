# Cortex-A55 DotProd ternary FFN kernel

This is the direct-INT8-trit reference for RK3566. Its layout is a row-major four-output by
four-input weight tile. Four activations are replicated across all `SDOT` lanes, so one instruction
updates four different output rows directly. Four accumulator vectors hide dependency latency and
are combined before one vector store; no horizontal reduction is needed. It validates all INT32
outputs against scalar code before timing. The compact base-3 tensor is expanded and repacked once
before the timed loop.

Compile and verify on Orange Pi 3B:

```bash
make -C kernels/a55 clean check
grep -qw asimddp /proc/cpuinfo
make -C kernels/a55 assembly
grep -n '\<sdot\>' kernels/a55/ternary_dotprod.s
```

Benchmark actual release tensors with fixed clocks and one pinned core first:

```bash
taskset -c 3 kernels/a55/ternary_dotprod --shdw \
  deployment/shadow250m_instruct.shdw b.0.up 500
taskset -c 3 kernels/a55/ternary_dotprod --shdw \
  deployment/shadow250m_instruct.shdw b.0.gt 500
taskset -c 3 kernels/a55/ternary_dotprod --shdw \
  deployment/shadow250m_instruct.shdw b.0.dn 500
```

Run the same tensors through the half-size signed-nibble candidate:

```bash
taskset -c 3 kernels/a55/ternary_dotprod --nibble --shdw \
  deployment/shadow250m_instruct.shdw b.0.up 500
taskset -c 3 kernels/a55/ternary_dotprod --nibble --shdw \
  deployment/shadow250m_instruct.shdw b.0.gt 500
taskset -c 3 kernels/a55/ternary_dotprod --nibble --shdw \
  deployment/shadow250m_instruct.shdw b.0.dn 500
```

Collect counters over enough iterations to reduce startup noise:

```bash
perf stat -r 5 -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,cache-misses \
  taskset -c 3 kernels/a55/ternary_dotprod --shdw \
  deployment/shadow250m_instruct.shdw b.0.up 1000
```

The 1-byte/weight path is the compute-minimal reference, not the presumed final format. The included
signed-nibble path halves DRAM traffic. Choose between them using complete FFN time and hardware
counters, not instruction count alone. Do not run this `+dotprod` benchmark binary on a CPU without
`asimddp`; production dispatch must occur from a baseline ARMv8-A translation unit before calling
the separately compiled DotProd code.
