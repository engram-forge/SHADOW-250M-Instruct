# ARM engine reference audit

## Objective

Find step-function CPU ideas applicable to exact single-stream decoding on the
Orange Pi H618 Cortex-A53. The target excludes DotProd, I8MM, SVE/SVE2, native
FP16 arithmetic, and native-host compilation. CUDA serving throughput is not a
proxy for single-request ARM latency.

## Transferable ideas

| Reference | Transferable mechanism | Immediate relevance |
| --- | --- | --- |
| llama.cpp / Ollama | Offline quantized packing, shape-specific GEMV kernels, CPU feature dispatch, persistent worker pools | High |
| KleidiAI | Separate packers and microkernels by matrix shape, packed metadata beside weights, explicit kernel capability contracts | High conceptually |
| XNNPACK | Generated microkernel families, cache-aware tiling, benchmark-selected dispatch | High conceptually |
| vLLM | Paged KV cache, prefix reuse, continuous batching | High for multi-request serving; low for one-stream decode |
| SGLang | Radix prefix cache and request scheduling | High for repeated prompts; low for one-stream matrix latency |

KleidiAI's current low-bit high-throughput kernels predominantly rely on ARM
DotProd, I8MM, or newer extensions. They cannot be used in the Cortex-A53
artifact. The useful lesson is its packer/microkernel interface, not its
instructions. CUDA projects similarly inform scheduling and cache ownership,
not NEON arithmetic.

## Current engine overlap

The runtime already implements several standard CPU-engine mechanisms:

- model-load-time ternary repacking;
- 16-output-row NEON microkernels;
- contiguous weight streaming and per-row scales;
- persistent worker threads;
- persistent RVQ lookup scratch;
- exact batch-4 prompt kernels;
- separate scalar/reference and optimized paths;
- Cortex-A53-fenced compilation and forbidden-instruction scanning.

Consequently, another loop-unroll, prefetch, dispatch fusion, or full int8
expansion is unlikely to produce a step-function gain. Those routes have also
failed measured gates in `linux_arm64_wsl_report.md`.

## Next step-function experiment

Prototype an offline **two-bitplane ternary pack** as an isolated operator
benchmark. For each output-row block and input tile, store positive and negative
membership masks instead of the current signed-nibble expansion. The experiment
must evaluate several fixed shapes independently:

- paired `up/gt`: `4224 x 1536` twice;
- `dn`: `1536 x 4224`;
- batch-4 variants for prefill.

The operator gate is intentionally demanding:

1. Preserve the exact FP32 input-column accumulation order and bitwise output.
2. Use ARMv8.0-A NEON only.
3. Reduce packed runtime weight storage versus signed nibbles; do not retain both
   layouts for the full model during the final runtime experiment.
4. Demonstrate at least 1.5x isolated throughput on both paired `up/gt` and `dn`
   before integration. A smaller operator result cannot plausibly create the
   requested 2x whole-runtime improvement.
5. If the bit-mask-to-FP32 reconstruction cost fails the operator gate, stop the
   exact representation route rather than integrating it into the model.

This is distinct from the previously rejected packed-nibble mask-select kernel:
that candidate decoded signed nibbles and then widened comparison masks. The new
gate is only justified if separate positive/negative bitplanes eliminate that
decode. The earlier 2.16x regression sets a strong prior that baseline NEON mask
expansion may still be too expensive.

## Routes capable of 2x--5x

If the compact exact operator fails, the remaining plausible step changes alter
model computation rather than kernel syntax:

1. training-aware conditional depth or early exit;
2. a smaller draft model for speculative decoding (currently lower priority);
3. retrained activation quantization paired with a Cortex-A53 integer kernel;
4. concurrent-request batching when aggregate throughput, rather than one-user
   latency, is the product goal;
5. hardware with DotProd/I8MM when the deployment target can change.

Each route requires separate quality evaluation and the prefill/decode matrices
defined in `AGENTS.md`.

## Two-bitplane operator result

The proposed operator was implemented in `shadow_ternary_microbench`. Separate
positive and negative 16-row membership masks reduced the tested matrix's
runtime weight storage from 3.094 MiB to 1.547 MiB and produced bit-identical
FP32 output. Across five runs, however, the current signed-nibble kernel measured
about 485--512 us while the bitplane kernel measured 749--786 us, a 32--38%
regression. Baseline ARMv8.0-A NEON still has to broadcast masks, perform
per-lane variable shifts, subtract positive/negative membership, widen twice,
convert to FP32, and FMA. Halving streamed bytes does not repay that decode.

The candidate fails the 1.5x operator gate before paired `up`/`gt`, `dn`, or
batch-4 integration. No runtime representation or model-load path was changed.
This closes the exact compact-bitplane route for Cortex-A53 unless a materially
different arithmetic formulation is found.

## Ternary representation design summary

The file representation and the execution representation serve different goals.
Do not assume that fewer stored bits automatically means lower inference latency.

A `.shdw` ternary matrix packs five values from `{-1, 0, +1}` into one base-3
byte because `3^5 = 243 <= 256`. Its effective density is therefore 1.6 bits
per weight. This is excellent for distribution and disk reads, but direct matrix
execution would repeatedly require base-3 division or table decoding.

At model load, the runtime repacks these weights into signed nibbles: one 4-bit
two's-complement value per weight, with two weights per byte. This doubles the
in-memory weight size relative to two bitplanes and uses 2.5x the bits of the
base-3 file, but it enables the fastest exact Cortex-A53 kernel measured so far:
load 16 nibbles, sign-extend, widen, convert to FP32, then FMA against a reused
FP32 activation.

| Representation | Bits/weight | Intended role | Measured tradeoff |
| --- | ---: | --- | --- |
| Base-3 `.shdw` | 1.6 | Disk/download/archive size | Compact, but expensive to decode directly |
| Two bitplanes | 2 | Compact runtime weights | 50% smaller than nibbles; 32--38% slower |
| Signed nibbles | 4 | Exact Cortex-A53 execution | Current best speed/memory compromise |
| Signed INT8 | 8 | Simplest vector loading | About 10% faster operator; model-wide gain erased and RSS roughly doubled |

The bitplane result demonstrates the governing latency model:

`total time = weight movement + representation decode + FP32 accumulation`.

Bitplanes reduce weight movement, but Cortex-A53 lacks an instruction that maps
packed positive/negative masks directly onto FP32 lanes. Broadcasting masks,
variable shifting, extracting membership, widening, converting, and accumulating
cost more than the saved bytes. The current workload is sufficiently constrained
by instruction/decode throughput that lower storage alone does not improve latency.

FP16 does not change this conclusion on H618. Cortex-A53 can store and convert
FP16 values but cannot use ARMv8.2 native FP16 vector arithmetic. FP16 activations
would still be converted to FP32 for accumulation, adding conversion work while
only shrinking an activation vector that is reused across thousands of rows and
normally remains cache-resident. Accumulators and row scales must remain FP32 for
the current exact path.

### Decision rules for future representations

1. Measure storage, decode instructions, accumulation work, RSS, and whole-runtime
   latency independently; never select a format from bits per weight alone.
2. Gate a new representation in an isolated operator benchmark before changing
   the model loader or retaining a second full-model layout.
3. Require exact parity for an exact path and at least 1.5x operator throughput
   before expecting a step-function whole-runtime improvement.
4. Test the real `up/gt`, `dn`, and batch-prefill shapes; one layout need not win
   for every matrix aspect ratio.
5. Count expanded runtime memory, not only `.shdw` file size. A faster isolated
   kernel can lose when its larger working set exceeds cache or memory bandwidth.
6. Reconsider compact masks only with a fundamentally cheaper arithmetic mapping
   or on hardware with DotProd/I8MM; do not repeat nibble/bitplane loop variants.
7. To avoid FP32 weight-lane conversion entirely, activations must also move to an
   integer formulation. That is an approximate, training/evaluation-aware design,
   not a bit-exact representation swap.
