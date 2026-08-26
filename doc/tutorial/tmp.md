# Pretraining stability and QAT diagnostics

This guide explains how to read the stability metrics emitted by the SHADOW 250M Dolma
pretrainer. It focuses on two failure modes that a single gradient norm cannot distinguish:

1. gradient energy collapsing into a small number of coordinates; and
2. floating-point master weights drifting away from the weights used by the quantized forward
   pass.

The normal training stream is `pretrain_runs/dolma-8b/metrics.jsonl`. The low-frequency
stability stream is `pretrain_runs/dolma-8b/diagnostics.jsonl`.

## 1. Why gradient norm is not enough

The Euclidean gradient norm is

$$
\lVert g\rVert_2=\sqrt{\sum_i g_i^2}.
$$

It measures total gradient magnitude, which is exactly what global norm clipping needs. It
does not describe how that magnitude is distributed. For example, these vectors have nearly
the same norm but very different concentration:

$$
[10,0.1,0.1,0.1]
\qquad  ext{and}\qquad
[5,5,5,5].
$$

The trainer therefore records the pre-clip norm on every update and normalized participation
every ten updates.

## 2. Participation ratio

The effective number of contributing coordinates is

$$
N_{\mathrm{eff}}
=
\frac{\left(\sum_i g_i^2\right)^2}
{\sum_i g_i^4}.
$$

To compare tensors with different numbers of elements, the trainer reports

$$
R_{\mathrm{eff}}
=
\frac{N_{\mathrm{eff}}}{n}
=
\frac{\left(\sum_i g_i^2\right)^2}
{n\sum_i g_i^4}.
$$

Its useful endpoints are:

- $R_{\mathrm{eff}}=1$ when all coordinates have equal magnitude;
- $R_{\mathrm{eff}}\approx1/n$ when one coordinate dominates;
- $N_{\mathrm{eff}}=nR_{\mathrm{eff}}$ converts the fraction back to an effective coordinate
  count.

Examples:

| Gradient | $N_{\mathrm{eff}}$ | $R_{\mathrm{eff}}$ |
|---|---:|---:|
| $[5,5,5,5]$ | $4$ | $1$ |
| $[10,0.1,0.1,0.1]$ | $1.0006$ | $0.25015$ |

Scaling every coordinate by the same constant does not change participation. This is why it
complements rather than replaces the gradient norm.

### Understand Participation ratio
The easiest way to understand participation ratio is to rewrite the gradient as a distribution of energy.

  For gradient coordinates $g_i$, define each coordinate’s gradient energy:

  $$
  e_i=g_i^2.
  $$

  Then define the fraction of total energy held by coordinate $i$:

  $$
  p_i=\frac{g_i^2}{\sum_j g_j^2}.
  $$

  These fractions satisfy:

  $$
  \sum_i p_i=1.
  $$

  So the question becomes:

  > Given an uneven distribution $p_i$, how many coordinates would an equally distributed gradient need to have the same concentration?

  The participation ratio answers this with:

  $$
  N_{\mathrm{eff}}=\frac{1}{\sum_i p_i^2}.
  $$

  Substituting the definition of $p_i$ gives the original formula:

  $$
  N_{\mathrm{eff}}
=
  \frac{1}{
  \sum_i
  \left(
  \frac{g_i^2}{\sum_jg_j^2}
  \right)^2
  }
=
  \frac{\left(\sum_i g_i^2\right)^2}
  {\sum_i g_i^4}.
  $$

  ### Why the inverse square sum works

  Suppose exactly $k$ coordinates contribute equally. Each holds $1/k$ of the energy:

  $$
  p_i=\frac{1}{k}.
  $$

  Their squared fractions sum to:

  $$
  \sum_i p_i^2
=
  k\left(\frac{1}{k}\right)^2
=
  \frac{1}{k}.
  $$

  Taking the inverse recovers the number of contributors:

  $$
  N_{\mathrm{eff}}
=
  \frac{1}{1/k}
=
  k.
  $$

  This remains meaningful when the distribution is not perfectly equal: it returns the number of equally contributing coordinates that would have equivalent concentration.

### Diagnostic fields

| Field | Meaning |
|---|---|
| `normalized_global` | $R_{\mathrm{eff}}$ after treating all parameter gradients as one vector |
| `layer_median` | median $R_{\mathrm{eff}}$ among parameter tensors with at least 256 values |
| `layer_p10` | 10th percentile; 10% of measured tensors are at or below this value |
| `worst_name` | name of the measured tensor with the lowest participation |
| `worst` | that tensor's $R_{\mathrm{eff}}$ |

`normalized_global` is weighted by every coordinate and can be dominated by one large or
high-magnitude tensor. Use it together with `layer_median` and `layer_p10`. A low global
value with stable layer statistics is not equivalent to every layer collapsing.

## 3. QAT quantization gap

Training keeps floating-point master weights $w$, while the forward pass uses reconstructed
quantized weights $Q(w)$. The diagnostic reports normalized mean squared error:

$$
\operatorname{NMSE}
=
\frac{\sum_i\left(w_i-Q(w_i)\right)^2}
{\sum_i w_i^2}.
$$

The aggregate is energy-weighted across all modules in a family. It is not an unweighted mean
of per-module NMSE values. The corresponding relative RMS error is

$$
\frac{\operatorname{RMS}(w-Q(w))}{\operatorname{RMS}(w)}
=\sqrt{\operatorname{NMSE}},
$$

and an optional signal-to-quantization-noise view is

$$
\operatorname{SQNR}_{\mathrm{dB}}
=-10\log_{10}(\operatorname{NMSE}).
$$

The two reported families are:

- `rvq`: attention and structural projection matrices reconstructed by residual vector
  quantization;
- `ternary`: FFN matrices reconstructed with per-row scale and values in
  $\{-s,0,+s\}$.

`worst_name` and `worst_nmse` identify the individual module with the largest relative
error. A worst value close to the aggregate means the gap is broadly distributed. A worst
value several times the aggregate suggests an isolated module deserves inspection.

Aggressive low-bit QAT can have a large absolute NMSE while remaining healthy: the network is
trained with that distortion in the forward pass. The important signals are its baseline,
trend, worst/aggregate ratio, and quantized validation loss—not proximity to zero.

## 4. Interpreting the first live record

At update 10, the run reported:

| Metric | Value | Interpretation |
|---|---:|---|
| Global participation | $0.001025$ | gradient energy is globally concentrated; treat this as an early-run baseline |
| Median layer participation | $0.09647$ | the median measured tensor effectively uses about 9.65% of its coordinates |
| P10 layer participation | $0.03996$ | the lower tail is concentrated but not universally collapsed |
| Worst tensor | `tied_bias` | the 131,072-way vocabulary bias is the most concentrated tensor |
| Worst participation | $8.392  imes10^{-5}$ | $N_{\mathrm{eff}}\approx131072R_{\mathrm{eff}}\approx11$ effective coordinates |
| RVQ NMSE | $0.36898$ | relative RMS error $\approx0.607$; SQNR $\approx4.33$ dB |
| Worst RVQ module | `b.5.o` at $0.37117$ | only about 0.6% above aggregate; not an isolated failure |
| Ternary NMSE | $0.26321$ | relative RMS error $\approx0.513$; SQNR $\approx5.80$ dB |
| Worst ternary module | `b.2.dn` at $0.26347$ | essentially equal to aggregate; error is broadly uniform |
| Measurement time | $0.083$ seconds | about 0.15% overhead at a ten-update cadence |

The low `tied_bias` participation is plausible early in language-model training: a very large
vocabulary tensor receives highly non-uniform token evidence. It should not be used alone to
stop training. More concerning cases would be a persistent collapse in large transformer
matrices, multiple layers collapsing together, or concentration accompanied by loss and norm
spikes.

### Trend through update 110

| Update | Global $R_{\mathrm{eff}}$ | Median | P10 | Worst | RVQ NMSE | Ternary NMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.001025 | 0.09647 | 0.03996 | 0.0000839 | 0.368979 | 0.263210 |
| 20 | 0.000201 | 0.10506 | 0.03298 | 0.0001503 | 0.368988 | 0.263214 |
| 30 | 0.000160 | 0.11470 | 0.04269 | 0.0001919 | 0.369003 | 0.263206 |
| 40 | 0.000218 | 0.11282 | 0.04469 | 0.0001945 | 0.369021 | 0.263187 |
| 70 | 0.005560 | 0.09937 | 0.05092 | 0.0003407 | 0.369123 | 0.263111 |
| 90 | 0.001534 | 0.12421 | 0.08040 | 0.0000896 | 0.369232 | 0.263066 |
| 110 | 0.001932 | 0.11867 | 0.07243 | 0.0000548 | 0.369405 | 0.263031 |

This trend is currently healthy:

- median participation is stable and P10 is above its update-10 value;
- `tied_bias` remains the worst tensor and is noisy, as expected for a vocabulary-sized bias;
- RVQ NMSE changed by about $+0.115\%$ relative from update 10 to 110;
- ternary NMSE changed by about $-0.068\%$ relative;
- worst modules remain close to their family aggregates;
- loss is decreasing and throughput remains stable.

The global participation fluctuations after update 10 do not contradict those observations:
the global statistic is sensitive to relative gradient magnitude across tensors, while the
median and P10 summarize within-tensor concentration.

## 5. Gradient clipping

Every normal metric record includes:

- `grad_norm_pre_clip`: total norm returned before clipping;
- `grad_clip_coefficient`: $\min(1,1/\lVert g\rVert_2)$ for the configured clip norm 1.

A coefficient of 1 means no clipping. A coefficient of 0.15 means the entire gradient was
scaled to about 15% of its original magnitude. Early updates in this run clipped strongly at
about 0.15, but by updates 41--48 most steps were below the clipping threshold. That relaxation
is more informative than the early absolute value.

Clipping protects update magnitude but does not repair concentrated gradients: uniform scaling
leaves $R_{\mathrm{eff}}$ unchanged. Monitor both.

## 6. What Walsh--Hadamard does and does not protect

The Walsh--Hadamard transform redistributes coordinate energy before low-bit KV quantization
and preserves the $L_2$ norm:

$$
\lVert Hx\rVert_2=\lVert x\rVert_2.
$$

In this codebase it applies to KV activation/cache encoding. It does not directly flatten
master-weight gradients, Adam moments, FFN ternary master weights, RVQ weight residuals, or the
vocabulary bias. Gradient participation and weight QAT-gap diagnostics are therefore still
necessary.

## 7. Alerting policy

Use rolling baselines after the early warmup rather than universal thresholds. A practical
policy is:

### Stop immediately

- non-finite loss or gradient norm;
- non-finite weights or optimizer state;
- an exception from `clip_grad_norm_(..., error_if_nonfinite=True)`.

### Investigate

- median or P10 participation falls by more than $5     imes$ relative to its rolling baseline
  and remains low for several diagnostic records;
- the same large transformer matrix becomes the worst tensor and its participation falls by
  more than $10 imes$;
- clipping occurs on most steps after warmup, or the coefficient trends toward zero;
- family NMSE rises by 10--20% relative to baseline over multiple records;
- worst-module NMSE becomes more than about $2  imes$ its family aggregate;
- training loss falls while quantized validation loss rises consistently.

These are investigation triggers, not proofs of divergence. Correlate gradient concentration,
gradient magnitude, QAT gap, loss, validation loss, and throughput before changing the run.

## 8. Monitoring commands

Follow ordinary metrics:

~~~bash
tail -f pretrain_runs/dolma-8b/metrics.jsonl
~~~

Follow low-frequency stability records:

~~~bash
tail -f pretrain_runs/dolma-8b/diagnostics.jsonl
~~~

Print a compact diagnostic table without external packages:

~~~bash
python - <<'PY'
import json
from pathlib import Path

path = Path("pretrain_runs/dolma-8b/diagnostics.jsonl")
for line in path.read_text().splitlines()[-20:]:
    row = json.loads(line)
    grad = row["gradient_participation"]
    gap = row["qat_gap"]
    print(
        f"u={row['update']:>5} "
        f"global={grad['normalized_global']:.3e} "
        f"median={grad['layer_median']:.3e} "
        f"p10={grad['layer_p10']:.3e} "
        f"worst={grad['worst_name']}:{grad['worst']:.3e} "
        f"rvq={gap['rvq']['nmse']:.4f} "
        f"ternary={gap['ternary']['nmse']:.4f}"
    )
PY
~~~

The current diagnostics are deliberately compact. If a warning persists, reproduce from a
checkpoint with temporary detailed per-module logging rather than permanently dumping every
layer into the main run log.