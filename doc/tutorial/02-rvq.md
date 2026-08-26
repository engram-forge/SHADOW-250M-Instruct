# 2. Residual vector quantization (RVQ)

RVQ compresses Q, K, V, O and structural-step weight matrices. In the release
design, groups have width $g=8$, there are 16 codewords per stage, and two
residual stages are used.

## Tensor reshaping

For $W\in\mathbb{R}^{o\times i}$, each output row is divided into $i/g$
contiguous vectors. The release Q matrix maps as follows:

$$
(o,i)=(1536,1536)
\rightarrow(1536\times192,8)=(294912,8).
$$

Each of those 294,912 vectors receives two 4-bit codebook indices. The
floating-point master matrix remains trainable; `cb` has shape $(2,16,8)$ and
the reconstructed cache `_q` has shape $(1536,1536)$.

## Row scaling

Every output row has one mean-absolute scale:

$$
s_n=\max\left(\frac1i\sum_j|W_{n,j}|,10^{-8}\right).
$$

The normalized matrix $W/s$ is split into width-$g$ vectors. Row scaling lets
global codebooks represent vector shape while $s_n$ restores output-channel
magnitude.

### Tiny tensor example

Use a toy matrix with $o=2$, $i=4$, and $g=2$:

$$
W=\begin{bmatrix}2&-2&1&-1\\4&0&-2&2\end{bmatrix}.
$$

The row scales are $s=[1.5,2.0]$. Normalization and grouping produce:

$$
W/s=\begin{bmatrix}
1.333&-1.333&0.667&-0.667\\
2&0&-1&1
\end{bmatrix}
\rightarrow
r_0=\begin{bmatrix}
1.333&-1.333\\0.667&-0.667\\2&0\\-1&1
\end{bmatrix}.
$$

The real model does the same reshape, just with width-8 groups.

## Two residual stages

For stage $t$:

$$
\rho_t=\sqrt{\operatorname{mean}(r_t^2)},\qquad
k_t=\arg\min_k\|r_t/\rho_t-C_{t,k}\|_2,
$$

$$
q_t=\rho_tC_{t,k_t},\qquad r_{t+1}=r_t-q_t.
$$

Later stages encode the residual left by earlier stages. To see the mechanism,
consider one normalized group $r_0=[1.2,-0.8]$ and simplified codebooks:

| Stage | RMS | nearest code | contribution | residual afterward |
|---:|---:|---:|---:|---:|
| 0 | $1.020$ | $C_{0,3}=[1,-1]$ | $[1.020,-1.020]$ | $[0.180,0.220]$ |
| 1 | $0.201$ | $C_{1,9}=[1,1]$ | $[0.201,0.201]$ | $[-0.021,0.019]$ |

The reconstructed normalized vector is
$[1.020,-1.020]+[0.201,0.201]=[1.221,-0.819]$. If its row scale is 1.5, the
stored representation reconstructs $[1.832,-1.229]$, close to the original
scaled group $[1.8,-1.2]$. Values are not binary: the two indices select two
floating-point vectors whose sum is scaled.

## Codebook fitting

`_fit()` samples at most 8192 groups. At each stage it runs 12 k-means
iterations on $r_t/\rho_t$, stores 16 centroids, assigns all sampled groups, and
fits the next stage to the remaining residual. `cb_init` prevents automatic
refitting later. Normal encoding reuses the codebooks but recomputes assignments
and residual RMS values from the current master matrix.

## Why “one bit per weight”

With $c$ codewords, every index requires $\log_2(c)$ bits. The nominal rate is

$$
b=\frac{st\log_2(c)}{g}.
$$

For $(st,c,g)=(2,16,8)$, $b=1$ bit/weight. For one width-8 group:

```text
stage 0 index: 4 bits ─┐
                       ├─ 8 index bits / 8 represented weights = 1 bit/weight
stage 1 index: 4 bits ─┘
```

This excludes FP32 codebooks, FP32 row scales, padding, and record metadata. It
is an amortized index rate, not the exact file rate.

## Exported index tensor

Rows are padded to a multiple of 64. For every stage, group, and pair of rows
$n$ and $n+32$, two 4-bit indices share a byte:

```text
bit 7              bit 4 bit 3               bit 0
+----------------------+--------------------------+
| index for row n + 32 |     index for row n      |
+----------------------+--------------------------+
```

Example: row 5 selects code 3 and row 37 selects code 12. The packed byte is
$3+(12\ll4)=195=\texttt{0xC3}$. Unpacking masks `0x0F` for row 5 and shifts
right four bits for row 37. The full index shape is
$(st,\lceil o/64\rceil,i/g,32)$.

[Next: Ternary FFN weights](03-ternary.md)

