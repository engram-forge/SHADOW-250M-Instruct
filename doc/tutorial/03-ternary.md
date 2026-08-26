# 3. Ternary FFN weights

The FFN `up`, `gt`, and `dn` layers are declared as `RVQ(g=32, st=1)`, but
`finetune.py` replaces their behavior with ternary fake quantization. Each
weight becomes a negative row scale, zero, or a positive row scale.

## Quantization rule

For output row $n$:

$$
a_n=\max(\operatorname{mean}_j|W_{n,j}|,10^{-5}),
$$

$$
T_{n,j}=\operatorname{clip}(\operatorname{round}(W_{n,j}/a_n),-1,1),
\qquad \widehat W_{n,j}=a_nT_{n,j}.
$$

The source computes `sc = 1 / a`, then stores `rs = a`. STE makes the forward
pass use $\widehat W$ and sends gradients to the floating-point $W$.

## Worked tensor example

Take one output row:

$$
W=[-0.90,-0.20,0.10,0.70,1.40].
$$

Its scale is $a=(0.9+0.2+0.1+0.7+1.4)/5=0.66$.

| $W_j$ | $W_j/a$ | rounded/clipped $T_j$ | reconstructed $aT_j$ |
|---:|---:|---:|---:|
| -0.90 | -1.364 | -1 | -0.66 |
| -0.20 | -0.303 | 0 | 0 |
| 0.10 | 0.152 | 0 | 0 |
| 0.70 | 1.061 | 1 | 0.66 |
| 1.40 | 2.121 | 1 | 0.66 |

The master values remain unchanged in the checkpoint; these reconstructed
values are what the quantized forward and deployment runtime use.

## Intermediate 2-bit packing

Adding one maps ternary values $[-1,0,+1]$ to codes $[0,1,2]$. Code 3 is
unused. Four codes occupy one byte:

$$
p=c_0+(c_1\ll2)+(c_2\ll4)+(c_3\ll6).
$$

For ternary values $[-1,0,+1,-1]$, codes are $[0,1,2,0]$, so
$p=0+4+32+0=36=\texttt{0x24}$. The first weight is in the least significant
two bits. This kind-3 format costs 2 symbol bits/weight plus one FP32 row scale.

## Final base-3 packing

The final repacker stores five codes in one byte:

$$
p=c_0+3c_1+9c_2+27c_3+81c_4.
$$

For $[-1,0,+1,-1,+1]$, codes $[0,1,2,0,2]$ produce
$0+3+18+0+162=183=\texttt{0xB7}$. Since $3^5=243\le256$, every possible
five-trit combination fits. Incomplete groups are padded with code 1, meaning
zero. The final symbol rate is $8/5=1.6$ bits/weight.

## Release tensor sizes

An FFN up matrix has shape $(4224,1536)$, or 6,488,064 weights. Its ideal final
symbol payload is $\lceil6{,}488{,}064/5\rceil=1{,}297{,}613$ bytes, plus
$4224\times4=16{,}896$ bytes of row scales. Packing is performed row by row, so
the exact padding calculation is $4224\lceil1536/5\rceil$ bytes.

[Next: One-bit and two-bit KV codecs](04-kv-codecs.md)

