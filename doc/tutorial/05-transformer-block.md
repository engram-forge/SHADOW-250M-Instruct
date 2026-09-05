# 其余核心组件：RMSNorm、RoPE、GQA、Shiftmax、SwiGLU 与 StructStep

## 1. RMSNorm

RMSNorm 不减均值，只按均方根缩放：

$$
\operatorname{RMSNorm}(x)=w\odot x\cdot
\frac{1}{\sqrt{\operatorname{mean}(x^2)+\epsilon}}.
$$

common.py 先用 float32 计算平方与均值，再转回输入 dtype，以降低 bfloat16 累积误差。Block 在 attention 和 FFN
前各用一次 RMSNorm；Q、K 还各有一个 head-dim RMSNorm（QK-Norm），帮助控制点积尺度。

## 2. RoPE

Rotary Position Embedding 将相邻两个通道组成二维向量，并按位置相关角度旋转：

$$
\begin{bmatrix}x'_{2j}\\x'_{2j+1}\end{bmatrix}
=
\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}
\begin{bmatrix}x_{2j}\\x_{2j+1}\end{bmatrix}.
$$

角频率为

$$
\omega_j=10000^{-2j/HD},\qquad \theta_{p,j}=p\omega_j.
$$

只旋转 Q 和 K，不旋转 V。cs 生成从位置 0 开始的一段 cos/sin；cs_at 用于 cached decode，生成从绝对位置
start 开始的一段，避免每次重建完整位置表。

## 3. GQA：为什么 24 个 Q head 只有 2 个 KV head

Grouped-Query Attention 让多组 query head 共享较少的 K/V head。发布配置为 NH=24、NKV=2，
所以每个 KV head 服务 12 个 query head：

```python
k = k.repeat_interleave(NH // NKV, dim=1)
v = v.repeat_interleave(NH // NKV, dim=1)
```

与 24 个独立 K/V head 相比，KV 投影和缓存主体缩小 12 倍。代价是多个 query head 看到相同 K/V 表示。

## 4. causal mask

完整序列训练时，第 $t$ 个位置不能看到未来位置，代码用下三角 mask。cached decode 中 query 长度可能小于
已有 key 长度，因此 mask 需要右对齐：新 query 的第一个位置可以看到全部历史 key，但不能看到同批次中比它晚的 key。

单 token decode 没有“批内未来 token”，所以 causal=False 也不会泄漏未来信息；缓存中只有已经生成的 token。

## 5. Shiftmax 与普通 softmax

普通 softmax 是

$$
\operatorname{softmax}(z_i)=\frac{e^{z_i-\max z}}{\sum_j e^{z_j-\max z}}.
$$

精确路径 shiftmax 将 scale 量化到 $1/4096$，把缩放后的 dot product 向下取整，再用 base-2 指数并将指数差
截断到 $[-15,0]$。这让指数查表/移位式实现更自然，同时避免极小权重继续消耗动态范围。

alpha 是每个 query head 一个可训练参数，初始 0.25，而不是标准 attention 固定的 $1/\sqrt{HD}$。
它通过 STE 以量化值参与前向。

FAST_ATTN 路径调用 PyTorch scaled_dot_product_attention，主要服务更快训练。它把 $\ln 2$ 合并到 query：

$$
e^{(\ln2)z}=2^z.
$$

但快速路径没有精确模拟 floor 和 $-15$ 截断，所以用于速度时应把它理解为近似训练路径，而不是逐 bit 等价。

## 6. gated attention residual

attention 输出在 o projection 前乘

$$
\sigma(g),
$$

其中 g 初始化为 0，因此初始门值为 0.5。它按 NH × HD 通道学习，不是每个 token 独立的动态 gate。

## 7. SwiGLU 风格 FFN

FFN 计算为

$$
\operatorname{FFN}(h)=W_{down}\left(\operatorname{SiLU}(W_{gate}h)\odot W_{up}h\right).
$$

up 分支提供内容，gate 分支控制通过量。结果再加回 residual。当前微调/导出入口把这三个 g=32 的 RVQ 包装层
实际改成三值权重前向和三值 payload，详见 RVQ 教程第 7 节。

## 8. StructStep 是什么

所有 Transformer block 后还有一个 StructStep：

1. 从 hidden state 生成 query；
2. 对整段 hidden context 再做一次单头 causal self-attention；
3. 将原 hidden 与读出的 context 拼接；
4. 经 cin → SiLU → cout 融合，并做 residual + RMSNorm；
5. verify 为每个位置输出一个 sigmoid 标量 conf。

公式概括为

$$
q=W_qh,\quad A=\operatorname{softmax}(qh^\top/\sqrt D),\quad r=Ah,
$$

$$
h'=\operatorname{RMSNorm}\left(h+W_{out}\operatorname{SiLU}(W_{in}[h;r])\right).
$$

当前 Shadow250M.forward / logits 主路径没有用 conf 调整 token logits；trunk 会返回它，但训练 loss 只使用 hidden
生成的 logits。因此不要仅凭 verify 这个名字推断它已有独立监督或在生成时执行拒答逻辑。

cached inference 中 StructStep 可以让新 token 读取 trunk context；若长期 memory 召回了旧 chunk，召回内容会先拼入
context，再供 StructStep 注意。

## 9. Block 中的张量形状（发布配置）

设输入 x 为 $(B,T,1536)$：

| 张量 | repeat 前形状 | 含义 |
|---|---|---|
| q | $(B,24,T,64)$ | 每个 query head 独立 |
| k | $(B,2,T,64)$ | 两个共享 KV head |
| v | $(B,2,T,64)$ | 两个共享 KV head |
| repeated k/v | $(B,24,T,64)$ | 每个 KV head 复制给 12 个 Q head |
| attention weights | $(B,24,T,T)$ | 完整训练路径的注意力矩阵 |
| merged y | $(B,T,1536)$ | 合并 24 个 head |
| FFN up/gate | $(B,T,4224)$ | SwiGLU 中间表示 |

沿形状阅读 common.py 往往比逐行追一字母变量更容易发现每一步在做什么。
