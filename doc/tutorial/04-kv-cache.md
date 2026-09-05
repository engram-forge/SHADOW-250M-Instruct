# 1-bit / 2-bit KV cache：旋转、打包与冷数据检索

## 1. KV cache 为什么值得单独量化

自回归解码每生成一个 token，都要让新 query 与过去 token 的 key/value 交互。若不缓存，过去 token 的 K/V
会反复重算；缓存后计算更快，但内存随上下文长度线性增长。

发布配置每层有 NKV=2 个 KV head，每个 head 宽 HD=64。若 K、V 都用 1 bit，则每层每 token 的纯 payload 为

$$
2\;(K,V)\times2\;heads\times64\;bits=256\;bits=32\;bytes.
$$

10 层合计 $320$ bytes/token，正好解释 README 中 100M token 约 32 GB 的主体数据量。

## 2. 为什么先做 Walsh–Hadamard 旋转

极低比特量化怕 outlier：少数特别大的坐标会控制 scale 或阈值，使其他坐标分辨率很差。归一化
Walsh–Hadamard transform（WHT）是正交变换：

$$
H^\top H=I,\qquad H^{-1}=H.
$$

它用加减法把能量混合到各个维度，通常让 outlier 不再集中在单一坐标。因为它是自逆的，解码再应用一次
同样的 transform 即可回到原空间。

common.py 的实现要求最后一维是 2 的幂；当前 HD=64 满足。复杂度为 $O(d\log d)$，不需要保存一个
$d\times d$ 稠密矩阵。

## 3. 2-bit codec

2-bit 路径先旋转：

$$
z=Hx.
$$

然后选择 2 的幂 scale

$$
s=2^{\lceil\log_2(\max|z|/1.5)\rceil}.
$$

整数 code 是 $c=\operatorname{clip}(\operatorname{round}(z/s),-2,1)$，但重建使用半格偏移：

$$
\hat z=(c+0.5)s\in\{-1.5s,-0.5s,0.5s,1.5s\},\qquad \hat x=H\hat z.
$$

为什么 pack 中存 $c+2\in\{0,1,2,3\}$，unpack 却计算 stored_code $-1.5$？因为 stored_code $=c+2$，
所以 $(c+2)-1.5=c+0.5$，两边完全对应。

每 4 个 2-bit code 放入一个 uint8：

```text
byte = c0 | (c1 << 2) | (c2 << 4) | (c3 << 6)
```

fake-quantized kv2 返回 ste(x, reconstructed)；真正缓存路径用 kv2_pack / kv2_unpack，并为每个向量额外保存 scale。

## 4. 1-bit KVCodec1

1 bit 只有两个状态，直接按正负号切分往往太粗。该 codec 为每个 KV head 和每个旋转后维度校准统计量。

### 4.1 变换

对输入 $x\in\mathbb{R}^{B\times H\times T\times d}$：

$$
z=H\left((x-\mu)\odot\sigma\right),
$$

其中：

- $\mu$ 是跨 batch 和 token 统计的逐 head、逐维均值；
- $\sigma\in\{-1,+1\}^d$ 是由 seed 固定生成的随机符号；
- $H$ 是 WHT；
- 随机符号再加旋转进一步打散具有固定方向的结构。

### 4.2 校准与编码

每个旋转坐标以中位数 ctv 为阈值：

$$
b=[z>\operatorname{median}(z)].
$$

中位数通常让 0/1 较均衡，从而充分利用单个 bit。decision_grid=256 先把决策值和阈值放到 $1/256$ 的格点上，
减少临界值比较对微小浮点差异的敏感性。

对低于和高于阈值的两组样本，分别记录重建质心 low 和 high，而不是固定重建为 -1/+1：

$$
\hat z=\begin{cases}
low,&b=0\\
high,&b=1.
\end{cases}
$$

训练模式下统计量用 momentum=0.01 的 EMA 更新；第一次校准直接采用当前 batch 的值。

### 4.3 解码

因为 $H^{-1}=H$ 且随机 sign 自己也是逆操作：

$$
\hat x=\mu+H(\hat z)\odot\sigma.
$$

pack 每 8 个 boolean 放入一个 uint8。HD=64 时，一个 head、一个 token 只需 8 bytes；K/V 各一份。

## 5. freeze_codec_updates 与 eval 状态

KVCodec1 的统计量是 buffer，不靠 optimizer 更新，而靠 calibrate 原地更新。在希望验证集、梯度累积或特殊前向
不改变 codec 状态时，可以临时将这些 codec 设为 eval。

当前 `finetune.py` 构造模型后明确把所有 `KVCodec1` 设为 eval，并在 validation 返回训练模式后再次设回
eval。这防止微调数据持续改变已经初始化的 1-bit codec。若 checkpoint 中的 codec 尚未初始化，第一次
实际使用仍会由 `_ensure` 校准一次；因此发布证据必须同时记录 `initialized` 状态和校准数据来源。

## 6. 热 KV、冷 archive 和两阶段检索

cached decode 将最近 max_ctx 个 packed K/V 留在热缓存。溢出的旧 token 通过 _archive_overflow 追加到
PagedKVArchive，页面式存储避免每次 append 都复制完整历史。

单 token query 的冷检索流程：

```mermaid
flowchart LR
    Q[new query] --> G[同一 KV head 下聚合 query heads]
    G --> P[1-bit pack]
    P --> H[全 archive 精确 Hamming top-k shortlist]
    H --> U[解包候选 K]
    U --> R[PoT 后点积 rerank]
    R --> S[取 final top-k 的 K/V]
    S --> A[与热 KV 拼接后做 attention]
```

第一阶段用 XOR + popcount 计算 Hamming distance，便宜地取 final_k 的最多 4 倍候选；第二阶段解码这些候选，
再用 query–key 点积排序。这样不必对全部冷 K 做浮点 attention。

PagedKVArchive.exact_hamming_topk 是按 page 扫描的精确 Hamming top-k，而不是近似索引；“shortlist”指它对
后续浮点 rerank 的候选缩减。相同距离时用全局 index 做稳定 tie-break。

## 7. 常见误解

- “1-bit KV”指 K 和 V 的每个旋转坐标主体用 1 bit，不代表整个缓存每 token 只有 1 bit。
- pack/unpack 才改变存储密度；forward 中 ste 的输出 dtype 仍是浮点。
- WHT 不是在减少维度，而是在同维正交旋转。
- Hamming 距离用于快速候选选择，最终 attention 仍使用解码后的近似 K/V。
- README 描述的预编译 runtime 与 Python modeling 代码应在格式和算法意图上对应，但不要据 Python tensor 的
  临时内存占用直接推断 CPU runtime 的实际 RAM。
