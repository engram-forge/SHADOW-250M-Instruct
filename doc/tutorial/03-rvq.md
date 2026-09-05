# RVQ：用多个码本逐级逼近权重

## 1. 从标量量化到向量量化

标量量化逐个处理数字；向量量化（Vector Quantization, VQ）把连续的 $g$ 个数看成一个向量，
并用码本中最近的向量替代它。

给定码本 $C=\{c_0,\ldots,c_{K-1}\}$，一个权重块 $w\in\mathbb{R}^g$ 被编码为

$$
i^*=\arg\min_i\lVert w-c_i\rVert_2,\qquad \hat w=c_{i^*}.
$$

存储时不必为每个块保存 $g$ 个浮点数，只需保存索引 $i^*$。码本被许多块共享，所以其成本可以摊薄。

## 2. Residual Vector Quantization

单个码本选完一个向量后通常仍有误差。RVQ 让第 2 个码本拟合第 1 个码本留下的残差，
第 3 个再拟合前两级留下的残差，依此类推：

$$
r_0=w,
$$

$$
i_t=\arg\min_j\left\lVert\frac{r_t}{\rho_t}-c_{t,j}\right\rVert_2,
\qquad q_t=\rho_t c_{t,i_t},
$$

$$
r_{t+1}=r_t-q_t,\qquad
\hat w=\sum_{t=0}^{S-1}q_t.
$$

$S$ 是 stage 数，$K$ 是每级 code 数，$\rho_t$ 是当前残差的 RMS。第二级不是重新量化原始 $w$，
而是专门修补第一级的误差。

## 3. SHADOW 的 RVQ 输入如何切块

一个 RVQ 线性层仍维护完整浮点矩阵

$$
W\in\mathbb{R}^{o\times i}.
$$

代码先计算每个输出行的平均绝对值：

$$
s_j=\max\left(\operatorname{mean}_k |W_{j,k}|,10^{-8}\right).
$$

然后做逐行归一化，再将每行切成长度为 $g$ 的小向量：

$$
R_0=\operatorname{reshape}(W/s,[-1,g]).
$$

行 scale 处理不同行的动态范围，码本负责学习归一化后的局部形状。当前配置中输入宽度都能被 $g$ 整除，
所以块不会跨越权重行。

## 4. _fit 和 enc 的职责不同

### _fit：第一次建立码本

首次量化时 cb_init 为 false，_fit 对最多 8192 个权重块运行简化 k-means：

1. 对当前残差求一个全局 RMS；
2. 用 k-means 拟合 K=16 个中心；
3. 每个块选择最近中心；
4. 从残差减去该中心的重建；
5. 下一 stage 继续拟合剩余残差。

### enc：用现有码本重编码当前权重

enc 在 no_grad 下重新计算行 scale、每级 residual RMS、最近中心索引与重建矩阵 _q。
它不会通过梯度训练码本，也不会在每次调用时重新跑 k-means。

这意味着当前实现的 codebook cb 是 buffer，不是 nn.Parameter；初始化后通常固定。训练更新的是浮点主权重，
每次 optimizer.step 后再用固定码本重新分配索引。

## 5. 前向为什么仍能更新浮点权重

```python
def qw(self):
    if self._q is None:
        self.enc()
    return ste(self.weight, self._q)

def forward(self, x):
    return F.linear(x, self.qw().to(x.dtype))
```

前向的矩阵乘法使用 _q；反向时 STE 把对量化权重的梯度交给 weight。码本选择、argmin、RMS 和
_q 的构建都在 no_grad 中，不属于可微计算图。

## 6. bits() 公式怎么来的

每个长度为 $g$ 的块，在每个 stage 保存一个 $K$ 选 1 的索引，需要 $\log_2K$ bit。共有 $S$ 级，
所以主体索引的理论 bit/weight 是

$$
b_{index}=\frac{S\log_2K}{g}.
$$

本项目 $K=16$，每个索引恰好 4 bit：

| 用途 | g | stages | 理论索引 bit/weight |
|---|---:|---:|---:|
| q/k/v/o、StructStep 投影 | 8 | 2 | $2\times4/8=1$ |
| up/gate/down 的 RVQ 类默认值 | 32 | 1 | $1\times4/32=0.125$ |

但 bits() 不是整个文件的精确压缩率。还需支付：

- 每个输出行一个浮点 scale；
- 每级 $K\times g$ 个 codebook 浮点值；
- 输出行按 64 对齐的 padding；
- norm、bias、部分线性层等非 RVQ 参数；
- 文件头和 tensor metadata。

## 7. 一个很重要的项目特例：FFN 最终不是 0.125-bit RVQ

Block 构造时 up、gt、dn 的确是 RVQ(g=32, st=1)。但实际微调入口 finetune.py 会 monkey-patch：

- g=32 的模块前向改为逐行三值量化 $\{-1,0,+1\}$ + scale；
- 它们的 RVQ enc 被跳过；
- export_ternary.py 也把这些模块导出为 ternary payload；
- 只有 g=8 的模块按 RVQ 格式导出。

三值前向为

$$
s_j=\frac{1}{\operatorname{mean}|W_j|},\qquad
T_j=\operatorname{clip}(\operatorname{round}(s_jW_j),-1,1),\qquad
\hat W_j=T_j/s_j.
$$

默认导出用 2 bit 保存一个三值 code（每 byte 四个 code）；compact 模式可用 base-3 把五个三值 code
放入一个 byte，主体成本约为 $8/5=1.6$ bit/weight。因此 model_250m.py 打印的 RVQ.bits() 是类构造参数
的理论值，不等于发布 .shdw 中 FFN 的实际 payload 位数。

## 8. export_rvq.py 保存了什么

导出会重新捕获当前索引，并做以下布局转换：

- 每 64 个输出行对齐；
- 两个 4-bit 索引合并到一个 uint8 的高、低 nibble；
- 每级 residual RMS 预乘进转置后的 codebook cbT；
- 每行 scale 单独保存。

运行时重建可概括成

$$
\hat W_j=s_j\sum_t \operatorname{concat}_b C'_{t,:,i_{t,j,b}},
$$

其中 $b$ 遍历一行里的长度 $g$ 小块。export_rvq.py 自带 round-trip：打包再解包后应与训练器的 _q
在浮点误差范围内一致。

## 9. 阅读与调试清单

- 不要把 RVQ residual 与网络 residual connection 混淆。
- _q 不是 Parameter，也不进入 state_dict；checkpoint 保存 weight 和 cb，加载后必须 requant。
- 手工改 weight.data 后先调用 enc 或 requant，否则前向可能继续用旧 _q。
- cb_init 为 true 时 _fit 不会重跑；若实验目标是重学码本，需要显式设计重置策略，而不是只训练更多 step。
- 比较压缩率时区分理论索引位数、payload 位数和整个部署文件位数。
