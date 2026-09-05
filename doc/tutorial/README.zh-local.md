# SHADOW 模型实现导读

这一组教程解释 `finetune/modeling/` 中最容易卡住阅读的模型知识，并始终区分两件事：

1. 一个术语在机器学习中的一般含义；
2. 它在 SHADOW 当前代码中的具体实现。

它是现有[英文量化教程](README.md)的中文实现导读：英文系列更强调部署字节布局，本文系列
补充 Transformer 数据流、STE、长期记忆和代码阅读边界。

## 推荐阅读顺序

1. [从 token 到 logits：先建立全局地图](01-model-map.md)
2. [量化、QAT 与 STE：为什么取整后还能训练](02-quantization-and-ste.md)
3. [RVQ：用多个码本逐级逼近权重](03-rvq.md)
4. [1-bit / 2-bit KV cache：Hadamard 旋转、打包与冷数据检索](04-kv-cache.md)
5. [其余核心组件：RMSNorm、RoPE、GQA、Shiftmax、SwiGLU 与 StructStep](05-transformer-block.md)
6. [Hamming–Hebbian 长期记忆与两种 archive](06-memory.md)

[RVQ 与 STE 交互演示](rvq-ste-explorer.html)适合在读完第 2、3 篇后打开。它用一个二维玩具例子展示
“前向使用量化值、反向把梯度送给浮点主权重”和“每一级 RVQ 只拟合上一级残差”。

## 先记住四个结论

- 模型训练时保留浮点 weight，前向计算却尽量看到部署时的量化权重；这是量化感知训练（QAT）。
- STE 不是在数学上求出了 round 的真实导数，而是人为指定一个有用的替代梯度。
- RVQ 的 residual 指“上一层码本没有解释掉的误差”，不是 Transformer 的 residual connection。
- “位数公式”只计算码本索引的理论主体成本；实际文件还包含码本、行 scale、padding 和其他浮点参数。

## 代码入口速查

| 想理解什么 | 主要代码/符号 |
|---|---|
| STE、PoT 激活量化、Walsh–Hadamard | `finetune/modeling/common.py`：`_ExactSTE`、`pot`、`walsh_hadamard` |
| 1-bit KV codec | `finetune/modeling/common.py`：`KVCodec1` |
| Hamming–Hebbian memory | `finetune/modeling/common.py`：`HammingHebbianMemory` |
| RVQ | `finetune/modeling/common.py`：`RVQ`、`requant` |
| Attention / FFN block | `finetune/modeling/common.py`：`Block` |
| StructStep | `finetune/modeling/common.py`：`StructStep` |
| 整体模型与 cached decode | `finetune/modeling/model_250m.py`：`Shadow250M` |
| 微调时的量化替换与 requant 时机 | `finetune/finetune.py`：`main` 内的 `_tern`、`_enc2` 与训练循环 |
| RVQ 文件布局和 round-trip | `finetune/modeling/export_rvq.py` |
| 三值 FFN 与 `.shdw` 导出 | `finetune/modeling/export_ternary.py` |

符号名比行号稳定；后续实现变化时可直接在仓库中搜索这些名称。
