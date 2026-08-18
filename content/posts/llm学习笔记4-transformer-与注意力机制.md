---
title: "『LLM学习笔记4』Transformer 与注意力机制"
category: "internship"
tags:
  - "LLM"
date: "2026-08-12"
summary: "系统梳理 Transformer 与注意力机制：从 Q/K/V、缩放点积、Mask 和多头注意力出发，进一步讲解 GQA/MQA、RoPE、归一化、FFN、FlashAttention、KV Cache 及训练与推理差异。"
pdf: ""
pdfTitle: ""
---

**符号约定**

| 符号 | 含义 |
|---|---|
| $B$ | batch size |
| $L_q$ | Query 序列长度，也称 target length |
| $L_{kv}$ | Key/Value 序列长度，也称 source length |
| $d_{model}$ | Transformer 隐藏维度 |
| $h$ | Query head 数量 |
| $h_{kv}$ | Key/Value head 数量；普通 MHA 中 $h_{kv}=h$ |
| $d_h$ | 单个 head 的维度，通常 $d_h=d_{model}/h$ |
| $d_{ff}$ | FFN 中间层维度 |

---

## 一、把 Scaled Dot-Product Attention 从公式推到数值实现

**1. Attention 不是“神秘的相关性”，而是一次可微分的软检索**

把一组信息看作数据库：每条记录由一个 Key 和一个 Value 组成。Query 到来以后，模型不会只取一条记录，而是计算 Query 与所有 Key 的匹配程度，然后对 Value 做加权求和：

$$
\operatorname{Attention}(q,K,V)=\sum_{j=1}^{L_{kv}}\alpha_jv_j,
\qquad
\sum_j\alpha_j=1,
\qquad
\alpha_j\ge 0.
$$

这里的 $\alpha_j$ 不是人工指定，而是由匹配分数经过 softmax 得到：

$$
s_j=\frac{qk_j^\top}{\sqrt{d_k}},
\qquad
\alpha_j=\frac{\exp(s_j)}{\sum_t\exp(s_t)}.
$$

因此，Attention 可以分成两件事：

1. **寻址**：根据 $q$ 与 $k_j$ 计算应该关注谁；
2. **读取**：根据权重 $\alpha_j$ 汇总对应的 $v_j$。

这一点能解释为什么一定要区分 Key 和 Value：**Key 用于比较，Value 用于传递内容。** 如果把两者强行设成同一种表示，模型就失去了分别学习“如何被检索”和“被检索后提供什么”的自由度。

Self-Attention 中，$Q,K,V$ 都来自同一个序列 $X$；Cross-Attention 中，$Q$ 来自当前序列，$K,V$ 来自另一段序列：

$$
\begin{aligned}
\text{Self-Attention: }&Q=XW_Q,\ K=XW_K,\ V=XW_V,\\
\text{Cross-Attention: }&Q=X_{query}W_Q,\ K=X_{memory}W_K,\ V=X_{memory}W_V.
\end{aligned}
$$

所以“Self”描述的不是“token 只看自己”，而是 **Q、K、V 的来源相同**。每个 token 依然可以看同一序列中的其他 token。

**2. 为什么需要三个投影矩阵 $W_Q,W_K,W_V$**

设输入为：

$$
X\in\mathbb{R}^{B\times L\times d_{model}}.
$$

线性投影为：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V.
$$

单头情形下，常见矩阵形状是：

$$
W_Q,W_K\in\mathbb{R}^{d_{model}\times d_k},
\qquad
W_V\in\mathbb{R}^{d_{model}\times d_v}.
$$

投影不是多此一举。原始 embedding 中包含词义、位置、句法和上下文等混合信息，但“用于匹配的特征”和“需要传递的特征”不必相同。举一个抽象例子：

- 代词 “它” 的 Query 可能突出“我需要寻找一个名词性先行词”；
- 候选词的 Key 可能突出“我是名词、单数、非生命或生命”；
- 候选词的 Value 则可以保存更丰富的语义内容。

$W_Q,W_K,W_V$ 的训练过程，本质上是在学习三套不同的坐标系。点积不是直接比较原始 token，而是在学习后的 Query-Key 空间中比较。

进一步看：

$$
q_i k_j^\top
= (x_iW_Q)(x_jW_K)^\top
= x_iW_QW_K^\top x_j^\top.
$$

令 $M=W_QW_K^\top$，则匹配函数相当于一个可学习的双线性形式：

$$
s_{ij}=x_iMx_j^\top.
$$

这比直接计算 $x_ix_j^\top$ 灵活得多，因为模型可以学习“哪些维度组合对匹配最重要”。

这里最重要的变化是：$x_i$​ 的第 $a$ 维，可以和 $x_j$​ 的第 $b$ 维进行匹配。

也就是说，模型可以学习：

- 哪些维度更重要；
- 哪些维度可以忽略；
- 一个维度应该和另一个不同维度如何匹配；
- 哪些特征组合应该得到正分或负分。

**3. 点积到底表示什么：方向、长度与相似度**

两个向量的点积可以写为：

$$
qk^\top=\lVert q\rVert\lVert k\rVert\cos\theta.
$$

所以点积同时受到三件事影响：

- Query 的模长；
- Key 的模长；
- 两者夹角。

它并不等于纯粹的余弦相似度，因为没有除以模长。模长也能参与表达“置信度”或“显著程度”。这使点积注意力具有更大的学习自由度，但也带来了一个数值问题：维度增大后，点积的典型尺度会增大。

**4. 为什么必须除以 $\sqrt{d_k}$：完整方差推导**

这是最常被背答案、却最少被真正讲清的地方。

为了分析，假设 Query 和 Key 的每一维满足：

$$
\mathbb{E}[q_r]=\mathbb{E}[k_r]=0,
\qquad
\operatorname{Var}(q_r)=\operatorname{Var}(k_r)=1,
$$

并近似认为各维独立。点积为：

$$
z=qk^\top=\sum_{r=1}^{d_k}q_rk_r.
$$

因为 $q_r$ 与 $k_r$ 均值为 0，独立时有：

$$
\mathbb{E}[q_rk_r]=0.
$$

又因为：

$$
\operatorname{Var}(q_rk_r)
=\mathbb{E}[q_r^2k_r^2]
=\mathbb{E}[q_r^2]\mathbb{E}[k_r^2]
=1,
$$

于是：

$$
\operatorname{Var}(z)
=\operatorname{Var}\left(\sum_{r=1}^{d_k}q_rk_r\right)
=\sum_{r=1}^{d_k}\operatorname{Var}(q_rk_r)
=d_k.
$$

因此点积的标准差约为：

$$
\operatorname{Std}(z)=\sqrt{d_k}.
$$

将点积除以 $\sqrt{d_k}$ 后：

$$
\operatorname{Var}\left(\frac{z}{\sqrt{d_k}}\right)
=\frac{1}{d_k}\operatorname{Var}(z)=1.
$$

所以这个缩放因子不是拍脑袋设计的，它让不同 head 维度下 logits 的典型尺度大致保持稳定。

**为什么 logits 过大会伤害训练？** 假设一行分数是 $[1,2,3]$，softmax 还比较平滑；若整体放大 20 倍，变成 $[20,40,60]$，softmax 几乎成为 one-hot。softmax 的导数为：

$$
\frac{\partial \alpha_i}{\partial s_j}
=\alpha_i(\delta_{ij}-\alpha_j).
$$

当某个 $\alpha_i\approx 1$、其他权重接近 0 时，大部分导数也接近 0，梯度难以有效调整匹配关系。这就是常说的 **softmax 饱和**。

从 temperature 的角度看：

$$
\operatorname{softmax}\left(\frac{s}{T}\right).
$$

$T$ 越小，分布越尖锐；$T$ 越大，分布越平滑。除以 $\sqrt{d_k}$ 等价于根据维度自动调整 temperature，避免维度一大，分布就无意中变得过尖。


![粘贴图片](/my-blog/resources/uploads/pasted-1786559891726.png)



**5. 从单个 Query 推到批量矩阵公式**

设：

$$
Q\in\mathbb{R}^{B\times L_q\times d_k},\quad
K\in\mathbb{R}^{B\times L_{kv}\times d_k},\quad
V\in\mathbb{R}^{B\times L_{kv}\times d_v}.
$$

先计算：

$$
S=\frac{QK^\top}{\sqrt{d_k}}.
$$

这里的 $K^\top$ 只转置最后两个维度，因此：

$$
K^\top\in\mathbb{R}^{B\times d_k\times L_{kv}},
$$

从而：

$$
S\in\mathbb{R}^{B\times L_q\times L_{kv}}.
$$

矩阵 $S$ 的第 $i$ 行、第 $j$ 列表示：第 $i$ 个 Query 对第 $j$ 个 Key 的匹配分数。softmax 必须沿最后一维 $L_{kv}$ 做：

$$
A=\operatorname{softmax}(S,\operatorname{dim}=-1).
$$

原因是：对每一个 Query，我们要在所有 Key 中分配权重。于是每一行满足：

$$
\sum_{j=1}^{L_{kv}}A_{ij}=1.
$$

最后：

$$
O=AV,
$$

形状为：

$$
O\in\mathbb{R}^{B\times L_q\times d_v}.
$$

请注意，输出长度由 $L_q$ 决定，而不是由 $L_{kv}$ 决定。这在 Cross-Attention 中尤其重要：Decoder 有多少个 Query 位置，就得到多少个输出位置。

**6. Mask 的本质是向 logits 添加偏置，不是把权重乘零**

标准写法可以统一成：

$$
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right).
$$

允许关注的位置令 $M_{ij}=0$；禁止关注的位置令 $M_{ij}=-\infty$。因为：

$$
\exp(-\infty)=0,
$$

所以 softmax 后该位置权重严格为 0。

为什么不先 softmax 再把权重乘 0？假设 softmax 后某行是 $[0.2,0.3,0.5]$，把最后一个位置乘 0 会得到 $[0.2,0.3,0]$，总和只剩 0.5，不再是概率分布；若再归一化，又多了一步。直接在 logits 上加 $-\infty$，softmax 会自动在允许的位置重新分配全部概率质量。

常见 Mask 有三类：

- **Causal Mask**：第 $i$ 个位置不能看 $j>i$ 的未来位置；
- **Padding Mask**：不能关注 `<PAD>`；
- **Arbitrary Attention Bias**：相对位置偏置、局部窗口或结构约束，也可以统一写成对 logits 的加性偏置。

长度为 4 的因果可见矩阵为：

$$
C=
\begin{bmatrix}
1&0&0&0\\
1&1&0&0\\
1&1&1&0\\
1&1&1&1
\end{bmatrix}.
$$

对应的加性 Mask 为：

$$
M_{ij}=\begin{cases}
0,&C_{ij}=1,\\
-\infty,&C_{ij}=0.
\end{cases}
$$

**必须注意的边界情况：整行全被 Mask。** 如果一行全部是 $-\infty$，softmax 需要计算 $-\infty-(-\infty)$，可能产生 `NaN`。工程上需要保证至少有一个可见位置，或者对全 Mask 行单独处理。

**7. 为什么 softmax 实现要先减最大值**

直接计算 $\exp(s_i)$ 可能溢出。例如 float32 中，$\exp(1000)$ 无法表示。利用 softmax 的平移不变性：

$$
\frac{\exp(s_i-c)}{\sum_j\exp(s_j-c)}
=\frac{\exp(s_i)}{\sum_j\exp(s_j)},
$$

通常取 $c=\max_j s_j$，使最大的指数为 $\exp(0)=1$，其他项不大于 1。这就是稳定 softmax：

$$
\operatorname{softmax}(s)_i
=\frac{\exp(s_i-m)}{\sum_j\exp(s_j-m)},
\qquad m=\max_j s_j.
$$

这也是理解 FlashAttention 在线 softmax 的基础，第四轮会继续展开。

**8. 三 Token 手算：不仅算结果，还解释矩阵每个轴**

设三个 token 的 $Q,K,V$ 为：

$$
Q=
\begin{bmatrix}
1&0\\
1&1\\
0&1
\end{bmatrix},
\quad
K=
\begin{bmatrix}
1&0\\
1&1\\
0&1
\end{bmatrix},
\quad
V=
\begin{bmatrix}
1&2\\
0&3\\
4&1
\end{bmatrix}.
$$

因为 $d_k=2$：

$$
S=\frac{QK^\top}{\sqrt{2}}
=\frac{1}{\sqrt{2}}
\begin{bmatrix}
1&1&0\\
1&2&1\\
0&1&1
\end{bmatrix}.
$$

第二个 Query 对应第二行：

$$
s_2=[0.7071,1.4142,0.7071].
$$

做 softmax：

$$
\alpha_2\approx[0.2483,0.5035,0.2483].
$$

于是：

$$
\begin{aligned}
o_2
&=0.2483[1,2]+0.5035[0,3]+0.2483[4,1]\\
&\approx[1.2415,2.2554].
\end{aligned}
$$

这个结果不是“选中了一个 token”，而是三个 Value 的凸组合。权重全为非负且和为 1，因此输出落在这些 Value 的凸包中。后续的输出投影、残差和 FFN 再进一步改变表示空间。

**9. 教学版实现：把数值稳定、两种 Mask 和形状检查写清楚**

```python
import math
from typing import Optional

import torch


def manual_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    causal: bool = False,
    key_padding_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    教学版 Scaled Dot-Product Attention。

    参数形状：
        query: [B, H, L_q, d_h]
        key:   [B, H_kv, L_kv, d_h]
        value: [B, H_kv, L_kv, d_v]

    为了聚焦最基本的 MHA，本函数要求 H == H_kv。
    GQA/MQA 会在第二轮单独解释。

    key_padding_mask:
        形状为 [B, L_kv]；True 表示真实 token，False 表示 padding。
        注意：不同 PyTorch API 对 bool mask 的 True/False 语义并不完全一致，
        所以实际使用官方 API 时一定要检查文档，而不能只靠记忆。
    """
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query、key、value 都必须是四维张量 [B, H, L, D]")

    B, H, L_q, d_h = query.shape
    B_k, H_k, L_kv, d_h_k = key.shape
    B_v, H_v, L_v, d_v = value.shape

    if (B, H, L_kv) != (B_k, H_k, L_v) or (B_k, H_k) != (B_v, H_v):
        raise ValueError("batch、head 或 Key/Value 序列长度不匹配")
    if d_h != d_h_k:
        raise ValueError("Query 和 Key 的 head_dim 必须相同，才能做点积")

    # [B, H, L_q, d_h] @ [B, H, d_h, L_kv]
    # -> [B, H, L_q, L_kv]
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / math.sqrt(d_h)

    # 因果 Mask 只允许 Query i 看到 Key j <= i。
    # 当 L_q != L_kv 时，Mask 的对齐方式要结合具体任务定义；
    # 这里给出最常见的自回归 self-attention 情形。
    if causal:
        if L_q != L_kv:
            raise ValueError("本教学实现的 causal=True 要求 L_q == L_kv")
        visible = torch.tril(
            torch.ones(L_q, L_kv, dtype=torch.bool, device=query.device)
        )
        scores = scores.masked_fill(~visible, float("-inf"))

    if key_padding_mask is not None:
        if key_padding_mask.shape != (B, L_kv):
            raise ValueError("key_padding_mask 应为 [B, L_kv]")

        # [B, L_kv] -> [B, 1, 1, L_kv]
        # 1 维 head 和 Query 轴会通过 broadcasting 自动扩展。
        visible_key = key_padding_mask[:, None, None, :]
        scores = scores.masked_fill(~visible_key, float("-inf"))

    # softmax 沿 Key 轴做，每个 Query 在所有 Key 之间分配权重。
    weights = torch.softmax(scores, dim=-1)

    # 防御性检查：如果出现全 Mask 行，softmax 可能产生 NaN。
    if torch.isnan(weights).any():
        raise RuntimeError("注意力权重出现 NaN，请检查是否存在整行全部被 Mask")

    # [B, H, L_q, L_kv] @ [B, H, L_kv, d_v]
    # -> [B, H, L_q, d_v]
    output = torch.matmul(weights, value)
    return output, weights


# 最小测试
if __name__ == "__main__":
    torch.manual_seed(0)
    q = torch.randn(2, 4, 5, 8)
    k = torch.randn(2, 4, 5, 8)
    v = torch.randn(2, 4, 5, 8)

    # 第二个样本最后两个位置是 PAD。
    padding_mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
        ]
    )

    out, attn = manual_scaled_dot_product_attention(
        q, k, v, causal=True, key_padding_mask=padding_mask
    )

    print("output:", out.shape)   # [2, 4, 5, 8]
    print("weights:", attn.shape) # [2, 4, 5, 5]
    print("row sums:", attn.sum(dim=-1)[0, 0])
```

**10. 第一轮面试题：按真实面试的回答长度组织**

| 面试问题                                  | 面试现场可以这样回答                                                                                                                     |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| Q、K、V 为什么要分开？                         | Q 用来描述当前 token 想检索什么，K 用来决定一条信息如何被匹配，V 是匹配后真正被读取的内容。三个独立投影让模型能在不同子空间中学习“寻址”和“传递”，比直接用同一个 embedding 更灵活。                        |
| 为什么使用点积注意力？                           | 点积可以用一次矩阵乘法并行计算所有 token 两两之间的匹配，GPU 利用率高。经过可学习的 $W_Q$ 和 $W_K$ 投影后，它实际上对应一个可学习的双线性匹配函数，不只是原始 embedding 的简单相似度。                  |
| 为什么除以 $\sqrt{d_k}$？                   | 在各维近似独立、方差为 1 的假设下，$q\cdot k$ 的方差约为 $d_k$，标准差约为 $\sqrt{d_k}$。除以 $\sqrt{d_k}$ 能把 logits 的尺度稳定在同一量级，避免 softmax 过于尖锐和梯度饱和。        |
| Mask 为什么加在 softmax 之前？                | 禁止位置在 logits 上加 $-\infty$，经过 softmax 后权重自然为 0，并且剩余可见位置会重新归一化。若 softmax 后再直接乘 0，整行权重和会小于 1，还需要额外归一化。                            |
| softmax 为什么沿最后一维做？                    | 对每一个 Query，需要在所有 Key 之间分配注意力，因此归一化轴是 Key 的序列维 $L_{kv}$。做完后每一行代表一个 Query 对所有 Key 的概率分布。                                         |
| Self-Attention 与 Cross-Attention 的区别？ | Self-Attention 的 Q、K、V 来自同一序列；Cross-Attention 的 Q 来自当前序列，K、V 来自外部 memory，例如机器翻译中 Decoder 用自己的状态查询 Encoder 输出。输出长度由 Query 长度决定。 |

---

## 二、多头注意力、Cross-Attention、GQA/MQA 与 KV Cache

**1. 多头注意力不是“把同一个注意力复制很多遍”**

普通单头注意力把整个 $d_{model}$ 维表示投影到一个 Query-Key 空间，并只做一次 softmax。多头注意力把表示划分到多个子空间，每个 head 各自拥有投影和各自的 softmax：

$$
\operatorname{head}_r
=\operatorname{Attention}(XW_Q^{(r)},XW_K^{(r)},XW_V^{(r)}),
$$

$$
\operatorname{MHA}(X)
=\operatorname{Concat}(\operatorname{head}_1,\ldots,\operatorname{head}_h)W_O.
$$

容易被忽略的关键是：**每个 head 不只投影不同，它还独立做一次 softmax。** softmax 是非线性操作，所以多个小 head 一般不能简单等价成一个同维度的大 head。

举一个概念性例子。假设同一个 token 同时需要：

- 强烈关注前一个动词；
- 强烈关注很远的指代词。

一个 softmax 必须在所有位置之间分配一份总和为 1 的概率质量，两种关系会相互竞争；两个 head 则各自有一份独立的概率分布，可以分别形成两个尖峰。最后输出投影 $W_O$ 再把多个视角融合。

**2. 从 `[B, L, d_model]` 到 `[B, h, L, d_h]` 的每一步**

设：

$$
d_h=\frac{d_{model}}{h}.
$$

输入和投影后张量通常为：

$$
X,Q,K,V\in\mathbb{R}^{B\times L\times d_{model}}.
$$

以 $Q$ 为例，先 reshape：

$$
[B,L,d_{model}]
\rightarrow[B,L,h,d_h].
$$

这里不是把一条序列分成 $h$ 份，而是把每个 token 的特征维拆成 $h$ 组。再交换 head 轴与序列轴：

$$
[B,L,h,d_h]
\rightarrow[B,h,L,d_h].
$$

这样矩阵乘法就能把 $B$ 和 $h$ 都当作批量维：

$$
[B,h,L_q,d_h]
\times[B,h,d_h,L_{kv}]
\rightarrow[B,h,L_q,L_{kv}].
$$

得到输出后反向操作：

$$
[B,h,L,d_h]
\xrightarrow{\text{transpose}}
[B,L,h,d_h]
\xrightarrow{\text{reshape}}
[B,L,d_{model}].
$$


![粘贴图片](/my-blog/resources/uploads/pasted-1786559990788.png)


**为什么代码里经常出现 `.contiguous()`？** `transpose` 通常只改变张量的 stride，不会真的按新顺序重新排列内存。此时张量在逻辑上是 `[B,L,h,d_h]`，但底层内存未必连续。`view` 要求内存布局与目标形状兼容，因此常写：

```python
x = x.transpose(1, 2).contiguous().view(B, L, d_model)
```

现代 PyTorch 中 `reshape` 在必要时会自动复制，通常更安全；但理解 `contiguous()` 有助于排查维度正确、数值却错乱或运行时报 stride 错误的问题。


![粘贴图片](/my-blog/resources/uploads/pasted-1786560087827.png)


**3. 头数增加是否会让参数量按比例增加？**

在标准实现中，通常不会。假设 Q、K、V 和输出投影都是 $d_{model}\to d_{model}$：

$$
W_Q,W_K,W_V,W_O\in\mathbb{R}^{d_{model}\times d_{model}}.
$$

忽略 bias，总参数量为：

$$
3d_{model}^2+d_{model}^2=4d_{model}^2.
$$

改变 head 数 $h$ 时，只要 $d_{model}$ 不变，矩阵总大小不变；变化的是：

$$
d_h=\frac{d_{model}}{h}.
$$

所以增加 head 数不是免费提升：

- head 更多，子空间数量增加；
- 但每个 head 的维度变小；
- 太小的 $d_h$ 可能限制单头表达能力；
- 更多 head 还可能带来调度、缓存和 kernel 开销。

**4. Multi-Head Attention 为什么在表示上更丰富**

可以从三个角度理解。

第一，**不同投影子空间**。每个 head 的 $W_Q^{(r)},W_K^{(r)},W_V^{(r)}$ 不同，同一个 token 会被映射成不同的特征视图。

第二，**多个独立概率分布**。每个 head 独立 softmax，能同时形成不同的关注模式。

第三，**输出投影做跨头混合**。Concat 之后的 $W_O$ 不是装饰，它允许模型把不同 head 的特征重新线性组合，生成下一层使用的统一表示。

但需要谨慎：不能把某个 head 简单命名为“语法头”或“指代头”，真实模型中 head 的功能可能分布式、冗余且会随层变化。注意力热力图能帮助观察，但不等于严格的因果解释。

**5. Cross-Attention：为什么 $L_q$ 和 $L_{kv}$ 可以不同**

在 Encoder-Decoder Transformer 中：

- Decoder 当前隐藏状态产生 Query；
- Encoder 输出产生 Key 和 Value。

设：

$$
Q\in\mathbb{R}^{B\times h\times L_{dec}\times d_h},
$$

$$
K,V\in\mathbb{R}^{B\times h\times L_{enc}\times d_h}.
$$

注意力矩阵形状为：

$$
A\in\mathbb{R}^{B\times h\times L_{dec}\times L_{enc}}.
$$

第 $i$ 个 Decoder token 会对所有 Encoder token 分配权重。输出形状是：

$$
O\in\mathbb{R}^{B\times h\times L_{dec}\times d_h}.
$$

因此输出位置数跟 Query 一致。Cross-Attention 的本质可以概括为：“当前序列用自己的问题，去读取另一段序列的 memory。”

**6. MHA、MQA、GQA 到底改变了什么**

标准 Multi-Head Attention（MHA）中：

$$
h_q=h_k=h_v=h.
$$

每个 Query head 都有自己的 Key head 和 Value head。

Multi-Query Attention（MQA）保留多个 Query head，但所有 Query head 共享一组 Key/Value：

$$
h_q=h,\qquad h_{kv}=1.
$$

Grouped-Query Attention（GQA）介于两者之间：

$$
1<h_{kv}<h_q.
$$

例如 $h_q=32,h_{kv}=8$，每 4 个 Query head 共享一组 Key/Value head。

为什么共享 K/V 对生成推理特别有用？因为自回归推理会保存历史 token 的 K、V，也就是 KV Cache。Query 只用于当前一步，用完即可；历史 K、V 每一步都要重复读取，因此它们是显存和内存带宽的重要负担。减少 KV head 数，可以按比例降低 KV Cache。


![粘贴图片](/my-blog/resources/uploads/pasted-1786560170898.png)


**7. KV Cache 的尺寸公式与实际意义**

对一层 Decoder，若缓存所有历史 token 的 Key 和 Value，元素个数为：

$$
N_{cache}=2\times B\times L\times h_{kv}\times d_h.
$$

前面的 2 来自 K 和 V 两份缓存。若每个元素占 $b$ 字节，共 $N_{layer}$ 层，则总字节数近似为：

$$
\operatorname{Memory}_{KV}
=2BLh_{kv}d_hbN_{layer}.
$$

举例：

- $B=1$；
- $L=32768$；
- $h_q=32$；
- $d_h=128$；
- $N_{layer}=32$；
- BF16，每元素 2 字节。

普通 MHA 令 $h_{kv}=32$：

$$
2\times1\times32768\times32\times128\times2\times32
\approx 17.18\text{ GB}.
$$

若 GQA 使用 $h_{kv}=8$，KV Cache 约下降到四分之一，即约 4.29 GB。这个估算忽略分配器开销、分页机制、量化与并行切分，但足以说明 GQA/MQA 为什么对长上下文推理很重要。

**8. KV Cache 为什么能把单步计算从“重算整段”变成“只算新 token”**

在第 $t$ 步生成新 token 时，过去 $1\ldots t-1$ 个 token 的表示已经固定。对于每一层：

- 历史 token 的 K、V 不会改变；
- 只需要为新 token 计算新的 $q_t,k_t,v_t$；
- 把 $k_t,v_t$ 追加到 Cache；
- 用 $q_t$ 与缓存的 $K_{1:t}$ 做注意力。

没有 Cache 时，每一步都重新对整段前缀计算 K、V 和注意力。只看注意力部分：第 $t$ 步需要约 $O(t^2d)$，生成长度 $T$ 的总和为约 $O(T^3d)$。使用 Cache 后，第 $t$ 步只需当前 Query 与 $t$ 个 Key 比较，约 $O(td)$，总和约 $O(T^2d)$。

KV Cache 主要降低重复计算，并不让长序列注意力变成线性总复杂度；它还带来显存占用和内存带宽压力。因此现代推理系统还会使用分页 KV Cache、KV 量化、GQA/MQA、前缀复用等方法。


![粘贴图片](/my-blog/resources/uploads/pasted-1786560218740.png)


**9. 实现一个同时支持 Self/Cross-Attention 和 GQA 的模块**

```python
import math
from typing import Optional

import torch
import torch.nn as nn


class GeneralAttention(nn.Module):
    """
    教学版通用注意力：
    - x_q 与 x_kv 相同：Self-Attention
    - x_q 与 x_kv 不同：Cross-Attention
    - num_kv_heads == num_query_heads：MHA
    - num_kv_heads == 1：MQA
    - 1 < num_kv_heads < num_query_heads：GQA
    """

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if d_model % num_query_heads != 0:
            raise ValueError("d_model 必须能被 num_query_heads 整除")
        if num_query_heads % num_kv_heads != 0:
            raise ValueError("num_query_heads 必须能被 num_kv_heads 整除")

        self.d_model = d_model
        self.h_q = num_query_heads
        self.h_kv = num_kv_heads
        self.d_h = d_model // num_query_heads
        self.group_size = self.h_q // self.h_kv
        self.dropout = nn.Dropout(dropout)

        # Query 仍有 h_q 个 head，总输出维度为 h_q * d_h = d_model。
        self.q_proj = nn.Linear(d_model, self.h_q * self.d_h, bias=False)

        # Key/Value 只产生 h_kv 个 head。
        self.k_proj = nn.Linear(d_model, self.h_kv * self.d_h, bias=False)
        self.v_proj = nn.Linear(d_model, self.h_kv * self.d_h, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        # [B, L, num_heads * d_h]
        B, L, _ = x.shape
        # -> [B, L, num_heads, d_h]
        x = x.reshape(B, L, num_heads, self.d_h)
        # -> [B, num_heads, L, d_h]
        return x.transpose(1, 2)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: Optional[torch.Tensor] = None,
        *,
        causal: bool = False,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x_kv is None:
            x_kv = x_q

        B, L_q, _ = x_q.shape
        B_kv, L_kv, _ = x_kv.shape
        if B != B_kv:
            raise ValueError("Query 和 KV 的 batch size 必须一致")

        q = self._split_heads(self.q_proj(x_q), self.h_q)
        k = self._split_heads(self.k_proj(x_kv), self.h_kv)
        v = self._split_heads(self.v_proj(x_kv), self.h_kv)

        # 教学上用 repeat_interleave 展开共享的 KV head。
        # 真实高性能 kernel 可以避免物理复制，直接按组读取。
        if self.h_kv != self.h_q:
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_h)

        if causal:
            if L_q != L_kv:
                raise ValueError("该教学 causal mask 只覆盖 L_q == L_kv 的 self-attention")
            visible = torch.tril(
                torch.ones(L_q, L_kv, dtype=torch.bool, device=x_q.device)
            )
            scores = scores.masked_fill(~visible, float("-inf"))

        if key_padding_mask is not None:
            if key_padding_mask.shape != (B, L_kv):
                raise ValueError("key_padding_mask 应为 [B, L_kv]")
            scores = scores.masked_fill(
                ~key_padding_mask[:, None, None, :], float("-inf")
            )

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)  # [B, h_q, L_q, d_h]

        # [B, h_q, L_q, d_h] -> [B, L_q, h_q, d_h]
        context = context.transpose(1, 2)
        # reshape 会在需要时自动创建连续副本。
        context = context.reshape(B, L_q, self.d_model)
        output = self.out_proj(context)
        return output, attn
```

这段代码为了展示 GQA 逻辑，用 `repeat_interleave` 把较少的 KV head 复制到 Query head 数。它在数学上正确，但会真的复制数据，不能体现 GQA 的内存优势。生产环境应使用支持 GQA 的 fused kernel，例如 PyTorch SDPA 的 `enable_gqa=True`，由底层实现处理共享关系。

**10. Attention Dropout 到底丢在哪里**

常见写法是在 softmax 之后、乘 V 之前对注意力权重做 dropout：

$$
O=\operatorname{Dropout}(A)V.
$$

它随机切断部分 Query-Key 连接，防止模型过度依赖少数固定位置。训练时 dropout 会进行尺度补偿，测试时关闭。使用 PyTorch 的 `scaled_dot_product_attention` 时要特别注意：官方文档要求根据 `self.training` 显式传入 `dropout_p`，否则即使模型处于 eval 模式，只要传入非零概率，它仍可能应用 dropout。

**11. 第二轮面试题：从结构追问到推理优化**

| 面试问题 | 面试现场可以这样回答 |
|---|---|
| Multi-Head Attention 为什么比单头更强？ | 每个 head 有独立的 Q/K/V 投影和独立 softmax，因此可以在不同子空间同时形成多种关注分布。它不只是把特征切开计算，最后还有输出投影将各头信息融合。 |
| 多头数增加会让参数量线性增加吗？ | 在固定 $d_{model}$ 的标准实现里通常不会，Q/K/V/O 四个投影总参数约为 $4d_{model}^2$，与 head 数基本无关。head 增多时每个 head 的维度变小，所以也不是越多越好。 |
| 为什么多个小 head 不等价于一个大 head？ | 因为每个 head 都独立执行 softmax，得到多份独立归一化的注意力分布。softmax 是非线性的，不能把这些结果简单合并成一次大维度 softmax。 |
| Cross-Attention 的输出长度由谁决定？ | 由 Query 长度 $L_q$ 决定，因为每个 Query 产生一个输出；Key/Value 长度只决定每个 Query 可以从多少条 memory 中读取信息。 |
| MHA、MQA、GQA 有什么区别？ | MHA 的每个 Query head 都有对应的 K/V head；MQA 让所有 Query head 共享一组 K/V；GQA 让一组 Query head 共享一组 K/V。后两者主要减少 KV Cache 和解码时的内存带宽。 |
| KV Cache 缓存的是什么？ | 缓存每一层历史 token 已经计算好的 Key 和 Value。生成新 token 时只计算新的 Q/K/V，并让新 Query 读取历史缓存，避免反复重算整个前缀。 |
| KV Cache 的代价是什么？ | 它用显存换计算，缓存量随层数、序列长度、batch、KV head 数和 head 维度线性增长。长上下文或大 batch 推理中，KV Cache 往往成为显存与带宽瓶颈。 |
| `.transpose()` 后为什么常接 `.contiguous()`？ | transpose 通常只改变 stride，内存未按新顺序连续排列；`view` 可能无法直接使用。调用 `.contiguous()` 会按逻辑顺序重新排布内存，或者使用能自动处理的 `reshape`。 |

---

## 三、位置编码、RoPE、LayerNorm、残差与 FFN——Transformer Block 的真正难点

**1. 为什么没有位置编码的 Self-Attention 不知道顺序：排列等变性推导**

很多教程只说“Attention 不知道顺序”，但没有解释原因。设输入矩阵为：

$$
X\in\mathbb{R}^{L\times d_{model}},
$$

用一个排列矩阵 $P\in\mathbb{R}^{L\times L}$ 打乱 token 顺序。$PX$ 只是把 $X$ 的行重新排列。没有位置编码时：

$$
Q'=PXW_Q=PQ,
\qquad
K'=PK,
\qquad
V'=PV.
$$

新的注意力分数为：

$$
Q'K'^\top=(PQ)(PK)^\top=PQK^\top P^\top.
$$

对矩阵行做 softmax 时，排列只会同步重排行与列，因此：

$$
\operatorname{softmax}(PQK^\top P^\top)
=P\operatorname{softmax}(QK^\top)P^\top.
$$

最后：

$$
\begin{aligned}
\operatorname{Attention}(PX)
&=P\operatorname{softmax}(QK^\top)P^\top PV\\
&=P\operatorname{Attention}(X).
\end{aligned}
$$

这叫 **排列等变性**：输入顺序怎么打乱，输出只是跟着同样打乱，模型本身无法知道“哪个排列才是正确语序”。因此必须向输入或注意力分数中注入位置信息。

注意，“不感知位置”不等于模型完全相同地处理所有 token。token 内容不同，注意力结果仍不同；真正缺失的是顺序依据。

**2. 原始正弦位置编码：公式中每一项的作用**

原始 Transformer 使用：

$$
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d_{model}}}\right),
$$

$$
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d_{model}}}\right).
$$

可以把频率写成：

$$
\omega_i=10000^{-2i/d_{model}},
$$

于是每一对维度为：

$$
[\sin(pos\omega_i),\cos(pos\omega_i)].
$$

低索引维度的 $\omega_i$ 较大，随位置变化快，适合区分邻近位置；高索引维度变化慢，能够表达更长尺度的位置变化。不同频率叠加后，位置向量既能区分近距离，也能编码较长范围。


![粘贴图片](/my-blog/resources/uploads/pasted-1786560328222.png)


位置编码通常与 token embedding 相加：

$$
H_0=E_{token}+PE.
$$

为什么是相加而不是拼接？相加保持维度 $d_{model}$ 不变，且允许每一维同时包含内容与位置。模型后续可通过线性变换学习如何分离和组合两类信息。拼接当然也能设计，但会改变维度和参数结构，不是原始 Transformer 的选择。

**3. 正弦位置编码为什么有相对位置性质**

对固定频率 $\omega$，有：

$$
\sin((pos+k)\omega)
=\sin(pos\omega)\cos(k\omega)
+\cos(pos\omega)\sin(k\omega),
$$

$$
\cos((pos+k)\omega)
=\cos(pos\omega)\cos(k\omega)
-\sin(pos\omega)\sin(k\omega).
$$

写成矩阵形式：

$$
\begin{bmatrix}
\sin((pos+k)\omega)\\
\cos((pos+k)\omega)
\end{bmatrix}
=
\begin{bmatrix}
\cos(k\omega)&\sin(k\omega)\\
-\sin(k\omega)&\cos(k\omega)
\end{bmatrix}
\begin{bmatrix}
\sin(pos\omega)\\
\cos(pos\omega)
\end{bmatrix}.
$$

也就是说，位置平移 $k$ 可以由一个只依赖于 $k$ 的线性旋转矩阵表示。这是原论文强调其可能帮助模型学习相对位置关系的原因之一。

但需要区分：正弦位置编码是把绝对位置向量加到输入中，它并没有像 RoPE 或相对位置偏置那样，在注意力点积中显式只保留相对距离。

**4. Learned Absolute Position、Relative Bias 与 RoPE 的区别**

| 方法 | 位置信息加在哪里 | 主要特点 |
|---|---|---|
| 正弦绝对位置编码 | 加到输入 embedding | 无额外参数，频率固定 |
| Learned Absolute Position | 学习一张位置 embedding 表并加到输入 | 灵活，但训练长度之外没有直接对应的已学习位置 |
| Relative Position Bias | 直接向 attention logits 加与相对距离有关的 bias | 显式影响不同相对距离的注意力分数 |
| RoPE | 对 Q、K 做随位置变化的旋转 | 点积中自然出现相对位置差，现代 LLM 常见 |

**5. RoPE 的二维旋转推导：为什么点积依赖相对位置**

先只看 Query/Key 的两个维度。定义位置 $m$ 的旋转矩阵：

$$
R_m=
\begin{bmatrix}
\cos(m\theta)&-\sin(m\theta)\\
\sin(m\theta)&\cos(m\theta)
\end{bmatrix}.
$$

对原始 Query 和 Key 做旋转：

$$
\tilde q_m=R_mq,
\qquad
\tilde k_n=R_nk.
$$

旋转后点积为：

$$
\tilde q_m^\top\tilde k_n
=q^\top R_m^\top R_nk.
$$

旋转矩阵满足：

$$
R_m^\top=R_{-m},
\qquad
R_{-m}R_n=R_{n-m}.
$$

因此：

$$
\tilde q_m^\top\tilde k_n=q^\top R_{n-m}k.
$$

右侧只通过 $n-m$ 依赖两个 token 的相对距离。这就是 RoPE 的核心：

- 每个位置仍有自己的绝对旋转角；
- 但 Q-K 点积里组合成相对位置差。

实际中把 $d_h$ 个维度两两分组，每一对使用不同频率 $\theta_i$：

$$
\theta_i=10000^{-2i/d_h}.
$$

RoPE 只应用于 Q 和 K，不直接应用于 V，因为位置主要需要影响“谁和谁匹配”，Value 负责传递内容。


![粘贴图片](/my-blog/resources/uploads/pasted-1786561549953.png)


**6. LayerNorm 的公式：它到底对哪个维度归一化**

对一个 token 的隐藏向量：

$$
x=[x_1,x_2,\ldots,x_{d_{model}}],
$$

LayerNorm 计算该 token 在特征维上的均值和方差：

$$
\mu=\frac{1}{d_{model}}\sum_{i=1}^{d_{model}}x_i,
$$

$$
\sigma^2=\frac{1}{d_{model}}\sum_{i=1}^{d_{model}}(x_i-\mu)^2.
$$

归一化并加入可学习缩放和平移：

$$
\operatorname{LN}(x)_i
=\gamma_i\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta_i.
$$

对形状 `[B,L,d_model]` 的输入，LayerNorm 通常独立处理每一个 `[d_model]` 向量，不跨 batch，也不跨 token 统计。这就是它不依赖 batch size、适合变长序列的直接原因。

BatchNorm 则通常在 batch 和空间/序列维上聚合统计量，训练和推理还要维护 running statistics。Transformer 中 token 长度、padding 模式和 batch size 变化大，LayerNorm 更自然。

**7. RMSNorm 为什么省掉均值仍然有效**

RMSNorm 不做去均值，只按均方根缩放：

$$
\operatorname{RMS}(x)=
\sqrt{\frac{1}{d_{model}}\sum_i x_i^2+\epsilon},
$$

$$
\operatorname{RMSNorm}(x)_i
=\gamma_i\frac{x_i}{\operatorname{RMS}(x)}.
$$

LayerNorm 同时提供平移不变性和缩放不变性；RMSNorm 主要保留缩放不变性，计算更简单。现代 Decoder-only LLM 中经常使用 RMSNorm，但不能简单理解成“LayerNorm 的绝对升级版”，它们的中心化性质不同，最终选择与架构和训练配方有关。

**8. Residual Connection 不只是“防止梯度消失”**

残差结构为：

$$
y=x+F(x).
$$

它让子层学习的是相对输入的增量 $F(x)$。反向传播时：

$$
\frac{\partial y}{\partial x}
=I+\frac{\partial F}{\partial x}.
$$

即使 $\partial F/\partial x$ 很小，梯度仍有恒等映射 $I$ 可以传播。更深层地看，残差流为整个 Transformer 提供一条贯穿各层的信息高速通道；Attention 和 FFN 不断向这条 residual stream 写入修正。

这也解释为什么残差分支的尺度很重要：若每层写入的增量方差过大，经过很多层累积后激活可能不断膨胀；归一化位置、初始化、残差缩放都会影响深层训练稳定性。

**9. Post-LN 与 Pre-LN：不仅是顺序不同**

原始 Transformer 常写成 Post-LN：

$$
x_{l+1}=\operatorname{LN}(x_l+F_l(x_l)).
$$

Pre-LN 写成：

$$
x_{l+1}=x_l+F_l(\operatorname{LN}(x_l)).
$$

二者差异：

- **Post-LN**：残差相加后立刻归一化，层输出尺度受控，但跨层梯度需要经过多个 LayerNorm 和子层路径；深层训练往往更依赖 warm-up 和初始化。
- **Pre-LN**：主残差路径是直接的加法链，梯度可以沿恒等路径传播，通常更容易稳定训练深层模型。

用展开形式看 Pre-LN：

$$
x_L=x_0+\sum_{l=0}^{L-1}F_l(\operatorname{LN}(x_l)).
$$

最终表示保留一条从 $x_0$ 到 $x_L$ 的显式加法通路。Post-LN 每层都把相加结果重新归一化，梯度和表示动力学不同。

但不能把结论简化为“Pre-LN 永远更好”。Post-LN 在充分调参时也可能有较强最终表现；现代架构还存在 Sandwich/Peri-LN、DeepNorm 等设计。Day 4 要掌握的核心是：**归一化放置会直接改变残差流和梯度传播，而不是无关紧要的代码顺序。**


![粘贴图片](/my-blog/resources/uploads/pasted-1786561627323.png)


**10. FFN 为什么占据大量参数：精确拆解**

标准 Position-wise FFN：

$$
\operatorname{FFN}(x)=W_2\phi(W_1x+b_1)+b_2.
$$

其中：

$$
W_1\in\mathbb{R}^{d_{model}\times d_{ff}},
\qquad
W_2\in\mathbb{R}^{d_{ff}\times d_{model}}.
$$

忽略 bias，参数量约为：

$$
2d_{model}d_{ff}.
$$

原始 Transformer 常取 $d_{ff}=4d_{model}$，于是 FFN 参数约为：

$$
8d_{model}^2.
$$

而普通 MHA 的 Q/K/V/O 投影约为 $4d_{model}^2$。所以单个 Block 中，FFN 参数通常比 Attention 投影更多。Attention 更受关注，是因为它负责 token 间交互；FFN 则承担每个 token 内部的大量非线性特征变换和容量。

为什么先升维再降维？$W_1$ 把 token 映射到更宽的特征空间，激活函数对这些中间特征做门控或非线性选择，$W_2$ 再投影回 residual stream。若中间维不扩张，非线性表示容量通常受限。

**11. ReLU、GELU 与 SwiGLU：FFN 不是固定模板**

原始 FFN 常用 ReLU：

$$
\operatorname{ReLU}(x)=\max(0,x).
$$

BERT 等模型常用 GELU：

$$
\operatorname{GELU}(x)=x\Phi(x),
$$

其中 $\Phi(x)$ 是标准正态分布 CDF。它不是硬切掉负数，而是平滑地按输入大小调节通过比例。

SwiGLU 是门控 FFN 的一种常见形式：

$$
\operatorname{SwiGLU}(x)
=\operatorname{SiLU}(xW_g)\odot(xW_u),
$$

再通过 down projection：

$$
\operatorname{FFN}(x)
=[\operatorname{SiLU}(xW_g)\odot(xW_u)]W_d.
$$

它有两条上投影分支：一条产生 gate，一条产生 value；逐元素相乘后再降维。门控允许网络动态决定哪些中间特征通过。

因为 SwiGLU 多了一个上投影矩阵，为了让参数量与普通 FFN 接近，中间维度通常不会仍取 $4d_{model}$，而会相应缩小。比较 FFN 结构时，不能只看 $d_{ff}$，还要计算总矩阵大小。

**12. Attention 与 FFN 的职责分工**

可以用一句更精确的话概括：

- Attention 是 **token mixing**：让不同位置交换信息；
- FFN 是 **channel mixing**：在单个 token 的特征维上做非线性变换。

Attention 的输出第 $i$ 个位置依赖其他 token；FFN 对每个位置使用相同参数、独立计算：

$$
y_i=\operatorname{FFN}(x_i).
$$

“独立计算”不等于没有上下文，因为进入 FFN 的 $x_i$ 已经通过 Attention 融合了上下文。

**13. 一个现代化的 Pre-Norm Decoder Block**

常见 Decoder-only Block 可以写为：

$$
x'=x+\operatorname{Attention}(\operatorname{Norm}_1(x)),
$$

$$
y=x'+\operatorname{FFN}(\operatorname{Norm}_2(x')).
$$

如果使用 RoPE，它通常作用于 Attention 内部投影出的 Q、K；Causal Mask 防止未来信息泄露；Norm 可能是 RMSNorm；FFN 可能是 SwiGLU。

下面实现 RMSNorm 和 SwiGLU，并复用第二轮的 `GeneralAttention`：

```python
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 建议在 float32 中计算均方，尤其输入为 fp16/bf16 时更稳定。
        rms_inv = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        normalized = x.float() * rms_inv
        # 转回输入 dtype，避免无意间让后续计算全部变成 float32。
        return (normalized * self.weight.float()).to(dtype=x.dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU(z) = z * sigmoid(z)
        gate = torch.nn.functional.silu(self.gate_proj(x))
        value = self.up_proj(x)
        return self.down_proj(gate * value)


class PreNormDecoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = GeneralAttention(
            d_model=d_model,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            dropout=dropout,
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-Norm：先归一化，再进入子层。
        attn_input = self.norm1(x)
        attn_output, attn_weights = self.attn(
            attn_input,
            causal=True,
            key_padding_mask=key_padding_mask,
        )
        x = x + self.resid_dropout(attn_output)

        ffn_output = self.ffn(self.norm2(x))
        x = x + self.resid_dropout(ffn_output)
        return x, attn_weights
```

注意：这段代码还没有真正应用 RoPE，也没有 KV Cache 接口，因为把所有高级机制混进同一段代码会掩盖主线。RoPE 应在 `q_proj/k_proj` 分 head 后、计算点积前应用；KV Cache 应允许传入过去的 K/V 并在序列轴追加。第四轮会用伪代码说明推理路径。

**14. Encoder、Decoder 与 Decoder-only 的结构差异**

原始 Encoder Layer：

1. 双向 Self-Attention，不使用 causal mask；
2. FFN；
3. 每个子层配残差和 Norm。

原始 Decoder Layer：

1. Causal Self-Attention；
2. Cross-Attention，读取 Encoder memory；
3. FFN；
4. 每个子层配残差和 Norm。

BERT 是 Encoder-only，所有非 PAD token 通常可以双向相互关注；GPT 类模型是 Decoder-only，但一般移除了 Encoder-Decoder 架构中的 Cross-Attention，只保留 Causal Self-Attention 和 FFN。称其为“Decoder-only”是结构谱系上的说法，不代表每层都仍包含原始 Decoder 的 Cross-Attention。

**15. 第三轮面试题：把公式、架构与训练稳定性串起来**

| 面试问题                                       | 面试现场可以这样回答                                                                                                                           |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| 为什么 Transformer 必须加入位置信息？                  | 不加位置时，Self-Attention 对 token 排列是等变的：输入行怎么重排，输出只会同步重排，模型无法区分“狗咬人”和“人咬狗”的顺序差异。因此需要绝对或相对位置机制打破这种对称性。                                    |
| 正弦位置编码有什么特点？                               | 它没有可学习参数，不同维度使用不同频率的正弦和余弦。利用三角恒等式，位置平移可以表示为只依赖相对位移的线性旋转，这给模型学习相对距离提供了结构。                                                             |
| RoPE 为什么能编码相对位置？                           | RoPE 按位置对 Q 和 K 做旋转。位置 $m$ 的 Q 与位置 $n$ 的 K 点积时，$R_m^TR_n$ 会化成 $R_{n-m}$，所以匹配分数自然依赖相对距离 $n-m$。                                        |
| LayerNorm 对哪些维度归一化？                        | 对每个样本的每个 token，沿最后的隐藏维 $d_{model}$ 独立计算均值和方差，不跨 batch，也不跨序列位置。这使它不依赖 batch 统计量，适合变长序列。                                               |
| RMSNorm 与 LayerNorm 有什么区别？                 | LayerNorm 会减均值再按标准差缩放，RMSNorm 不做中心化，只按均方根缩放，计算更简单。两者都能控制激活尺度，但不变性和训练动力学并不完全相同。                                                       |
| Pre-LN 为什么通常更容易训练深层 Transformer？           | Pre-LN 把 Norm 放在子层内部，主 residual stream 保留直接的恒等加法路径，梯度可以更顺畅地跨层传播。Post-LN 的梯度需要经过层层归一化与子层，通常更依赖 warm-up 和精细初始化。                        |
| FFN 为什么先升维再降维？                             | Attention 完成 token 间信息交换后，FFN 在更宽的中间空间对每个 token 做非线性特征变换。升维提供更多中间特征和门控容量，再投影回 residual stream 的固定维度。                                 |
| Attention 和 FFN 分别负责什么？                    | Attention 主要做 token mixing，让不同位置交换信息；FFN 主要做 channel mixing，在每个 token 的特征维上做非线性变换。进入 FFN 的 token 已经包含上下文，所以逐位置计算不代表缺少上下文。            |
| GPT 为什么叫 Decoder-only 却没有 Cross-Attention？ | 它继承原始 Transformer Decoder 的因果自注意力和自回归生成方式，但移除了读取 Encoder memory 的 Cross-Attention。这里的 decoder-only 是架构谱系名称，不是原始 Decoder Layer 的逐项复制。 |

---

## 四、复杂度、FlashAttention、训练与推理路径、论文精读和面试总复习

**1. Transformer 的复杂度不能只背一个 $O(L^2)$**

设 batch 为 $B$，隐藏维为 $d=d_{model}$，序列长度为 $L$，head 数为 $h$，单头维度为 $d_h=d/h$。

**Q/K/V 投影**

输入 `[B,L,d]` 分别乘三个 `[d,d]` 矩阵：

$$
\operatorname{FLOPs}_{QKV}=O(3BLd^2).
$$

**注意力分数 $QK^T$**

每个 head 计算 `[L,d_h] @ [d_h,L]`：

$$
O(BhL^2d_h)=O(BL^2d).
$$

**注意力权重乘 V**

同样为：

$$
O(BL^2d).
$$

**输出投影**

$$
O(BLd^2).
$$

因此整个 MHA 主要计算量约为：

$$
O(BLd^2+BL^2d).
$$

严格来说有多个常数项，但面试中写成这两个主项即可。

这揭示一个重要事实：

- 当 $L\ll d$ 时，线性投影 $Ld^2$ 可能并不比 $L^2d$ 小；
- 当 $L\gg d$ 时，二次注意力项成为主导。

所以“Attention 总是被 $L^2$ 完全支配”也不精确，要看模型宽度与序列长度的相对大小。

**FFN 计算量**

普通两层 FFN 约为：

$$
O(2BLdd_{ff}).
$$

若 $d_{ff}\approx4d$，约为：

$$
O(8BLd^2).
$$

短序列大模型中，FFN 可能占相当大的计算；长序列时，Attention 的二次项迅速上升。

**2. 内存复杂度为什么更容易先卡住**

朴素 Attention 会显式保存：

$$
S,A\in\mathbb{R}^{B\times h\times L\times L}.
$$

仅注意力权重的元素数就是：

$$
BhL^2.
$$

例如 $B=1,h=32,L=8192$，元素数约为：

$$
32\times8192^2\approx2.15\times10^9.
$$

即使每个元素只占 2 字节，也超过 4 GB；训练时还需保存更多中间量用于反向传播。实际 kernel 会融合操作、重计算或分块，因此不能直接用这个数等同于真实峰值显存，但它说明了为什么不显式物化完整 $L\times L$ 矩阵非常关键。

**3. FlashAttention 改变了什么，又没有改变什么**

FlashAttention 是 **精确 Attention 算法**，不是把注意力近似成稀疏或低秩。它输出与标准 Attention 在允许的数值误差范围内等价。

它没有改变标准 Attention 的渐近 FLOPs：仍然需要处理大量 Query-Key 配对，计算复杂度仍是 $O(L^2d)$。它主要解决的是 **GPU 内存层级之间的 IO 开销**。

GPU 中：

- HBM（显存）容量大，但访问相对慢；
- SRAM/共享内存容量小，但访问快；
- 若朴素实现先把 $S=QK^T$ 写回 HBM，再读出做 softmax，再写回，再读出乘 V，会产生大量数据搬运。

FlashAttention 将 Q、K、V 切成小块，在片上存储中完成一块分数的计算、softmax 统计和对 V 的累积，避免完整 $L\times L$ 注意力矩阵反复写回 HBM。


![粘贴图片](/my-blog/resources/uploads/pasted-1786561791267.png)


所以它的加速来自：

> **减少 HBM 与片上内存之间的读写，而不是减少所有 token 两两配对的数学计算。**

**4. 难点：分块后如何得到和完整 softmax 一样的结果**

普通 softmax 对一行分数 $s=[s_1,\ldots,s_L]$ 需要全局最大值和全局归一化因子：

$$
m=\max_j s_j,
\qquad
\ell=\sum_j\exp(s_j-m).
$$

如果 Key 被切成多个 block，当前只看到其中一部分，如何在不保存全部分数的情况下算出准确结果？答案是维护可合并的在线统计量。

假设已经处理旧 block，保存：

- 旧最大值 $m_{old}$；
- 旧指数和 $\ell_{old}$；
- 旧的未最终归一化输出累积。

新 block 的分数最大值为 $m_{block}$。更新后的全局最大值：

$$
m_{new}=\max(m_{old},m_{block}).
$$

旧 block 的指数原本以 $m_{old}$ 为基准，现在要换成 $m_{new}$：

$$
\exp(s-m_{old})
=\exp(s-m_{new})\exp(m_{new}-m_{old}).
$$

等价地，旧指数和缩放为：

$$
\ell_{old}\exp(m_{old}-m_{new}).
$$

新归一化因子为：

$$
\ell_{new}
=\ell_{old}\exp(m_{old}-m_{new})
+\sum_{j\in block}\exp(s_j-m_{new}).
$$

对 Value 的加权和也用相同缩放规则更新。处理完所有 block 后再除以最终 $\ell$，就得到与一次性完整 softmax 等价的结果。

这个在线 softmax 技巧的意义是：每次只需保存小块分数和每行少量统计量，不需要将完整注意力矩阵写回 HBM。


![粘贴图片](/my-blog/resources/uploads/pasted-1786561858344.png)


**5. 训练阶段与自回归推理阶段为何完全不同**

训练 Decoder-only LM 时，给定整段 token：

$$
[x_1,x_2,\ldots,x_L],
$$

通过 causal mask，可以一次并行计算所有位置对下一个 token 的预测：

- 位置 1 预测 $x_2$；
- 位置 2 预测 $x_3$；
- ……
- 位置 $L-1$ 预测 $x_L$。

虽然逻辑上每个位置只能看左侧，但矩阵运算仍可一次完成。

推理时，未来 token 尚不存在，只能：

1. 处理 prompt，得到第一个新 token；
2. 把新 token 追加到上下文；
3. 再生成下一个；
4. 循环。

因此推理有两个阶段：

- **Prefill**：并行处理完整 prompt，计算密集，更像训练前向；
- **Decode**：每步只有一个或少量新 Query，但要读取很长的 KV Cache，常更受内存带宽限制。

这就是为什么某个优化可能提升 prefill，却不一定同样提升 decode；MQA/GQA 对 decode 尤其重要，因为它们减少每一步读取的历史 K/V。

**6. KV Cache 伪代码：看清每一层发生什么**

```python
# 伪代码：单层 Decoder Attention 的增量推理

def decode_one_step(x_new, past_k, past_v):
    # x_new 只包含本轮新增 token，形状可为 [B, 1, d_model]
    q_new = q_proj(x_new)
    k_new = k_proj(x_new)
    v_new = v_proj(x_new)

    # 给 q_new 和 k_new 加上“当前位置”的 RoPE。
    q_new = apply_rope(q_new, position=past_k.length)
    k_new = apply_rope(k_new, position=past_k.length)

    # 在序列轴上追加新 K/V；真实系统常使用预分配或分页缓存，
    # 避免 Python 级 torch.cat 每一步重新分配整块内存。
    all_k = append_to_cache(past_k, k_new)
    all_v = append_to_cache(past_v, v_new)

    # 只有一个新 Query，它可以读取所有历史 Key/Value。
    output = scaled_dot_product_attention(q_new, all_k, all_v)
    return output, all_k, all_v
```

在单 token decode 中，不需要构造完整的下三角 causal mask，因为当前 Query 位于最末端，本来就只与缓存中的过去和当前 K 交互；但批量 speculative decode、分块 decode 或不同长度 batch 会让 mask 管理重新变复杂。

**7. 数值精度：为什么 fp16/bf16 代码不能完全照抄数学式**

Attention 中容易数值不稳定的位置有：

- 大点积 logits；
- softmax 指数；
- LayerNorm/RMSNorm 的方差或均方统计；
- 长序列累加。

常见策略包括：

- 对 logits 使用 $1/\sqrt{d_h}$ 缩放；
- softmax 减去行最大值；
- 在 fused kernel 内用更高精度累积；
- Norm 的统计量转 float32 计算，再转回低精度；
- Mask 使用合适的极小值，避免半精度下不受控的溢出或全 Mask NaN。

BF16 与 FP16 的有效精度和指数范围不同。不能只说“都是 16 位所以一样”；BF16 指数范围接近 FP32，通常更不容易溢出，但尾数位更少。具体训练选择还取决于硬件、loss scaling 和 kernel 支持。

**8. PyTorch 官方 SDPA：为什么建议工程中优先使用**

教学时手写 $QK^T\to$ softmax $\to AV$ 很有价值；工程中优先考虑：

```python
import torch.nn.functional as F

output = F.scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=attn_mask,
    dropout_p=(dropout_p if self.training else 0.0),
    is_causal=False,
    enable_gqa=False,
)
```

官方 SDPA 会根据设备、dtype、形状和参数选择可用后端，例如 FlashAttention、memory-efficient kernel 或通用实现。它还能避免 Python 层显式生成多个大中间张量。

需要重点记住两个工程坑：

1. **Dropout 要显式根据训练状态传 0**；
2. **不同 API 的 bool mask 语义可能相反**。在 `scaled_dot_product_attention` 中，bool mask 的 `True` 通常表示允许参与；而某些 `MultiheadAttention` padding mask 接口中，`True` 表示需要屏蔽。迁移代码时必须查当前官方文档。

**9. 一个 Transformer Block 的完整前向数据流**

以现代 Pre-Norm Decoder Block 为例：

1. 输入 $x_l\in\mathbb{R}^{B\times L\times d}$；
2. RMSNorm/LayerNorm 得到归一化表示；
3. 线性投影成 Q、K、V；
4. 把特征维拆成 head；
5. 给 Q、K 应用位置机制，如 RoPE；
6. 计算缩放点积；
7. 加 causal/padding/relative bias；
8. softmax 与 attention dropout；
9. 权重乘 V；
10. 多头拼接和输出投影；
11. 与 residual stream 相加；
12. 第二个 Norm；
13. SwiGLU/FFN；
14. 再次 residual 相加；
15. 输出进入下一层。


![粘贴图片](/my-blog/resources/uploads/pasted-1786565592613.png)


**10. 经典论文精读：这次不只告诉你“看第 3 节”**

**必读 1：《Attention Is All You Need》**

建议按下面问题驱动阅读：

- 看 Figure 1：Encoder 与 Decoder 为什么不对称？Decoder 多出的子层是什么？
- 看 3.2.1：作者为什么比较 additive attention 和 dot-product attention？缩放项解决什么数值问题？
- 看 3.2.2：Multi-Head 为什么要把 $d_{model}$ 分到多个 head，而不是每个 head 都保持完整维度？
- 看 3.2.3：Encoder-Decoder Attention 中，Q、K、V 分别来自哪里？
- 看 3.3：FFN 为什么称为 position-wise？它与 $1\times1$ 卷积有何相似性？
- 看 3.5：正弦位置编码为什么使用多个频率？
- 看 Table 1：Self-Attention、RNN、CNN 的复杂度、顺序操作和最大路径长度各说明什么？

读完后必须能回答：

1. Transformer 的核心创新是单独一个注意力公式，还是整套可并行的架构组合？
2. 原论文的 Post-LN、ReLU FFN、正弦位置编码与现代 LLM 有哪些差异？
3. 原论文讨论的 $O(n^2d)$ 为什么至今仍是长上下文的关键瓶颈？

**必读 2：《On Layer Normalization in the Transformer Architecture》**

重点理解：

- Post-LN 在初始化附近为什么更容易出现不均衡梯度；
- warm-up 与归一化位置的关系；
- Pre-LN 改善的是训练动力学，而不只是代码风格。

**拓展 1：《RoFormer》**

重点看 RoPE 的旋转矩阵表达，以及 $R_m^TR_n=R_{n-m}$ 如何让内积出现相对位置。

**拓展 2：《FlashAttention》**

重点看 IO-aware 的问题定义：为什么减少 FLOPs 不一定等于真实运行更快，以及 tiling/online softmax 如何避免物化完整注意力矩阵。

**拓展 3：MQA 与 GQA 论文**

把论文动机和 KV Cache 公式对应起来，理解它们主要优化 Decoder 增量推理，而不是单纯为了减少训练参数。


**面试高频总表：答案控制在真实口述长度**

| 面试问题                                      | 面试现场可以这样回答                                                                                                                                              |
| ----------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 请完整讲一下 Self-Attention。                    | 输入先通过三个线性投影得到 Q、K、V。Q 和 K 做点积并除以 $\sqrt{d_k}$，加上 causal 或 padding mask 后沿 Key 维做 softmax，得到每个 Query 对所有 Key 的权重，再对 V 加权求和。多头版本在多个子空间独立执行这一过程，拼接后经过输出投影。 |
| 为什么 Transformer 比 RNN 更适合并行？              | RNN 的第 $t$ 个状态依赖第 $t-1$ 个状态，序列轴上必须串行。Transformer 训练时可以一次构造整段 Q、K、V 和注意力矩阵，即使有 causal mask，也能并行计算所有位置。                                                   |
| Transformer 的复杂度是多少？                      | MHA 不只是 $O(L^2)$，更完整地说是 $O(BLd^2+BL^2d)$：前一项来自投影，后一项来自 $QK^T$ 和权重乘 V。长序列下二次项成为主要瓶颈，朴素注意力矩阵的内存也是 $O(BhL^2)$。                                             |
| 为什么多头参数量通常不随 head 数增加？                    | 固定 $d_{model}$ 时，Q/K/V/O 投影仍是四个 $d\times d$ 矩阵，总量约 $4d^2$。head 数只是决定如何把 $d$ 拆成 $h\times d_h$，并改变独立 softmax 的数量。                                         |
| 为什么要位置编码？                                 | 无位置编码的 Self-Attention 对输入排列是等变的，无法判断 token 的真实先后顺序。位置机制通过输入向量、attention bias 或 Q/K 旋转，将绝对或相对位置信息注入模型。                                                   |
| RoPE 与绝对位置 embedding 有什么差别？               | 绝对位置 embedding 通常直接加到 token 表示上；RoPE 对 Q、K 的二维维度对做位置相关旋转。旋转后点积出现 $R_m^TR_n=R_{n-m}$，因此匹配分数自然包含相对距离。                                                     |
| Pre-LN 与 Post-LN 的区别？                     | Post-LN 是先做子层和残差相加，再归一化；Pre-LN 是先归一化再进入子层，最后直接加回 residual stream。Pre-LN 有更直接的跨层恒等梯度通路，通常更容易稳定训练深层模型。                                                    |
| LayerNorm 为什么比 BatchNorm 更适合 Transformer？ | LayerNorm 对每个 token 的隐藏维独立统计，不依赖 batch 大小、序列长度或 running statistics。NLP 中 batch 和长度经常变化，因此它更自然，也能保持训练和推理行为一致。                                            |
| FFN 为什么重要？                                | Attention 主要让 token 之间交换信息，FFN 则在每个 token 的通道维上提供大容量非线性变换。FFN 往往占一个 Block 中更多参数，现代模型还常用 SwiGLU 等门控结构提升表达能力。                                             |
| KV Cache 为什么能加速生成？                        | 历史 token 在每层的 K、V 一旦算出就不会变化，缓存后每一步只需要计算新 token 的 Q/K/V，并让新 Query 读取历史缓存。这样避免重复计算整个前缀，但会带来随上下文长度增长的显存和带宽开销。                                              |
| MQA/GQA 为什么能减少推理开销？                       | 它们减少 K/V head 数，让多个 Query head 共享 K/V。KV Cache 的大小与 $h_{kv}$ 成正比，所以共享后缓存和每步读取量明显下降，GQA 通常在质量和速度之间折中。                                                    |
| FlashAttention 为什么快？                      | 它通过 tiling 和在线 softmax，在片上 SRAM 中完成分块计算与累积，避免把完整 $L\times L$ 注意力矩阵反复写入和读出 HBM。它是精确 attention，主要降低 IO，不改变 $O(L^2d)$ 的渐近 FLOPs。                           |
| Attention 权重能否直接作为模型解释？                   | 它能展示某层某头的读取分布，是有价值的诊断信号，但不能直接等同于特征的重要性或因果解释。模型还有残差、FFN、多个层和多个 head，最终决策可能通过多条路径形成。                                                                      |
| Causal Mask 与 Padding Mask 有什么区别？         | Causal Mask 与时间方向有关，禁止当前位置看到未来；Padding Mask 与 batch 对齐有关，禁止读取补齐 token。二者都可转成 logits 上的加性 bias，并可同时使用。                                                   |
| 为什么 softmax 前要减最大值？                       | softmax 对所有 logits 同时减常数不改变结果。减去行最大值后，最大指数变成 1，其余不超过 1，可以避免指数溢出；FlashAttention 的在线 softmax也是在分块场景下维护等价的最大值和归一化因子。                                       |
