---
title: "『LLM学习笔记5』大模型预训练目标与训练流程"
category: "internship"
tags:
  - "LLM"
date: "2026-08-18"
summary: ""
pdf: ""
pdfTitle: ""
---

## 1. 三类 Transformer 架构与三种预训练目标

| 资源                                                                                        | 今天读什么                                                                      | 阅读时盯住什么                                                          |
| ----------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**      | Abstract、Introduction、BERT 部分中的 Input/Output Representation 与 Pre-training | “deep bidirectional”到底如何实现；MLM 的 15%/80%-10%-10%；MLM loss 算在哪些位置 |
| **Language Models are Unsupervised Multitask Learners（GPT-2）**                            | Approach，尤其语言模型概率分解；Training Dataset；Input Representation；Model            | 为什么“预测下一个 token”可以统一大量自然语言任务；为什么数据的广度和质量重要                       |
| **T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer** | text-to-text framework 与 denoising/span corruption 相关部分                    | 为什么输入和输出都写成文本；Encoder-Decoder 如何把“理解”和“生成”连接起来                   |
| Hugging Face：Causal Language Modeling                                                     | 数据整理和训练示例                                                                  | `labels=input_ids` 与模型内部 shift 的关系                               |

**先从一个统一视角看三类架构：它们真正不同的是“条件概率允许依赖什么信息”。**

给定序列：

$$
x=(x_1,x_2,\ldots,x_T)
$$

Transformer 本身只是一套把 token 表示不断变换的网络。**预训练目标才告诉它：最后究竟应该把什么信息压进 hidden state。**

| 路线 | 典型模型 | Attention 可见范围 | 典型目标 | 最自然的能力 |
|---|---|---|---|---|
| Encoder-only | BERT | 每个 token 可看左右两侧 | Masked Language Modeling | 表示、理解、分类、抽取 |
| Decoder-only | GPT、LLaMA 类 | 第 $t$ 个位置只能看 $\le t$ | Causal Language Modeling | 自回归生成、对话、续写、ICL |
| Encoder-Decoder | T5 | Encoder 双向；Decoder 因果；Decoder 还可看 Encoder | Conditional generation / denoising | 翻译、摘要、输入到输出映射 |

**1）Encoder-only：为什么 BERT 能“双向看”？**

假设输入：

```text
我 喜欢 [MASK] 学习
```

如果 `[MASK]` 在位置 3，那么 BERT 预测它时可以同时使用：

```text
左侧：我 喜欢
右侧：学习
```

它的 self-attention 没有 causal triangle，理论上每个非 padding token 都可以与其它非 padding token 交互。简化后的 attention 可见矩阵为：

```text
          Key
        1  2  3  4
Query 1 ✓  ✓  ✓  ✓
      2 ✓  ✓  ✓  ✓
      3 ✓  ✓  ✓  ✓
      4 ✓  ✓  ✓  ✓
```

所以“BERT 是双向的”不是一句玄学描述，它具体意味着：

$$
h_t=f(x_1,\ldots,x_T)
$$

第 $t$ 个位置的 hidden state 可以融合整句左右两侧的信息。

但这里马上出现一个问题：如果训练时把完整句子全部给模型，再要求它预测第 $t$ 个词，模型直接看到答案了。于是 BERT 必须先**破坏输入**，再让模型恢复原 token，这就是 Masked Language Modeling（MLM）。

令被选中的 masked 位置集合为 $\mathcal M$，MLM 的核心目标可以写为：

$$
\mathcal L_{\mathrm{MLM}}
=
-\sum_{t\in\mathcal M}
\log p_\theta(x_t\mid x_{\setminus \mathcal M})
$$

其中：

- $x_t$：位置 $t$ 原来的真实 token；
- $\mathcal M$：被选中作为预测目标的位置；
- $x_{\setminus \mathcal M}$：经过 corruption 后模型能够观察到的上下文；
- $\theta$：模型参数；
- 注意 loss **通常只在被选中的预测位置计算**，不是整句每个位置都算。

原始 BERT 的经典 masking 方案是：先随机选择约 15% token 作为预测目标；对这些被选中的 token，再做：

```text
80% → 替换成 [MASK]
10% → 替换成随机 token
10% → 保持原 token 不变
```

举例，原句：

```text
我 喜欢 机器 学习
```

假设“机器”被选中，那么训练输入可能是：

```text
80% 情况：我 喜欢 [MASK] 学习
10% 情况：我 喜欢 苹果 学习
10% 情况：我 喜欢 机器 学习
```

但三种情况下的监督标签都仍然是：

```text
机器
```

**为什么不是 100% 都换成 `[MASK]`？**

因为真实下游输入通常不会充满 `[MASK]`。如果模型在预训练阶段永远只在 `[MASK]` 位置做预测，它可能过度依赖这个人工符号。随机 token 和保持原 token 的分支让训练输入与自然文本稍微接近一些。

> 一个很容易混淆的点：**MLM 的 `[MASK]` token 与 `attention_mask` 完全不是同一个“mask”。**
>
> - `[MASK]`：词表中的一个特殊 token，用来破坏输入内容；
> - `attention_mask`：张量，用来告诉注意力层哪些位置有效、哪些是 PAD，或哪些位置禁止互相注意；
> - `causal mask`：一种特殊 attention mask，用来禁止看未来。

**BERT 的 NSP（Next Sentence Prediction）是什么？**

原始 BERT 还加入了 NSP：给两段文本 A/B，判断 B 是否真的是 A 的下一句。它属于 BERT 原论文设计的一部分，但要注意：**NSP 不是 Encoder-only 架构的必要条件，也不是 MLM 的组成部分。** 后续很多模型会移除或重新设计句间预训练目标。面试时不要说“BERT 的本质就是 MLM+NSP”，更准确的是：BERT 的核心结构是双向 Transformer Encoder，原始预训练方案包含 MLM 与 NSP。

![Pasted image 20260818132501](/my-blog/resources/uploads/obsidian-1787043651200-1.png)

**2）Decoder-only：为什么“预测下一个 token”足以训练 GPT？**

自然语言序列的联合概率可以用链式法则分解：

$$
p(x_1,x_2,\ldots,x_T)
=
\prod_{t=1}^{T}
p(x_t\mid x_{<t})
$$

其中：

$$
x_{<t}=(x_1,\ldots,x_{t-1})
$$

这意味着，如果模型能学好每一个条件概率：

$$
p(x_t\mid x_{<t})
$$

那么它就在学习整个文本分布。

例如：

```text
我 / 喜欢 / 机器 / 学习
```

模型实际上同时学习：

$$
p(\text{我})
$$

$$
p(\text{喜欢}\mid\text{我})
$$

$$
p(\text{机器}\mid\text{我, 喜欢})
$$

$$
p(\text{学习}\mid\text{我, 喜欢, 机器})
$$

这就是 Causal Language Modeling（CLM）。

对应负对数似然：

$$
\mathcal L_{\mathrm{CLM}}
=
-\sum_{t=1}^{T}
\log p_\theta(x_t\mid x_{<t})
$$

实际代码一般对有效预测位置取平均：

$$
\mathcal L
=
-\frac{1}{N_{\mathrm{valid}}}
\sum_{(b,t)\in \mathcal V}
\log p_\theta
\left(
x_{b,t+1}\mid x_{b,\le t}
\right)
$$

这里 $\mathcal V$ 表示所有没有被 padding 忽略的有效预测位置。

**Decoder-only 最重要的优势不是“它比 Encoder 更高级”，而是目标非常统一。**

任意自然语言都能直接变成训练样本：

```text
网页文章 → 预测下一 token
代码     → 预测下一 token
问答对   → 预测下一 token
数学推导 → 预测下一 token
对话     → 预测下一 token
教程     → 预测下一 token
```

当训练数据中自然出现：

```text
Question: 2+3=?
Answer: 5
```

为了降低 next-token loss，模型必须学会：看到这种上下文后，“5”的条件概率应该很高。数据规模足够大时，很多任务就被嵌入到统一的序列预测目标里。GPT-2 报告强调的核心思想之一正是：语言建模本身有可能从自然文本中吸收多任务结构。

**为什么 Decoder-only 很适合生成？**

推理时：

$$
x_{T+1}\sim p_\theta(x_{T+1}\mid x_{\le T})
$$

得到 $x_{T+1}$ 后再拼回上下文：

$$
x_{T+2}\sim
p_\theta(x_{T+2}\mid x_{\le T+1})
$$

它的训练目标和使用方式天然一致：都是“给前缀，预测后续”。

**3）Encoder-Decoder：为什么 T5 特别适合“输入 → 输出”？**

Encoder-Decoder 把问题显式拆成两部分：

```text
Encoder：理解输入 x
Decoder：基于 x 自回归生成 y
```

概率分解：

$$
p(y\mid x)
=
\prod_{t=1}^{T_y}
p(y_t\mid y_{<t},x)
$$

Decoder 的第 $t$ 层通常包含：

```text
Masked Self-Attention
        ↓
Cross-Attention to Encoder states
        ↓
FFN
```

因此它一边看已经生成的目标前缀 $y_{<t}$，一边从 Encoder 输出中读取完整输入 $x$。

翻译例子：

```text
输入 x：
translate English to German: The house is wonderful.

输出 y：
Das Haus ist wunderbar.
```

摘要例子：

```text
输入 x：
summarize: [一整篇长文章]

输出 y：
[摘要]
```

T5 更进一步，把任务都变成 text-to-text：

```text
分类：
输入：sst2 sentence: this movie is excellent
输出：positive

翻译：
输入：translate English to German: good morning
输出：guten Morgen

摘要：
输入：summarize: ...
输出：...
```

这样不同任务不再需要一堆完全不同的输出头，任务描述、输入、输出都统一成 token sequence。

**T5 的 span corruption 要理解到什么程度？**

不是简单地一个 token 一个 token 地随机 `[MASK]`。典型做法是随机移除连续 span，再用 sentinel token 标记：

原句：

```text
I like studying large language models very much
```

可能构造成：

```text
Encoder 输入：
I like <extra_id_0> large language <extra_id_1> very much

Decoder 目标：
<extra_id_0> studying <extra_id_1> models <extra_id_2>
```

模型必须利用剩余上下文恢复被删除的 span。它既保留了 Encoder 的双向理解能力，又让 Decoder 学习条件生成。

![Pasted image 20260818133645](/my-blog/resources/uploads/obsidian-1787043651200-2.png)

**为什么今天重点最终要落在 Decoder-only？**

对于目标为“大模型算法岗 / 基座模型训练”的学习路线，后面绝大多数训练、SFT、RLHF/RL、KV Cache、推理优化都会直接建立在自回归 Decoder 上。所以 BERT/T5 要理解其设计逻辑，但 **CLM 必须达到可以自己手算 loss、自己写训练代码的程度**。

**三个架构不要这样死记：**

错误记忆：

```text
BERT = 理解
GPT = 生成
T5 = 翻译
```

更好的记忆：

```text
BERT：
完整上下文 → 恢复被遮住的信息
因此 hidden state 天然适合表示学习

GPT：
左侧前缀 → 预测下一个 token
因此可以不断把预测结果追加到前缀中继续生成

T5：
完整编码输入 x + 已生成目标前缀 → 下一个输出 token
因此天然表达条件生成 p(y|x)
```

**本轮自检例子**

现在有三个 attention pattern：

```text
A:
1111
1111
1111
1111

B:
1000
1100
1110
1111

C:
Encoder = A
Decoder = B
Decoder additionally attends to all Encoder states
```

应当立即判断：

```text
A → Encoder-only self-attention
B → Decoder-only causal self-attention
C → Encoder-Decoder
```

如果任务是：

```text
文本情感分类
```

Encoder-only 很自然；如果任务是：

```text
继续写下一段
```

Decoder-only 很自然；如果任务是：

```text
给一段英文，输出德文
```

Encoder-Decoder 很自然。

但“很自然”不等于“只能”。现代 Decoder-only 模型通过 prompt 也可以做分类、翻译和摘要。

**本轮面试题**

| 面试题                                                     | 面试场景回答                                                                                                                                                                                                                                                                                       |
| ------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **BERT、GPT、T5 最核心的区别是什么？**                              | 最核心的不是 Encoder/Decoder 这个名字，而是**信息可见范围和训练目标不同**。BERT 是 Encoder-only，可以双向看上下文，所以采用 MLM，预测被遮挡的 token；GPT 是 Decoder-only，通过 Causal Mask 只能看当前位置及之前的信息，学习 $p(x)=\prod_t p(x_t\mid x_{<t})$；T5 是 Encoder-Decoder，Encoder 双向理解输入，Decoder 自回归生成输出，并通过 Cross-Attention 读取 Encoder，因此学习 $p(y\mid x)$。 |
| **BERT 为什么不能直接让每个 token 看完整句子再预测自身？**                   | 因为会产生 **label leakage**。如果位置 $t$ 已经能看到真实的 $x_t$，再要求它预测 $x_t$，模型很容易学习复制，而不是理解上下文。所以 BERT 会先把部分 token 替换或扰动，再根据左右上下文恢复原 token，本质上是先把答案隐藏起来。                                                                                                                                                    |
| **BERT 的 `[MASK]` 与 `attention_mask` 有什么区别？**           | `[MASK]` 是**词表中的特殊 token**，用于替换原始 token，让模型预测被遮住的内容；`attention_mask` 是控制 Attention 哪些位置有效的张量，例如屏蔽 PAD。简单来说，`[MASK]` 决定“预测什么”，`attention_mask` 决定“Attention 能看哪些位置”。Causal Mask 又是另一种 mask，用来屏蔽未来 token。                                                                                      |
| **原始 BERT 为什么 15% token 中还要 10% random、10% unchanged？** | 原始 BERT 对选中的 15% token 采用约 80% `[MASK]`、10% 随机 token、10% 保持原样。主要是缓解预训练和下游任务的输入分布差异，因为真实输入基本没有 `[MASK]`。同时也防止模型只在看到 `[MASK]` 时才进行上下文推理。无论哪种处理，这些位置的 label 都仍然是原始 token。                                                                                                                       |
| **GPT 的语言模型目标为什么能写成联合概率乘积？**                            | 来自概率的**链式法则**。一个序列的联合概率可以写成 $p(x_1,\dots,x_T)=\prod_{t=1}^{T}p(x_t\mid x_{<t})$。GPT 用 Transformer 去参数化每一个条件概率。因此 next-token prediction 表面上是在做很多词表分类，组合起来实际上是在学习整个文本序列的概率分布。                                                                                                                  |
| **Decoder-only 为什么适合生成？**                               | 因为训练和推理形式高度一致。训练时学习 $p(x_{t+1}\mid x_{\le t})$；推理时给定当前 prefix，同样预测下一个 token，再把它加入 prefix 继续生成。而且任意自然文本都能自动构造成大量 next-token 监督信号，非常适合大规模自监督预训练。                                                                                                                                               |
| **Encoder-Decoder 的 Cross-Attention 在做什么？**             | Cross-Attention 让 Decoder 在生成时读取 Encoder 对输入的表示。通常 Decoder hidden state 生成 Query，而 Encoder 输出提供 Key 和 Value，即 $\mathrm{Attention}(Q_{\text{decoder}},K_{\text{encoder}},V_{\text{encoder}})$。比如翻译时，Decoder 生成当前目标词时，可以查询完整源语言句子的相关信息。                                                        |
| **T5 为什么叫 text-to-text？**                               | 因为它把各种 NLP 任务统一成“文本输入 → 文本输出”。例如分类输出 `"positive"`，翻译输出目标语言文本，摘要输出摘要文本。这样不同任务都可以统一成条件生成问题 $p(y\mid x)$，共享同一个 Encoder-Decoder 架构和 Cross-Entropy 训练目标。                                                                                                                                          |
| **NSP 是 BERT 的必要组成吗？**                                  | 不是。原始 BERT 确实使用 MLM + NSP，但 NSP 只是原论文采用的一种额外预训练目标，并不是 Encoder-only 或 MLM 的必要条件。后续很多 BERT 类模型去掉 NSP 仍然效果很好。所以更准确的说法是：**原始 BERT 使用 MLM 和 NSP，而 BERT 架构的核心仍是双向 Transformer Encoder。**                                                                                                           |
| **“GPT 只能看左边”准确吗？**                                     | 作为直觉基本正确，但严谨来说，第 $t$ 个位置可以看到 **当前位置以及左侧 token**，即 $x_{\le t}$，只是不能看到未来的 $x_{>t}$。例如当前位置输入“机器”，它可以看到“我 喜欢 机器”，但它预测的是下一个 token“学习”，因此不会泄露答案。                                                                                                                                                   |


## 2. Next-Token Prediction：从 token 序列到 Loss

**本轮目标**：彻底掌握 Decoder-only 训练最关键的一条链：`input_ids → hidden states → logits → shift → CrossEntropy → loss`。这一轮是 Day 5 的核心，必须能手推。

先从最小序列开始：

```text
[BOS, 我, 喜欢, 机器, 学习, EOS]
```

如果按数学形式明确写输入/目标：

```text
模型看到： [BOS, 我,   喜欢, 机器, 学习]
应该预测： [我,  喜欢, 机器, 学习, EOS]
```

这叫 **next-token shift**。

注意：真实训练实现常把整个序列一次送入模型：

```text
input_ids = [BOS, 我, 喜欢, 机器, 学习, EOS]
labels    = [BOS, 我, 喜欢, 机器, 学习, EOS]
```

然后在 loss 内部变成：

```text
shift_logits = logits[:, :-1, :]
shift_labels = labels[:, 1:]
```

于是：

```text
position 0 logits → 预测 “我”
position 1 logits → 预测 “喜欢”
position 2 logits → 预测 “机器”
position 3 logits → 预测 “学习”
position 4 logits → 预测 “EOS”
最后一个 logits 没有下一个 token 可监督 → 丢掉
```

这就是为什么你会同时听见两种说法：

```text
“labels 要向左错一位”
```

和：

```text
“在 Hugging Face 中 labels 可以直接等于 input_ids”
```

它们**并不冲突**。前者描述数学监督关系；后者描述特定模型 API 帮你在内部做了 shift。

> **工程高频坑**：如果你已经手工把 `labels` 变成 `[我, 喜欢, 机器, ...]`，又把它传给一个内部还会 shift 的 CausalLM，那么就会发生 double shift，模型变成用当前位置去预测下下个 token。

**为什么训练时能把完整序列一次输入？这不是把答案给模型了吗？**

因为有 causal attention mask。

假设：

```text
x1 = BOS
x2 = 我
x3 = 喜欢
x4 = 机器
x5 = 学习
```

attention 可见关系：

$$
M_{ij}
=
\begin{cases}
0, & j\le i\\
-\infty, & j>i
\end{cases}
$$

attention score 加 mask：

$$
S
=
\frac{QK^\top}{\sqrt{d_k}}+M
$$

softmax 后，未来位置对应 $-\infty$：

$$
\mathrm{softmax}(-\infty)\approx 0
$$

因此第 2 个位置“我”的 hidden state 虽然和“喜欢、机器、学习”一起存在于同一个 tensor 里，却**不能从 attention 路径读取未来 token 的信息**。

这正是：

```text
计算上：整段并行
信息上：严格自回归
```

两者可以同时成立的原因。

![Pasted image 20260818142052](/my-blog/resources/uploads/obsidian-1787043651200-3.png)

**Teacher Forcing 到底是什么？**

生成阶段，第 $t$ 步输入的历史里可能含有模型自己之前生成的 token：

```text
模型生成：
我 → 喜欢 → 机气 → ...
                ↑
          如果这里生成错了
```

后面的预测只能基于这个已经错掉的前缀继续。

训练阶段则不同。训练样本已经有真实文本：

```text
真实序列：
我 → 喜欢 → 机器 → 学习
```

计算“学习”的预测时，模型看到的历史是：

```text
我 喜欢 机器
```

而不是它自己之前可能预测出来的：

```text
我 喜欢 机气
```

也就是说，训练时每个位置的条件上下文来自 **ground-truth sequence**。这就是 teacher forcing 的核心思想。

形式上：

训练：

$$
p_\theta(x_t\mid x_{<t}^{\mathrm{gold}})
$$

自由生成：

$$
p_\theta(\hat x_t\mid \hat x_{<t})
$$

其中 $\hat x$ 是模型自己生成的 token。

这会产生经典的 train-inference mismatch：训练时模型总处在“正确历史”上，推理时可能进入自己训练时很少见的错误前缀。早期序列建模常把它讨论为 exposure bias。对于今天，你先记住：**teacher forcing 不是“把未来 token 直接告诉当前预测位置”，causal mask 仍然禁止未来信息；它是说历史前缀使用真实 token。**

**为什么训练能并行，而推理通常不能？**

训练时完整答案已知：

```text
[BOS, 我, 喜欢, 机器, 学习]
```

于是所有位置的输入 token 都已经存在，可以用一个下三角 mask 一次计算：

```text
BOS      → predict 我
BOS 我   → predict 喜欢
... 
```

推理时下一 token 尚未知：

```text
已有：BOS 我 喜欢
下一 token：???
```

必须先得到：

```text
机器
```

才能构成新的前缀：

```text
BOS 我 喜欢 机器
```

再预测“学习”。

所以依赖关系是：

$$
x_{t+1}\rightarrow x_{t+2}\rightarrow x_{t+3}
$$

这是一条数据依赖链，不能像训练那样把所有未来 token 提前填入。

KV Cache 能减少**重复计算**，但它没有消除“必须先知道上一 token，才能决定下一 token”这一自回归依赖。

![Pasted image 20260818144253](/my-blog/resources/uploads/obsidian-1787043651200-4.png)

**从 hidden state 到 vocabulary logits**

假设：

```text
batch size B = 2
sequence length L = 5
hidden size d_model = 768
vocab size V = 50,000
```

Transformer 最后输出：

$$
H\in\mathbb R^{B\times L\times d_{\mathrm{model}}}
$$

即：

```text
H.shape = [2, 5, 768]
```

经过 LM Head：

$$
Z=HW_{\mathrm{vocab}}^\top+b
$$

其中：

$$
W_{\mathrm{vocab}}\in
\mathbb R^{V\times d_{\mathrm{model}}}
$$

得到：

$$
Z\in\mathbb R^{B\times L\times V}
$$

即：

```text
logits.shape = [2, 5, 50000]
```

第 `(b,t,:)` 个向量就是：

```text
第 b 个样本
第 t 个位置
对整个 50,000 词表中每个 token 的未归一化分数
```

例如某位置 logits 只展示 4 个词：

```text
机器   4.0
苹果   1.0
天气   0.2
睡觉  -0.5
```

softmax：

$$
p(v)
=
\frac{e^{z_v}}{\sum_{j=1}^{V}e^{z_j}}
$$

使“机器”获得最高概率。

**Cross Entropy 到底算了什么？**

如果真实下一 token 是“机器”，token-level CE：

$$
\ell_t
=
-\log p_\theta(\text{机器}\mid\text{prefix})
$$

假设模型概率：

```text
P(机器)=0.70
```

则：

$$
\ell_t=-\ln0.70\approx0.357
$$

如果模型只给：

```text
P(机器)=0.01
```

则：

$$
\ell_t=-\ln0.01\approx4.605
$$

所以 Cross Entropy 的直觉非常简单：

```text
真实 token 概率越高 → loss 越低
真实 token 概率越低 → loss 越高
```

对 batch 中所有有效 token：

$$
\mathcal L
=
\frac{
\sum_{b,t}
m_{b,t}
\left[-\log p_\theta(y_{b,t}\mid x_b)\right]
}{
\sum_{b,t}m_{b,t}
}
$$

其中：

$$
m_{b,t}
=
\begin{cases}
1,& y_{b,t}\neq -100\\
0,& y_{b,t}=-100
\end{cases}
$$

这就是 `ignore_index=-100` 的作用。

**为什么 Padding 不能参与 loss？**

有两条不同长度文本：

```text
样本 A：我 喜欢 机器 学习
样本 B：你好
```

为了组成 tensor，可能 pad 到相同长度：

```text
A: 我   喜欢 机器 学习
B: 你好 PAD  PAD  PAD
```

如果 PAD 位置也算 loss，模型就被迫大量学习：

```text
P(PAD | 你好) → 越来越高
P(PAD | 你好 PAD) → 越来越高
...
```

这不是我们要学习的语言规律，而且短样本会产生大量人工 PAD 监督。

因此通常把 labels 中 padding 位置设为：

```python
-100
```

然后：

```python
CrossEntropyLoss(ignore_index=-100)
```

忽略这些位置。

注意再次区分：

```text
attention_mask 的 PAD=0
→ 避免注意力把 padding 当成有效上下文

labels 中 PAD=-100
→ 避免 padding 参与 loss
```

**这两个操作经常都要做，但负责不同事情。**

![Pasted image 20260818150403](/my-blog/resources/uploads/obsidian-1787043651200-5.png)

**CrossEntropy 的张量为什么通常要 reshape？**

手工计算时：

```text
shift_logits.shape = [B, L-1, V]
shift_labels.shape = [B, L-1]
```

而 `F.cross_entropy` 常用输入形式：

```text
[N, C]
```

其中 `C=V` 是类别数。

所以把前两维合并：

```python
shift_logits.view(-1, V)   # [B*(L-1), V]
shift_labels.view(-1)      # [B*(L-1)]
```

本质上就是把：

```text
batch × position
```

看成很多独立的 token classification examples。

**手工实现一次 Causal LM loss**

```python
import torch
import torch.nn.functional as F

# 假设模型已经产生 logits
# B = 2, L = 5, V = 100
B, L, V = 2, 5, 100
logits = torch.randn(B, L, V, requires_grad=True)

# labels 与 input_ids 形状相同
labels = torch.tensor([
    [10, 20, 30, 40, 50],
    [11, 21, 31, -100, -100],  # padding 位置不参与 loss
])

# 位置 t 的 logits 要预测位置 t+1 的 token。
# 因此最后一个 logits 没有监督目标，丢掉。
shift_logits = logits[:, :-1, :].contiguous()   # [B, L-1, V]

# 第一个 label 没有“前一个位置”去预测它，因此丢掉。
shift_labels = labels[:, 1:].contiguous()        # [B, L-1]

loss = F.cross_entropy(
    shift_logits.view(-1, V),
    shift_labels.view(-1),
    ignore_index=-100,
)

print("shift_logits:", shift_logits.shape)
print("shift_labels:", shift_labels.shape)
print("loss:", loss.item())

# loss 是标量，可以反向传播
loss.backward()

print("logits.grad:", logits.grad.shape)
```

你应当能解释每一行：

```text
logits[:, :-1, :]
为什么去掉最后位置？
→ 因为它没有“下一个真实 token”可以预测。

labels[:, 1:]
为什么去掉第一个标签？
→ 因为没有位置负责预测序列最开头那个 token。

contiguous()
为什么常出现？
→ slicing 后 tensor 的内存布局可能不是连续的；
  某些 view 操作要求连续内存，因此先 contiguous 更稳妥。

view(-1, V)
为什么？
→ 把 B 和 token position 合并成 N 个分类样本。
```

**Perplexity（困惑度）顺手理解**

如果 loss 是平均 token negative log-likelihood：

$$
\mathrm{PPL}
=
\exp(\mathcal L)
$$

例如：

$$
\mathcal L=2
$$

则：

$$
\mathrm{PPL}\approx e^2\approx7.39
$$

直觉上可以粗略理解为：模型在每一步像是在若干个“等概率候选”之间犹豫。但这只是直觉，不能把 PPL 机械解释成真实候选词数量。

更重要的是：**不同 tokenizer、不同数据集上的 PPL 不一定可以直接横向比较。** tokenizer 会改变 token 粒度，loss 是按 token 统计的。

**为什么预训练 loss 下降不等于“所有能力都同比例提升”？**

训练 loss 直接衡量的是：

```text
对训练分布中的 next token 预测得有多好
```

而“数学推理”“事实问答”“长上下文”“代码修复”等是下游行为。它们通常与语言建模能力相关，但映射不是一一对应。模型可能在常见 token 上降低了很多 loss，却对稀有推理模式帮助不大；也可能平均 loss 改善不大，但某些能力在规模或数据到达阈值后表现明显变化。

所以工程中通常同时观察：

```text
training loss
validation loss
perplexity
held-out benchmark
domain-specific eval
generation quality / safety eval
```

**本轮面试题**

| 面试题                                          | 面试场景回答                                                                                                                                                                                                                            |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Causal LM 的 labels 为什么要 shift？**           | 因为位置 $t$ 的 hidden state 应该预测 **下一个 token $x_{t+1}$**。例如输入 `[BOS, 我, 喜欢, 机器]`，对应目标是 `[我, 喜欢, 机器, 学习]`。因此数学上通常计算 `logits[:, :-1, :]` 与 `labels[:, 1:]` 的 Cross-Entropy，实现 $\mathcal L=-\sum_t\log p_\theta(x_{t+1}\mid x_{\le t})$。 |
| **为什么 Hugging Face 中经常 `labels=input_ids`？** | 数学上的 shift 仍然存在，只是很多 Hugging Face `CausalLM` 在 `forward()` 内部自动执行。因此外面可以直接传 `labels=input_ids`，内部再让位置 $t$ 的 logits 对齐位置 $t+1$ 的 label。要特别注意不要自己 shift 一次后，模型内部又 shift 一次，否则会产生 double shift。                                      |
| **为什么训练时能并行预测所有 token？**                     | 因为训练阶段完整 ground-truth 序列已经知道，可以一次输入 Transformer。Causal Mask 保证第 $t$ 个位置只读取 $x_{\le t}$，所以虽然所有位置同时在 GPU 上计算，但未来信息不会流入当前位置。这样一次 forward 就可以得到所有位置的 `[B,L,V]` logits，并行计算多个 next-token loss。                                         |
| **为什么推理仍然需要逐 token？**                        | 因为推理时未来 token 不存在。必须先生成 $x_{t+1}$，才能把它加入上下文，再计算 $x_{t+2}$，因此存在 $x_{t+1}\rightarrow x_{t+2}$ 的自回归数据依赖。KV Cache 可以减少旧 token 的重复计算，但不能消除这种逐 token 生成依赖。                                                                              |
| **Teacher Forcing 是不是未来信息泄漏？**               | 不是。Teacher Forcing 指训练时使用真实的历史 token 作为 prefix，而不是使用模型自己之前生成的 token。模型依然只能看到当前位置以前的信息，未来 token 仍由 Causal Mask 屏蔽。它真正的问题是训练时历史始终正确，而推理时可能包含模型自己的错误，形成 train-inference mismatch。                                                    |
| **logits 为什么是 `[B,L,V]`？**                   | 因为 batch 中每条序列的每个位置都需要对整个词表做一次分类。Transformer 输出 `[B,L,d_{\text{model}}]`，LM Head 将最后一维映射到词表大小 $V$，得到 `[B,L,V]`。其中 `logits[b,t,:]` 就表示第 $b$ 个样本第 $t$ 个位置对全部词表 token 的预测分数。                                                         |
| **为什么 PAD labels 通常设成 `-100`？**              | Padding 只是为了组成规则 tensor，并不是真实文本，所以不能参与语言模型 loss。PyTorch 的 `CrossEntropyLoss` 可以设置 `ignore_index=-100`，因此通常把 PAD 对应的 label 改成 `-100`。这样最终 loss 只在有效 token 上求平均，不会训练模型去预测大量 PAD。                                                    |
| **`attention_mask=0` 与 `labels=-100` 各管什么？** | `attention_mask=0` 控制 **forward 时哪些输入位置参与 Attention**；`labels=-100` 控制 **哪些位置参与 loss**。前者管信息流，后者管监督信号。可以简单记成：`attention_mask` 决定“能看谁”，`-100` 决定“哪里算错”。                                                                            |
| **CrossEntropy 为什么不需要手动先 softmax？**          | 因为 `CrossEntropyLoss` 直接接收 logits，内部等价于稳定实现的 `log_softmax + NLLLoss`。对于正确类别 $y$，损失可以写成 $\mathcal L=-z_y+\log\sum_j e^{z_j}$。所以训练时不要先手动 softmax，否则既多余又可能降低数值稳定性。                                                                   |
| **PPL 如何由 loss 得到？**                         | 如果 loss 是使用自然对数计算的平均 Negative Log-Likelihood，那么 $\mathrm{PPL}=e^{\mathcal L}$。例如平均 loss 为 $\ln 10$，则 PPL 大约为 10。PPL 越低通常表示模型对真实下一个 token 越有把握，但它不直接等价于推理、知识或指令遵循能力。                                                               |
| **不同模型 PPL 能直接比较吗？**                         | 要谨慎。不同 tokenizer 会把同一句话切成不同数量的 token，而 PPL 是 token-level 指标，所以数值单位并不完全相同。此外数据集、context length、BOS/EOS 处理方式也都会影响结果。通常只有在 tokenizer 和评估协议接近时，PPL 才适合直接比较。                                                                           |

## 3. 从网页到 Batch：真实预训练数据流水线

**本轮目标**：不再把“预训练数据”理解成一个 `texts = [...]` 列表，而是知道海量原始网页如何经历提取、过滤、去重、混合、tokenization、chunking/packing，最终变成 GPU 上的 `[B,L]`。

完整流程先记住：

```text
Raw Sources
    ↓
Text Extraction
    ↓
Normalization / Cleaning
    ↓
Language & Quality Filtering
    ↓
Exact / Near Deduplication
    ↓
Safety / Policy / PII Processing
    ↓
Dataset Mixture & Sampling Weights
    ↓
Tokenizer
    ↓
Document Boundary / EOS
    ↓
Chunking or Sequence Packing
    ↓
input_ids / attention_mask / labels
    ↓
Batch
    ↓
GPU
```

这条链里任何一步做差，模型最终都会“学进去”。

**1）Raw data 不等于 training data**

网页抓取结果可能含：

```text
导航栏
cookie banner
广告
HTML 标签
重复页脚
乱码
SEO 垃圾文本
代码
表格
评论
模板
正文
```

第一步通常需要从 HTML / PDF / code repo / book 等源中提取真正文本。

例如网页：

```html
<nav>Home Products Contact</nav>
<article>
  Transformer models learn from token sequences.
</article>
<footer>Copyright 2026...</footer>
```

真正想留下的可能只有：

```text
Transformer models learn from token sequences.
```

如果数十亿网页都反复保留导航和页脚，模型会浪费 token budget 学模板噪声。

**2）Normalization / Cleaning 做什么？**

常见处理思路包括：

```text
Unicode normalization
空白符整理
控制字符清理
异常重复字符检测
过短/过长文档处理
语言识别
明显乱码过滤
```

但“清洗得越狠越好”是错的。过度规则化可能把：

```text
代码缩进
数学符号
表格结构
不同语言字符
特殊标点
```

一起误删。

所以高质量数据工程不是“把文本洗得特别干净”，而是**在噪声去除和信息保真之间取平衡**。

**3）为什么 Deduplication 极其重要？**

假设训练集中某段文本重复 1000 次：

```text
Paris is the capital of France.
```

而某篇稀有科研文本只出现一次。

SGD 看到的训练信号近似按出现频率加权，重复 1000 次等于对前一句施加远高于其它内容的优化权重。

重复数据可能导致：

```text
训练 token 浪费
分布被少数模板扭曲
记忆/复现倾向增强
benchmark contamination 风险上升
泛化收益下降
```

Dedup 一般可分：

```text
Exact dedup：
完全相同的文档/片段直接去重

Near dedup：
文本不完全相同，但高度相似
例如只改标题、日期、少数单词
```

近似去重常借助 hash、n-gram fingerprint、MinHash/LSH 等思想快速找相似文档。今天不需要实现大规模 dedup，但要理解为什么它是**模型训练算法之外，直接改变有效数据分布的重要步骤**。

**4）Quality Filtering 到底在过滤什么？**

可以想象每篇文档有一个潜在质量分数：

$$
q(d)
=
f(
\text{fluency},
\text{informativeness},
\text{structure},
\text{spam signals},
\text{source},
\ldots
)
$$

然后：

```text
高质量文档 → 更大保留概率 / 更高采样权重
低质量文档 → 过滤或降权
```

策略可能基于：

```text
启发式规则
语言模型 perplexity
分类器
来源信誉
重复度
文本结构统计
```

但这里存在真实 trade-off：

```text
过滤太弱 → 垃圾多
过滤太强 → 数据多样性下降，长尾语言/写作风格可能被误伤
```

所以“高质量”不是单维度。

**5）Dataset Mixture：不同数据源为什么不能简单拼起来？**

假设手里有：

```text
Web     8 TB
Code    1 TB
Books   0.5 TB
Math    0.05 TB
```

如果完全按原始 token 数均匀抽，Web 会淹没 Math。

但如果你希望模型增强数学能力，可以人为设置 mixture：

$$
p(\text{source}=k)=\pi_k
$$

其中：

$$
\sum_k\pi_k=1
$$

例如概念性地：

```text
Web   60%
Code  20%
Books 10%
Math  10%
```

即使 Math 的原始数据远少于 Web，也可以通过重复采样或上调权重获得更高训练占比。

这会直接影响模型能力分布，所以数据 mixture 本质上也是一种“训练超参数”。

**6）Tokenizer 后发生了什么？**

原始文本：

```text
Deep learning is fun.
```

tokenizer 得到：

```text
[15496, 4673, 318, 1257, 13]
```

模型只看 token ids，不直接看字符串。

多个文档通常会显式插入 EOS：

```text
Doc A tokens
<EOS>
Doc B tokens
<EOS>
Doc C tokens
<EOS>
```

EOS 至少承担两个作用：

```text
1. 告诉模型一个文本单元结束；
2. 给语言模型学习“何时结束生成”的监督。
```

**跨文档 packing 是否一定允许互相 attention？**

不一定，要区分具体训练实现。

最简单的做法：

```text
Doc A <EOS> Doc B <EOS>
```

整体放进同一个 causal sequence，这时 Doc B 的 token 理论上能看到前面 Doc A。

更严格的 document-aware packing 可以额外构造 block-diagonal attention mask，让不同文档彼此不可见，但这样实现更复杂。

所以面试时不要绝对地说：

```text
“packing 后不同 document 一定互相看不到”
```

或：

```text
“一定互相能看到”
```

正确回答是：**取决于文档边界和 attention mask 的实现。**

**7）为什么需要 Chunking？**

模型最大训练长度假设：

```text
L = 4096
```

一个文档可能有：

```text
20,000 tokens
```

必须切块：

```text
chunk 1: token 0     ~ 4095
chunk 2: token 4096  ~ 8191
...
```

最简单的 fixed-length chunking 会丢失跨 chunk 上下文。

也可以使用 overlap：

```text
chunk 1: 0    ~ 4095
chunk 2: 3584 ~ 7679
```

但 overlap 会重复训练一部分 token，增加计算量。

**8）为什么需要 Packing？**

现在反过来，假设最大长度 $L=8$：

```text
样本 A：5 tokens
样本 B：2 tokens
样本 C：7 tokens
```

如果分别 padding：

```text
A: A A A A A PAD PAD PAD
B: B B PAD PAD PAD PAD PAD PAD
C: C C C C C C C PAD
```

总共：

```text
3 × 8 = 24 token slots
```

有效 token：

```text
5 + 2 + 7 = 14
```

利用率：

$$
\eta
=
\frac{14}{24}
\approx58.3\%
$$

约 41.7% 计算位置浪费在 padding 上。

Packing 可以把短序列尽量塞进一个长度固定的 block：

```text
[A A A A A EOS B B]
[C C C C C C C EOS]
```

这样 GPU 的 token slots 利用率大幅提高。

定义 token utilization：

$$
\eta
=
\frac{N_{\mathrm{real\ tokens}}}
{B\times L}
$$

对于大模型预训练，哪怕利用率提升几个百分点，乘上数十亿、数万亿 token 后都可能对应非常大的算力差异。

![Pasted image 20260818160904](/my-blog/resources/uploads/obsidian-1787043651200-6.png)

**9）Dynamic Padding 是什么？**

如果不做 full packing，至少可以在每个 batch 内只 pad 到当前 batch 最长样本，而不是整个数据集的全局最大长度。

例如全局最大长度：

```text
2048
```

当前 batch 最长只有：

```text
243
```

那么动态 padding 到 243 显然比所有样本 pad 到 2048 节省得多。

Hugging Face 的 data collator 就常负责：

```text
单条样本
    ↓
按当前 batch 最长长度 pad
    ↓
组成 tensor
```

**10）一个 epoch、一步 step、token budget 分别是什么？**

传统监督学习很爱说：

```text
训练 100 epochs
```

大模型预训练更常围绕 **tokens** 和 **steps** 思考。

假设：

```text
global batch = 2,000,000 tokens / step
total training tokens = 1T tokens
```

则训练 step 大约：

$$
\frac{10^{12}}{2\times10^6}
=
5\times10^5
$$

即：

```text
500,000 optimizer steps
```

为什么 epoch 概念变弱？

因为真实预训练常见：

```text
多数据源 mixture
某些源重复采样
某些源只看一次
持续加入新数据
过滤后语料量动态变化
```

很难用一个“所有数据完整遍历几遍”概括。

**11）Batch size 对大模型到底应该怎样理解？**

有三种常见单位：

```text
sequence batch size：
每 step 有多少条 sequence

token batch size：
每 step 总共有多少 token

global batch size：
所有 GPU + gradient accumulation 合起来的 batch
```

假设：

```text
8 GPUs
每 GPU 4 sequences
sequence length = 2048
gradient accumulation = 4
```

每次 optimizer step 的全局 sequence 数：

$$
8\times4\times4=128
$$

全局 token 数：

$$
128\times2048
=
262144
$$

所以：

```text
global token batch = 262,144 tokens / optimizer step
```

**12）数据污染（benchmark contamination）是什么？**

如果训练语料里直接出现某个 benchmark 的测试题和答案：

```text
MMLU test question + answer
```

模型评测时可能不是“泛化解决”，而是部分依靠记忆。

因此数据构建时经常要检查：

```text
与 evaluation set 的 exact overlap
n-gram overlap
near duplicate
题目/答案模板变体
```

但 contamination 检测本身并不完美。

所以看到 benchmark 高分时，要问：

```text
训练数据是什么？
评测题是否可能在训练语料出现？
是否做过去污染？
是否公布过检测方法？
```

这是研究中很重要的实验素养。

**13）Scaling Law：为什么“模型越大”不是唯一变量？**

Kaplan 等人的 Scaling Laws 工作展示了语言模型 loss 与：

```text
模型参数量 N
数据量 D
训练计算量 C
```

之间存在经验上的幂律关系。可以用极度简化的形式建立直觉：

$$
L(N)
\approx
L_\infty
+
aN^{-\alpha}
$$

$$
L(D)
\approx
L_\infty
+
bD^{-\beta}
$$

这不是让你今天背系数，而是理解：

```text
模型规模 ↑
数据规模 ↑
计算量 ↑
```

通常都能系统性降低可约减 loss，但收益逐渐递减。

随后 Chinchilla 的关键结论是：在固定 compute budget 下，很多当时的大模型“参数很多、训练 token 相对不足”。其计算最优分析强调 **model size 与 training tokens 应更平衡地同时扩展**。

经典对比例子：

```text
Gopher：280B parameters
Chinchilla：70B parameters
```

Chinchilla 参数更少，却使用更多训练数据，并在相同训练计算预算附近获得更好的多项下游表现。

Chinchilla 实验中的一个著名量级：

```text
70B parameters
约 1.4T training tokens
```

因此常见记忆点是：

$$
\frac{D}{N}\approx20
\quad
\text{tokens / parameter}
$$

但一定要加一句：**“20 tokens/parameter”是非常有名的经验记忆点，不是跨模型、跨数据、跨训练目标都必须遵守的自然定律。** 后续实践会根据数据质量、重复训练、推理成本、模型架构和目标能力做不同选择。

![Pasted image 20260818163639](/my-blog/resources/uploads/obsidian-1787043651200-7.png)

更完整地说：

$$
\text{Model behavior}
=
f(
\text{architecture},
\text{objective},
\text{data},
\text{optimization},
\text{compute},
\text{post-training}
)
$$

同一个 Transformer，如果训练数据 mixture 完全不同，最后的能力形态可以非常不同。

**本轮面试题**


| 面试题                                     | 面试场景回答                                                                                                                                                                          |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **原始网页为什么不能直接 tokenize 后训练？**           | 因为网页里包含 HTML、导航栏、广告、cookie banner、重复页脚、乱码、SEO 垃圾等内容。如果直接 tokenize，这些垃圾 token 也会占用宝贵的训练 compute，并改变数据分布。因此通常要先做正文提取、语言识别、质量过滤、去重等，再进入 tokenizer。                                 |
| **exact dedup 和 near dedup 有什么区别？**     | Exact dedup 删除完全相同的文本，通常可以直接通过 hash 检测。Near dedup 处理内容高度相似但不完全一致的文本，例如转载文章多了标题或几个段落，常使用 n-gram、MinHash、LSH 等方法。真实互联网数据通常两种去重都会做。                                                |
| **为什么重复数据会影响模型？**                       | 因为样本重复 $k$ 次，相当于它在训练目标里的权重被放大约 $k$ 倍。大量重复文本会降低有效数据多样性，并增加模型记忆特定文本的概率。如果重复内容来自 benchmark，还可能导致 contamination，让评测结果虚高。                                                            |
| **dataset mixture 是什么？**                | 预训练通常会混合网页、代码、书籍、论文等多个数据源。Dataset mixture 就是设置不同数据源的采样概率 $\pi_k$。例如人为提高代码数据的权重，可以增加训练过程中代码 token 的比例。因此 mixture 实际上是在决定模型把多少训练预算分配给不同类型的能力。                                     |
| **为什么文档之间常插 EOS？**                      | EOS 用来显式表示文档结束边界，例如 `Doc A <EOS> Doc B <EOS>`。这样模型能够学习什么时候结束生成，也不会把两个文档简单理解为一句连续文本。不过插 EOS **不等于完全阻止跨文档 Attention**；如果需要完全隔离，还需要 document-aware attention mask。                 |
| **packing 和 padding 有什么区别？**            | Padding 是把短序列补到固定长度，补出来的位置基本都是无效计算；Packing 则把多个较短样本尽量填入同一个固定长度窗口，提高真实 token 的比例。例如 context length 为 2048，packing 可以把多个 300～500 token 的文档组合起来，显著提高 token utilization。            |
| **packing 后不同文档一定互相看不到吗？**              | 不一定。普通 packing 只是把多个文档拼在同一个 token 序列里，如果仍使用普通 Causal Mask，后面的文档仍可能 attend 到前面的文档。只有额外使用 document-aware 或 block-diagonal attention mask，才能真正阻止不同文档之间的信息交互。                       |
| **dynamic padding 有什么好处？**              | Dynamic padding 不把所有样本都 pad 到全局最大长度，而只 pad 到**当前 batch 中最长序列**。例如 batch 最长只有 700 tokens，就没有必要全部 pad 到 2048。这样可以减少 PAD 带来的无效 Attention 和 FFN 计算，提高训练吞吐。                          |
| **大模型训练为什么更常说 token budget 而不是 epoch？** | 因为大模型数据通常来自多个数据源，并存在不同采样权重、重复采样和 packing，“完整遍历一次所有数据”的 epoch 概念并不自然。相比之下，训练了多少 token 更能直接反映模型实际消耗的数据量和计算量，因此常说训练 1T、10T tokens。                                                 |
| **benchmark contamination 为什么危险？**      | 如果 benchmark 的题目或答案出现在预训练数据中，模型可能直接记住这些内容，而不是依靠泛化能力解决问题。最终 benchmark 分数会虚高。所以构造训练集时通常要检测与测试集相似的数据，并尽量移除精确或近似重复样本。                                                               |
| **Scaling Laws 给我们的核心启发是什么？**           | Scaling Laws 说明语言模型 loss 与模型参数量 $N$、训练数据量 $D$、计算量 $C$ 之间存在比较稳定的经验缩放规律。核心意义是：性能提升不是只能靠堆参数，而是需要合理分配参数、数据和 compute，并可以利用小规模实验预测更大规模训练趋势。                                           |
| **Chinchilla 的核心启发是什么？**                | Chinchilla 的关键结论是：在固定 compute 下，很多早期大模型 **参数过多、训练 token 太少**。与其只扩大参数量，不如同时增加训练数据。经典例子是约 70B 参数的 Chinchilla 使用约 1.4T tokens，在相近 compute 下超过更大的模型。重点是参数量和训练 token 要匹配，而不是机械背固定比例。 |

## 4. 把一切接起来：一次完整的语言模型训练 Step

**本轮目标**：从字符串出发，亲手构造 batch，检查 tensor shape，计算 loss，执行 backward 与 optimizer step。学完后你应该能对面试官画出“一次大模型训练 step”的完整计算图。

先记住最小闭环：

```text
text
 ↓ tokenizer
input_ids [B,L]
 ↓ Transformer
hidden_states [B,L,d_model]
 ↓ LM Head
logits [B,L,V]
 ↓ shift + CrossEntropy
loss scalar
 ↓ backward
gradients
 ↓ optimizer.step()
updated parameters
```

![Pasted image 20260818164908](/my-blog/resources/uploads/obsidian-1787043651200-8.png)

**实验 A：先用 Hugging Face 看真实 CausalLM 的输入、logits 和 loss**

安装：

```bash
pip install torch transformers
```

代码：

```python
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. 加载一个较小的 causal language model。
#    这里用 distilgpt2 是为了方便学习流程，不代表现代大模型结构完全等同于它。
model_name = "distilgpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 2. GPT-2 系列默认没有独立 pad token。
#    教学示例中可将 eos_token 暂时同时作为 pad_token，
#    这样 tokenizer 才能把不同长度文本组成一个 batch。
tokenizer.pad_token = tokenizer.eos_token

texts = [
    "I like machine learning.",
    "Transformers predict the next token.",
]

# 3. tokenizer 将字符串变成 input_ids 和 attention_mask。
#    padding=True：只 pad 到当前 batch 的最长序列。
batch = tokenizer(
    texts,
    return_tensors="pt",
    padding=True,
)

input_ids = batch["input_ids"]             # [B, L]
attention_mask = batch["attention_mask"]   # [B, L]

print("input_ids shape:", input_ids.shape)
print("attention_mask shape:", attention_mask.shape)

# 4. labels 初始复制 input_ids。
#    对 padding 位置改成 -100，使其不参与 CrossEntropy。
labels = input_ids.clone()
labels[attention_mask == 0] = -100

# 5. 传入 labels 后，AutoModelForCausalLM 会返回语言建模 loss。
#    许多 Hugging Face CausalLM 会在内部执行 next-token shift，
#    所以这里 labels 不需要提前手工错位。
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
)

loss_hf = outputs.loss
logits = outputs.logits

print("logits shape:", logits.shape)   # [B, L, V]
print("HF loss:", loss_hf.item())
```

**现在不要急着相信 `outputs.loss`，我们自己把它拆开。**

```python
# 6. 手工复现 next-token shift。
#    第 t 个位置的 logits 预测第 t+1 个真实 token。
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = labels[:, 1:].contiguous()

B, L_minus_1, V = shift_logits.shape

loss_manual = F.cross_entropy(
    shift_logits.view(-1, V),
    shift_labels.view(-1),
    ignore_index=-100,
)

print("manual loss:", loss_manual.item())
print("difference:", abs(loss_manual.item() - loss_hf.item()))
```

如果模型实现和设置符合预期，两者应非常接近。

这一步非常重要，因为你从此应该知道：

```text
model(..., labels=labels)
```

不是魔法，而是在内部做了大致类似：

```text
logits
→ shift
→ flatten
→ cross entropy
```

**把一个位置的预测真的打印出来**

```python
# 取第一个样本的某个位置。
b = 0
t = 0

# logits[b, t] 是一个长度为 V 的向量。
token_logits = logits[b, t]

# softmax 变成词表概率。
probs = torch.softmax(token_logits, dim=-1)

# 取概率最大的前 5 个 token。
top_probs, top_ids = torch.topk(probs, k=5)

print("\nPrefix token:")
print(tokenizer.decode([input_ids[b, t].item()]))

print("\nTop-5 next-token candidates:")
for p, idx in zip(top_probs, top_ids):
    print(
        repr(tokenizer.decode([idx.item()])),
        float(p),
    )
```

但要注意：`t=0` 的上下文可能只是一个很短的起始片段，因此预测不一定有很强语义。更有意义的是选句子中间位置，然后把 prefix 解码出来：

```python
t = min(3, input_ids.shape[1] - 2)

prefix = input_ids[b, : t + 1]
print("prefix:", tokenizer.decode(prefix))

probs = torch.softmax(logits[b, t], dim=-1)
top_probs, top_ids = torch.topk(probs, k=5)

for p, idx in zip(top_probs, top_ids):
    print(repr(tokenizer.decode([idx.item()])), float(p))
```

**实验 B：真正执行一次参数更新**

注意：如果直接用预训练模型，做一次 optimizer step 会真的改变它的参数。因此我们这里只是为了观察训练机制。

```python
import torch

# 使用 AdamW，学习率故意设得比较小。
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5,
)

model.train()

# 清除上一次 step 残留的 gradient。
optimizer.zero_grad()

outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
)

loss = outputs.loss

print("loss before backward:", loss.item())

# 反向传播：
# autograd 从 loss 沿计算图求每个可训练参数的梯度。
loss.backward()

# 找一个参数观察 gradient。
name, param = next(
    (name, p)
    for name, p in model.named_parameters()
    if p.requires_grad and p.grad is not None
)

print("observed parameter:", name)
print("parameter shape:", param.shape)
print("gradient norm:", param.grad.norm().item())

# 保存更新前的一小部分参数值，用于对比。
before = param.detach().flatten()[:5].clone()

# AdamW 根据 gradient、动量统计和 weight decay 更新参数。
optimizer.step()

after = param.detach().flatten()[:5].clone()

print("before:", before)
print("after :", after)
print("changed:", not torch.allclose(before, after))
```

现在把每一步翻译成人话：

```text
optimizer.zero_grad()
→ 清掉上一个 batch 的梯度。
  PyTorch 默认梯度会累加，不清除就会把多个 step 混在一起。

forward
→ 用当前参数预测下一 token，得到 logits 和 loss。

loss.backward()
→ 计算“每个参数稍微变化会让 loss 怎么变化”，
  即 ∂L/∂θ。

optimizer.step()
→ 使用梯度真正修改 θ。

进入下一 batch
→ 新参数再次做预测。
```

抽象写成：

$$
g_t
=
\nabla_\theta
\mathcal L_t
$$

最基础的 SGD 是：

$$
\theta_{t+1}
=
\theta_t-\eta g_t
$$

AdamW 会在此基础上加入一阶/二阶矩估计以及 decoupled weight decay，你在前面的优化器学习中已经接触过；Day 5 只需把它放回完整预训练流水线中。

**实验 C：打印每个张量的 shape**

建议你运行时确保看到类似：

```text
input_ids:
[B, L]

attention_mask:
[B, L]

hidden_states:
[B, L, d_model]

logits:
[B, L, V]

shift_logits:
[B, L-1, V]

shift_labels:
[B, L-1]

loss:
[]
```

标量 `loss` 的 shape 是：

```text
torch.Size([])
```

因为整个 batch 最终被 reduction 成一个 scalar。

**如果想拿到 hidden states**

```python
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=labels,
    output_hidden_states=True,
)

last_hidden = outputs.hidden_states[-1]

print("last hidden:", last_hidden.shape)
print("logits:", outputs.logits.shape)
```

于是你可以明确看到：

```text
[B,L,d_model]
     ↓ LM Head
[B,L,V]
```

**实验 D：亲手验证 `-100` 的确不会贡献 loss**

可以构造两个版本：

```python
labels_ignore = input_ids.clone()
labels_ignore[attention_mask == 0] = -100

labels_wrong = input_ids.clone()
# 故意不把 padding 位置设成 -100
```

分别算 loss。

但这里要注意：如果 padding token 正好用的是 EOS，`labels_wrong` 会把人工补出来的大量 EOS 当作真实预测目标，改变训练信号。你应该看到 loss 与梯度都发生变化。

**训练循环扩成多个 step 是什么样？**

概念版：

```python
for batch in dataloader:
    optimizer.zero_grad()

    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"],
    )

    loss = outputs.loss
    loss.backward()

    optimizer.step()
    scheduler.step()
```

大型训练会在这个骨架上加入：

```text
mixed precision / BF16
gradient accumulation
gradient clipping
distributed data parallel
tensor/pipeline/sequence parallel
FSDP / ZeRO
activation checkpointing
FlashAttention
checkpoint saving
evaluation
logging
fault tolerance
```

但请注意：**这些都是让同一个核心训练闭环更省显存、更快、更稳定、更能扩到多 GPU；底层目标依然是 next-token loss。**

**Gradient Accumulation 为什么存在？**

假设 GPU 一次只能放：

```text
micro batch = 2 sequences
```

但你希望优化器看到：

```text
effective batch = 8 sequences
```

可以累计 4 次梯度：

```text
micro batch 1 → backward
micro batch 2 → backward
micro batch 3 → backward
micro batch 4 → backward
                ↓
          optimizer.step()
```

如果有 $G$ 张 GPU，每张卡 micro batch size 为 $B_\mu$，gradient accumulation steps 为 $A$：

$$
B_{\mathrm{global}}
=
G\times B_\mu\times A
$$

若每条序列固定长度为 $L$：

$$
T_{\mathrm{global}}
=
G\times B_\mu\times A\times L
$$

但若使用 variable-length packing，更准确的工程统计应该直接数真实 token，而不是简单用 sequence 数乘 $L$。

**为什么还需要 Learning Rate Scheduler？**

大模型训练通常不会从第一个 step 就突然使用最大学习率。常见策略有：

```text
warmup
→ peak learning rate
→ decay
```

warmup 的直觉：

```text
训练刚开始
参数和 optimizer moments 都不稳定
↓
先用较小 LR
↓
逐渐升到目标 LR
```

后期 decay：

```text
模型逐渐靠近较优区域
↓
降低步长
↓
做更细的优化
```

具体 scheduler 在后面的训练工程部分还会继续展开。

**Checkpoint 里到底存什么？**

至少可能包含：

```text
model parameters
optimizer states
scheduler states
current step
random states
mixed-precision scaler（若使用）
training metadata
```

为什么不只保存 model weights？

如果只是推理：

```text
weights 足够
```

如果想从中断位置**无缝继续训练**：

```text
optimizer 的一阶/二阶矩
LR scheduler 的当前状态
当前 global step
```

都很重要。否则恢复后优化动态可能突然改变。

**把 Day 5 的全部知识压成一条“面试级回答”**

面试官问：

> “一个 Decoder-only LLM 到底怎么预训练？”

你应该能顺畅回答：

```text
首先从大量原始语料中进行正文提取、质量过滤、去重和数据源混合，
然后使用 tokenizer 转成 token ids，并通过 EOS、chunking 或 packing
构造固定上下文长度的训练序列。

训练目标通常是 causal language modeling。
对于长度 L 的序列，模型在 causal mask 下并行产生 [B,L,V] logits，
位置 t 的 logits 用来预测位置 t+1 的真实 token。
实现上会把 logits[:,:-1,:] 与 labels[:,1:] 对齐，
对有效 token 计算 cross entropy，padding 标签通常用 -100 忽略。

得到标量 loss 后执行 backward 得到参数梯度，再由 AdamW 等优化器更新。
大规模预训练再在这一基本闭环上增加 mixed precision、
gradient accumulation、分布式并行、checkpointing 和高效 attention。
训练通常按累计 token 数和 optimizer steps 管理，而不只是 epoch。
```

如果这一段你能脱稿讲清楚，Day 5 的主线就真正打通了。

**今天必须精读的论文与阅读任务**

| 优先级 | 论文 | 今天具体读什么 | 读完必须回答 |
|---|---|---|---|
| **必读 A** | BERT | Abstract；BERT 架构；Input/Output Representations；Pre-training | 为什么双向 attention 需要 MLM？15%/80-10-10 到底怎么走？loss 在哪里算？ |
| **必读 A** | GPT-2 | Approach；语言模型概率分解；Training Dataset；Input Representation；Model | 为什么 language modeling 可以被理解成 unsupervised multitask learning？ |
| **必读 B** | T5 | Text-to-Text 框架；预训练 objective 对比；span corruption | 为什么 Encoder-Decoder 能统一很多 NLP 任务？span corruption 与 BERT MLM 有何不同？ |
| **扩展 B** | Scaling Laws for Neural Language Models | Abstract、核心 scaling 图和 compute/data/model size 关系 | 为什么固定 compute 下不能只讨论参数量？ |
| **扩展 B** | Training Compute-Optimal Large Language Models（Chinchilla） | Abstract、核心结论与 Gopher/Chinchilla 对比 | 什么叫“undertrained large model”？为什么更多数据能让更小模型赢？ |


**Day 5 最终自测：下面 12 个问题至少能口头回答 10 个**

1. 为什么 BERT 可以看右侧 token，而 GPT 不可以？
2. MLM 的 `[MASK]` 和 causal attention mask 是不是一回事？
3. 为什么 GPT 的联合概率可以拆成一串 next-token 条件概率？
4. `input_ids=[A,B,C,D]` 时，位置 B 的 logits 在训练中监督哪个 token？
5. 为什么训练输入整段文本却不构成未来信息泄漏？
6. Teacher Forcing 到底“force”了什么？
7. 为什么训练可以 parallel，而 autoregressive inference 基本上是 sequential？
8. `[B,L,d_model]` 如何变成 `[B,L,V]`？
9. 为什么 `labels=-100` 与 `attention_mask=0` 不能混为一谈？
10. packing 为什么能提高训练效率？它与 document boundary 有什么关系？
11. Chinchilla 为什么说明“参数越大越好”这个说法不完整？
12. 从原始网页到 `optimizer.step()`，能否完整画出全部关键环节？

**本轮面试题**

| 面试题                                          | 面试场景回答                                                                                                                                                                                                                            |
| -------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Causal LM 的 labels 为什么要 shift？**           | 因为位置 $t$ 的 hidden state 应该预测 **下一个 token $x_{t+1}$**。例如输入 `[BOS, 我, 喜欢, 机器]`，对应目标是 `[我, 喜欢, 机器, 学习]`。因此数学上通常计算 `logits[:, :-1, :]` 与 `labels[:, 1:]` 的 Cross-Entropy，实现 $\mathcal L=-\sum_t\log p_\theta(x_{t+1}\mid x_{\le t})$。 |
| **为什么 Hugging Face 中经常 `labels=input_ids`？** | 数学上的 shift 仍然存在，只是很多 Hugging Face `CausalLM` 在 `forward()` 内部自动执行。因此外面可以直接传 `labels=input_ids`，内部再让位置 $t$ 的 logits 对齐位置 $t+1$ 的 label。要特别注意不要自己 shift 一次后，模型内部又 shift 一次，否则会产生 double shift。                                      |
| **为什么训练时能并行预测所有 token？**                     | 因为训练阶段完整 ground-truth 序列已经知道，可以一次输入 Transformer。Causal Mask 保证第 $t$ 个位置只读取 $x_{\le t}$，所以虽然所有位置同时在 GPU 上计算，但未来信息不会流入当前位置。这样一次 forward 就可以得到所有位置的 `[B,L,V]` logits，并行计算多个 next-token loss。                                         |
| **为什么推理仍然需要逐 token？**                        | 因为推理时未来 token 不存在。必须先生成 $x_{t+1}$，才能把它加入上下文，再计算 $x_{t+2}$，因此存在 $x_{t+1}\rightarrow x_{t+2}$ 的自回归数据依赖。KV Cache 可以减少旧 token 的重复计算，但不能消除这种逐 token 生成依赖。                                                                              |
| **Teacher Forcing 是不是未来信息泄漏？**               | 不是。Teacher Forcing 指训练时使用真实的历史 token 作为 prefix，而不是使用模型自己之前生成的 token。模型依然只能看到当前位置以前的信息，未来 token 仍由 Causal Mask 屏蔽。它真正的问题是训练时历史始终正确，而推理时可能包含模型自己的错误，形成 train-inference mismatch。                                                    |
| **logits 为什么是 `[B,L,V]`？**                   | 因为 batch 中每条序列的每个位置都需要对整个词表做一次分类。Transformer 输出 `[B,L,d_{\text{model}}]`，LM Head 将最后一维映射到词表大小 $V$，得到 `[B,L,V]`。其中 `logits[b,t,:]` 就表示第 $b$ 个样本第 $t$ 个位置对全部词表 token 的预测分数。                                                         |
| **为什么 PAD labels 通常设成 `-100`？**              | Padding 只是为了组成规则 tensor，并不是真实文本，所以不能参与语言模型 loss。PyTorch 的 `CrossEntropyLoss` 可以设置 `ignore_index=-100`，因此通常把 PAD 对应的 label 改成 `-100`。这样最终 loss 只在有效 token 上求平均，不会训练模型去预测大量 PAD。                                                    |
| **`attention_mask=0` 与 `labels=-100` 各管什么？** | `attention_mask=0` 控制 **forward 时哪些输入位置参与 Attention**；`labels=-100` 控制 **哪些位置参与 loss**。前者管信息流，后者管监督信号。可以简单记成：`attention_mask` 决定“能看谁”，`-100` 决定“哪里算错”。                                                                            |
| **CrossEntropy 为什么不需要手动先 softmax？**          | 因为 `CrossEntropyLoss` 直接接收 logits，内部等价于稳定实现的 `log_softmax + NLLLoss`。对于正确类别 $y$，损失可以写成 $\mathcal L=-z_y+\log\sum_j e^{z_j}$。所以训练时不要先手动 softmax，否则既多余又可能降低数值稳定性。                                                                   |
| **PPL 如何由 loss 得到？**                         | 如果 loss 是使用自然对数计算的平均 Negative Log-Likelihood，那么 $\mathrm{PPL}=e^{\mathcal L}$。例如平均 loss 为 $\ln 10$，则 PPL 大约为 10。PPL 越低通常表示模型对真实下一个 token 越有把握，但它不直接等价于推理、知识或指令遵循能力。                                                               |
| **不同模型 PPL 能直接比较吗？**                         | 要谨慎。不同 tokenizer 会把同一句话切成不同数量的 token，而 PPL 是 token-level 指标，所以数值单位并不完全相同。此外数据集、context length、BOS/EOS 处理方式也都会影响结果。通常只有在 tokenizer 和评估协议接近时，PPL 才适合直接比较。                                                                           |
