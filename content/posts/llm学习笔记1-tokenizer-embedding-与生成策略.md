---
title: "『LLM学习笔记2』Tokenizer、Embedding 与生成策略"
category: "internship"
tags:
  []
date: "2026-07-31"
summary: ""
pdf: ""
pdfTitle: ""
---

## 一、Tokenizer 与分词算法

**1. 为什么模型不能直接读取文字？**

计算机中的神经网络接收的是数值张量，无法直接把“机器学习”这四个字当作输入。Tokenizer 负责在文字和数字之间搭桥：

1. 将原始文本切分成若干 token；
2. 根据词表把每个 token 映射成唯一的 token ID；
3. 添加模型需要的特殊 token；
4. 返回 `input_ids`、`attention_mask` 等张量。



![粘贴图片](/my-blog/resources/uploads/pasted-1785439324331.png)


例如，下面只是一个示意，不代表所有模型都会这样切分：

```text
原始文本：机器学习真有趣！
tokens：  ["机器", "学习", "真", "有趣", "！"]
token IDs：[91, 204, 17, 806, 3]
```

其中：

| 概念 | 含义 | 容易混淆的地方 |
|---|---|---|
| token | 模型处理文本的基本单元，可以是字、子词、单词、字节或标点 | token 不一定等于“一个汉字”或“一个英文单词” |
| vocabulary / 词表 | token 到整数 ID 的映射表 | 词表是训练 tokenizer 后固定下来的 |
| token ID | token 在词表中的索引 | ID 的数值大小没有语义距离 |
| tokenizer | 完成规范化、切分、映射和特殊 token 处理的完整组件 | tokenizer 不只是一个 `split()` |
| detokenization / decode | 把 token ID 转回可读文本 | 受规范化和空格规则影响，不一定完全还原原始字符串 |

> [!warning] 高频误区  
> token ID 为 100 和 101，不表示两个 token 的语义比 ID 为 100 和 900 更接近。ID 只是词表行号，真正可学习的语义表示来自后面的 Embedding。

**2. 字符级、词级与子词级分词**

| 粒度 | 例子 | 优点 | 缺点 |
|---|---|---|---|
| 字符级 | `playing → p l a y i n g` | 词表小，几乎没有未登录词 | 序列很长，单个 token 语义弱 |
| 词级 | `I love NLP → I / love / NLP` | token 语义完整，序列较短 | 词表巨大，新词和词形变化容易 OOV |
| 子词级 | `unhappiness → un / happi / ness` | 在词表大小和序列长度之间折中 | 切分结果不一定符合语言学直觉 |
| 字节级 | Unicode 文本先编码为字节 | 理论上可表示任何文本，不需要真正的 OOV | 稀有字符、中文或 emoji 可能占多个 token |

现代大模型通常采用**子词或字节级子词**方案。直观上，它会把常见片段保留为整体，把罕见词拆成更小单元。例如：

```text
常见词：learning → learning
罕见词：unlearnable → un + learn + able
更罕见字符串：XQZ-2026 → X + Q + Z + - + 202 + 6
```

同一个字符串在不同模型中可能得到完全不同的 token 数，因为它们的训练语料、词表大小、规范化规则和分词算法不同。因此不能笼统地认为“一个汉字就是一个 token”或“一个英文单词就是一个 token”。

**3. Tokenizer 通常包含哪些阶段？**

一个完整 tokenizer 常见地包含以下模块：

- **Normalization：** 处理 Unicode 形式、大小写、重音符号或空白。不同模型规则不同，有些模型尽量保留原文。
- **Pre-tokenization：** 先根据空格、标点或字节边界进行粗切分。
- **Subword model：** 使用 BPE、WordPiece 或 Unigram 等规则得到最终子词。
- **Post-processing：** 添加 BOS、EOS、CLS、SEP 等特殊 token。
- **ID mapping：** 在词表中查找 token ID，并生成 `attention_mask` 等辅助输入。

`encode` 和“训练 tokenizer”是两件事：

```text
训练 tokenizer：从大量语料中学习词表和切分规则，通常只做一次。
使用 tokenizer：用固定词表和规则切分新文本，每次推理都会发生。
```

**4. BPE：Byte Pair Encoding**

BPE 的核心思想是：**从细粒度单元开始，反复合并训练语料中最常出现的相邻 token 对。**



![粘贴图片](/my-blog/resources/uploads/pasted-1785439378504.png)


一个简化训练过程如下：

```text
初始：
l o w
l o w e r

统计相邻对：
(l, o) 出现 2 次
(o, w) 出现 2 次
(w, e) 出现 1 次
...

选择一个最高频词对，例如 (l, o)，合并为 lo：
lo w
lo w e r

重新统计并继续合并：
(lo, w) → low
```

BPE 训练结束后会保存：

- 基础 token；
- 按顺序排列的合并规则；
- token 到 ID 的词表。

编码新词时，要按照已经学到的合并优先级执行，而不是针对新句子重新统计频率。

BPE 的伪代码可以写成：

```python
vocab = initialize_as_characters(corpus)  # 初始时把词拆成字符或字节

while len(vocab) < target_vocab_size:
    pair_counts = count_adjacent_pairs(corpus)
    best_pair = max(pair_counts, key=pair_counts.get)
    corpus = merge_pair(corpus, best_pair)
    vocab.add("".join(best_pair))
```

这段代码表达的是算法框架，真正实现还要维护词频、合并优先级和高效的数据结构。

**5. Byte-level BPE / BBPE**

普通 BPE 的初始单元可以是字符；Byte-level BPE 则从字节出发。Unicode 文本先被编码成字节，随后再执行 BPE 合并。它的重要优势是：

- 初始字节集合规模有限；
- 任意 Unicode 字符串都可以被表示；
- 几乎不需要真正的 `[UNK]`；
- 对拼写错误、代码、混合语言和特殊符号更稳健。

代价是某些字符需要多个字节。一个罕见汉字或 emoji 可能被切成多个 token，从而增加序列长度。GPT-2 使用了字节级 BPE 思路，许多后续生成式模型也采用了相近方案。

**6. WordPiece**

WordPiece 与 BPE 都是逐步构建子词词表，但选择合并对象的标准不同：

- BPE 主要关注相邻对的出现频率；
- WordPiece 更关注某次合并对整体语言建模或词表质量的提升，不只是看原始共现次数。

为了直观理解，工程教程常用下面的归一化分数作近似说明：

$$
\operatorname{score}(a,b)
=
\frac{\operatorname{freq}(a,b)}
{\operatorname{freq}(a)\operatorname{freq}(b)}
$$

例如，`a` 和 `b` 本身都非常常见，它们即使相邻出现很多次，也可能只是碰巧，因此分数不一定高。反过来，两个 token 平时不太常见，但几乎每次出现都挨在一起，说明它们关系紧密，WordPiece 会更倾向于将它们合并。

可以简单记成：

> **BPE 看“见面次数多不多”，WordPiece 看“关系亲不亲密”。**

如果两个 token 本身都极常见，但只偶尔相邻，它们未必值得合并。WordPiece 常使用特殊前缀表示“词中间的子词”，例如 BERT 风格的：

```text
playing → play + ##ing
```

`##ing` 表示它通常接在另一个子词后面，而不是独立词首 token。具体符号由 tokenizer 实现决定。

**7. SentencePiece**

SentencePiece 更准确地说是一个**可直接从原始文本训练和使用子词模型的工具体系**，常见模型包括 BPE 和 Unigram。它的几个关键特点是：

- 不要求输入提前按空格切成单词；
- 可以直接在原始句子上训练；
- 将空格显式编码为特殊符号，常见显示形式是 `▁`；
- 适合中文、日文等不天然依赖空格分词的语言。

例如：

```text
原文：I love NLP
可能的 token：["▁I", "▁love", "▁N", "LP"]
```

这里的 `▁` 表示该 token 前存在空格。它让解码器能够较稳定地恢复空格信息。

**8. 四种方案的核心对比**

| 方法 | 初始单元 | 如何形成子词 | 典型特点 |
|---|---|---|---|
| BPE | 字符或其他基础单元 | 合并高频相邻对 | 简单、直观、应用广 |
| Byte-level BPE | 字节 | 在字节序列上执行 BPE | 任意字符串可编码，几乎无 OOV |
| WordPiece | 字符或子词 | 选择更能改善词表/似然的合并 | BERT 系模型常见，常见 `##` 标记 |
| SentencePiece | 原始文本，可使用字符基础 | 可训练 BPE 或 Unigram | 不依赖预分词，显式处理空格 |

> [!important] 面试表达  
> “SentencePiece”和“BPE”并不处于完全相同的分类层级。BPE 是具体子词算法；SentencePiece 是从原始文本训练、编码和解码子词的工具体系，它内部可以采用 BPE 或 Unigram。

**9. 特殊 token**

常见特殊 token 包括：

| token | 作用 |
|---|---|
| BOS | 序列开始 |
| EOS | 序列结束；生成时常作为停止条件 |
| PAD | 补齐 batch 中不同长度的序列 |
| UNK | 表示词表无法编码的内容；字节级方案通常很少需要 |
| CLS | BERT 类模型用于整句表示或分类任务 |
| SEP | 分隔两个句子或不同片段 |
| MASK | MLM 预训练中遮蔽待预测 token |

同一个名称在不同模型中可能对应不同字符串和 ID。不要手工假设某个模型的 EOS ID 一定是多少，应读取：

```python
print(tokenizer.eos_token)
print(tokenizer.eos_token_id)
```

**10. `input_ids` 与 `attention_mask`**

把不同长度的句子组成 batch 时，短句通常需要 PAD：

```text
句子 A： [12, 48, 91, 2]
句子 B： [36, 7, 2, PAD]
```

对应的 `attention_mask` 可以是：

```text
句子 A： [1, 1, 1, 1]
句子 B： [1, 1, 1, 0]
```

其中 `1` 表示有效 token，`0` 表示 padding。模型可据此避免把补齐位置当作正常内容处理。注意，`attention_mask` 和 Transformer 内部的 causal mask 不是同一个概念：

- `attention_mask` 常用于屏蔽 PAD 等无效位置；
- causal mask 用于阻止生成式模型看到未来 token。

**11. Tokenizer 代码实验**

先安装依赖：

```bash
python -m pip install -U transformers sentencepiece
```

- `python -m pip`：确保使用当前 Python 解释器对应的 pip；
- `install -U`：没有安装就安装，已经安装则升级；
- `transformers`：加载 Hugging Face tokenizer；
- `sentencepiece`：部分模型的 tokenizer 会依赖它。

下面使用一个较小的 Qwen tokenizer。只加载 tokenizer 时不需要把完整模型权重下载到内存：

```python
from transformers import AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "大语言模型正在改变 NLP, and it is exciting!"

# tokenize() 只返回可读 token，方便观察切分方式。
tokens = tokenizer.tokenize(text)

# encode() 返回 token ID；add_special_tokens=False 便于只观察正文。
token_ids = tokenizer.encode(text, add_special_tokens=False)

print("原始文本：", text)
print("tokens：", tokens)
print("token IDs：", token_ids)
print("token 数量：", len(token_ids))
print("解码结果：", tokenizer.decode(token_ids))
```

进一步查看模型真正接收的字典结构：

```python
encoded = tokenizer(
    text,
    return_tensors="pt",  # 返回 PyTorch 张量
    add_special_tokens=True,
)

print(encoded)
print("input_ids shape:", encoded["input_ids"].shape)
print("attention_mask shape:", encoded["attention_mask"].shape)
```

单条文本的形状通常是：

$$
[batch\_size, sequence\_length] = [1,L]
$$

比较中英文、数字、emoji 和生僻字符串：

```python
samples = [
    "Machine learning is interesting.",
    "机器学习很有趣。",
    "ChatGPT-2026",
    "😂🚀",
    "龘靐齉",
]

for text in samples:
    ids = tokenizer.encode(text, add_special_tokens=False)
    pieces = tokenizer.convert_ids_to_tokens(ids)
    print("-" * 60)
    print("文本：", text)
    print("token：", pieces)
    print("ID：", ids)
    print("数量：", len(ids))
```

观察重点不是背下具体切分，而是回答三个问题：

1. 哪些常见片段被保留成了一个 token？
2. 哪些罕见字符被拆成多个 token？
3. 同样长度的字符串，token 数是否明显不同？

比较 batch padding：

```python
batch = tokenizer(
    ["短句。", "这是一个明显更长的句子，用于观察 padding。"],
    padding=True,
    return_tensors="pt",
)

print("input_ids:")
print(batch["input_ids"])
print("attention_mask:")
print(batch["attention_mask"])
print("shape:", batch["input_ids"].shape)
```

如果 tokenizer 没有预设 `pad_token`，某些模型需要显式设置：

```python
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # 推理实验中常用 EOS 临时充当 PAD
```

这是一种常见工程处理，但训练时是否适合这样做，要根据模型和损失屏蔽方式判断。

**本轮常见面试题**

| 面试问题 | 标准回答 |
|---|---|
| 1. Token、token ID 和词表分别是什么？ | Token 是模型处理文本的基本单元；词表保存 token 到整数索引的映射；token ID 是某个 token 在词表中的索引。ID 只是地址，不具有数值大小意义。 |
| 2. 为什么现代大模型通常使用子词分词？ | 子词在词级和字符级之间折中：词表不会无限膨胀，同时常见词可以保持较短表示，罕见词也能拆成可复用片段，降低 OOV。 |
| 3. BPE 的训练过程是什么？ | 从字符或字节等基础单元开始，统计整个语料中的相邻 token 对，反复合并最高频词对，直到达到目标词表大小或停止条件，并保存合并规则。 |
| 4. BPE 训练和 BPE 编码有什么区别？ | 训练阶段从语料中学习词表与合并规则；编码阶段使用已经固定的规则处理新文本，不会针对每个句子重新统计词频。 |
| 5. Byte-level BPE 有什么优势？ | 从有限字节集合出发，理论上可以编码任意 Unicode 字符串，几乎没有真正的未登录字符，对代码、拼写错误、混合语言和符号更稳健。 |
| 6. WordPiece 与 BPE 的主要区别是什么？ | BPE 主要按相邻对频率合并；WordPiece 更关注合并后对语言模型似然或词表质量的提升，不只看原始频率。 |
| 7. SentencePiece 是一种算法吗？ | 更准确地说，它是从原始文本训练和使用子词模型的工具体系，内部可以采用 BPE 或 Unigram；它不要求预先按空格分词，并常用 `▁` 显式表示空格。 |
| 8. 为什么不同模型对同一句话的 token 数不同？ | 因为训练语料、词表大小、规范化方式、基础单元和合并规则不同。token 数是 tokenizer 特定的，不能按字符数或单词数直接推断。 |
| 9. `attention_mask` 有什么作用？ | 它标记哪些位置是有效 token、哪些位置是 padding 等无效内容，使模型避免关注或计算这些无效位置。它与阻止看到未来信息的 causal mask 不同。 |
| 10. Tokenizer 为什么会影响训练和推理成本？ | 模型计算量、KV Cache 和上下文占用都与 token 序列长度相关。同一文本被切得越碎，序列越长，计算和显存开销通常越大。 |

## 二、Embedding 与 LM Head

**1. 为什么 token ID 不能直接作为连续数值输入？**

假设词表中：

```text
“猫” → 10
“狗” → 11
“量子力学” → 9000
```

不能因此得出“猫”和“狗”的语义距离是 1，“猫”和“量子力学”的距离是 8990。token ID 是离散类别索引，不是带有大小与距离含义的连续特征。

Embedding 层为每个 token 学习一个 $d$ 维向量，把离散索引映射到连续空间：

$$
E \in \mathbb{R}^{V\times d}
$$

其中：

- $V$：词表大小；
- $d$：隐藏维度或 embedding dimension；
- 第 $i$ 行 $E_i$：token ID 为 $i$ 的向量。


![粘贴图片](/my-blog/resources/uploads/pasted-1785439423163.png)


若输入为：

$$
\text{input\_ids}\in\mathbb{N}^{B\times L}
$$

查表后得到：

$$
X=E[\text{input\_ids}]
\in\mathbb{R}^{B\times L\times d}
$$

**2. 查表与 one-hot 乘矩阵**

假设词表大小 $V=5$，token ID 为 2。它的 one-hot 向量是：

$$
x=[0,0,1,0,0]
$$

与 Embedding 矩阵相乘：

$$
xE=E_2
$$

结果就是取出矩阵第 2 行。因此：

```text
Embedding 查表 ≈ one-hot 向量 × Embedding 矩阵
```

数学上等价，但工程实现不会真的构造巨大 one-hot 向量，因为 one-hot 绝大多数位置为 0，直接按索引取行更高效。

**3. Embedding 的参数量**

Embedding 的主要参数量是：

$$
N_{\text{embedding}}=V\times d
$$

例如词表大小 $V=50{,}000$，隐藏维度 $d=768$：

$$
50{,}000\times768=38{,}400{,}000
$$

即 3840 万个参数。仅计算参数存储：

- FP32：$38.4\text{M}\times4$ 字节，约 153.6 MB；
- FP16/BF16：$38.4\text{M}\times2$ 字节，约 76.8 MB。

这说明大词表会直接增大输入 Embedding 和输出层的参数量。多语言模型为了覆盖更多字符和语言，往往需要在词表覆盖率、token 长度和参数量之间权衡。

**4. `nn.Embedding` 的直观实验**

```python
import torch
from torch import nn

torch.manual_seed(42)

vocab_size = 10
embedding_dim = 4

embedding = nn.Embedding(
    num_embeddings=vocab_size,
    embedding_dim=embedding_dim,
    padding_idx=0,  # ID=0 作为 PAD，该行默认不参与正常梯度更新
)

input_ids = torch.tensor([
    [2, 5, 8],
    [4, 1, 0],
])

output = embedding(input_ids)

print("input_ids shape:", input_ids.shape)
print("embedding output shape:", output.shape)
print("第一个样本的向量：")
print(output[0])
```

形状变化为：

$$
[2,3]\rightarrow[2,3,4]
$$

其中：

- 2：batch size；
- 3：每条样本的 token 数；
- 4：每个 token 的向量维度。

验证“输出就是取矩阵对应行”：

```python
token_id = input_ids[0, 1].item()  # 取第一个样本的第二个 token ID
lookup_vector = output[0, 1]
weight_row = embedding.weight[token_id]

print("token ID:", token_id)
print("查表输出:", lookup_vector)
print("矩阵对应行:", weight_row)
print("是否完全一致:", torch.allclose(lookup_vector, weight_row))
```

**5. Embedding 是如何学到语义的？**

Embedding 向量通常随机初始化，然后随模型一起通过反向传播更新。它没有被人工规定“猫应该接近狗”，而是在语言建模目标下逐渐形成有用结构：

```text
输入 token → Embedding → Transformer → logits → 预测下一个 token
                                            ↓
                                      计算交叉熵损失
                                            ↓
                                      反向传播更新 E
```

如果两个 token 在大量相似上下文中承担相似作用，它们的向量可能逐渐形成某种接近关系。但不要把“Embedding 距离近”机械等同于“语义完全相同”，因为向量还会编码语法、频率、任务和训练语料中的多种模式。

**6. 静态 Token Embedding 与上下文表示**

这是非常高频的区别：

- **Token Embedding：** 查表后得到的初始向量，同一个 token ID 在不同句子中初始向量相同；
- **Contextual Hidden State：** 经过 Transformer 多层上下文交互后的向量，会随句子语境变化。

例如“苹果”：

```text
我买了一斤苹果。
苹果发布了新设备。
```

两处“苹果”如果 token ID 相同，最初查到的 Token Embedding 相同；经过 Transformer 后，它们的 hidden state 会因为上下文不同而不同。

**7. 输入 Embedding 还可能包含什么？**

经典 Transformer/BERT 风格常把多种向量相加：

$$
X=
E_{\text{token}}
+
E_{\text{position}}
+
E_{\text{segment}}
$$

- Token Embedding：表示 token 身份；
- Position Embedding：表示序列位置；
- Segment Embedding：区分句子 A、句子 B 等片段，BERT 中常见。

现代生成式模型未必把位置向量直接加到输入上。例如采用 RoPE 的模型会在 Attention 的 Query/Key 上注入位置信息。先记住：**模型必须获得位置信息，但实现不一定都是“加一个 position embedding”。**

**8. Padding Embedding**

`nn.Embedding(..., padding_idx=0)` 的作用是让 PAD 行作为固定填充向量，并避免它像普通 token 一样被更新。注意两层含义：

- Embedding 层可让 PAD 向量保持固定；
- Attention 或损失函数仍需要 mask，避免 PAD 参与上下文计算和损失。

只设置 `padding_idx`，不能自动替代所有 mask。

**9. LM Head：从隐藏向量回到词表**

Transformer 最后一层输出 hidden state：

$$
H\in\mathbb{R}^{B\times L\times d}
$$

LM Head 将每个位置的 $d$ 维向量映射到词表大小 $V$：

$$
Z=HW_{\text{out}}^{\top}+b
$$

其中：

$$
W_{\text{out}}\in\mathbb{R}^{V\times d},
\qquad
Z\in\mathbb{R}^{B\times L\times V}
$$

$Z$ 就是 logits。每个位置都有一个长度为 $V$ 的向量，表示模型对词表中每个 token 的未归一化偏好。


![粘贴图片](/my-blog/resources/uploads/pasted-1785439516824.png)


**10. Weight Tying**

输入 Embedding 和输出 LM Head 都涉及一个 $V\times d$ 的矩阵：

```text
输入端：token ID → d 维向量
输出端：d 维 hidden state → V 个 logits
```

Weight Tying 让两端共享同一组权重，常表示为：

$$
W_{\text{out}}=E
$$

或者在代码中让输出层权重引用输入 Embedding 权重。主要好处：

- 少保存一份 $V\times d$ 参数；
- 输入和输出 token 表示处于同一参数空间；
- 共享可形成一定正则化效果。

它不是所有模型都必须使用的规则。某些模型因为结构、维度或任务不同，会保留独立输出层。

**11. 查看真实模型的 Embedding**

下面只加载模型配置可以先查看关键维度，不需要立即运行推理：

```python
from transformers import AutoConfig

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
config = AutoConfig.from_pretrained(model_name)

print("词表大小:", config.vocab_size)
print("隐藏维度:", config.hidden_size)
print("Embedding 参数量:", config.vocab_size * config.hidden_size)
```

加载完整模型后查看输入 Embedding：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
).to(device)
model.eval()

inputs = tokenizer("机器学习", return_tensors="pt").to(device)

embedding_layer = model.get_input_embeddings()
embeddings = embedding_layer(inputs["input_ids"])

print("input_ids shape:", inputs["input_ids"].shape)
print("embedding weight shape:", embedding_layer.weight.shape)
print("embedding output shape:", embeddings.shape)
```

`embedding_layer.weight.shape` 通常是：

$$
[V,d]
$$

`embeddings.shape` 通常是：

$$
[B,L,d]
$$

检查是否进行了 Weight Tying：

```python
input_weight = model.get_input_embeddings().weight
output_weight = model.get_output_embeddings().weight

print("输入权重 shape:", input_weight.shape)
print("输出权重 shape:", output_weight.shape)
print("是否指向同一块存储:", input_weight.data_ptr() == output_weight.data_ptr())
```

不同模型的输出可能不同，应以实际结果和模型配置为准。

**12. Patch Embedding 与文本 Embedding**

视觉 Transformer 会把图像切成 patch。例如输入图像大小为 $H\times W$，patch 大小为 $P\times P$，patch 数量约为：

$$
N=\frac{H}{P}\times\frac{W}{P}
$$

每个 patch 展平后通过线性层映射成 $d$ 维向量：

$$
x_{\text{patch}}\in\mathbb{R}^{P^2C}
\rightarrow
e_{\text{patch}}\in\mathbb{R}^{d}
$$

它与文本 Embedding 的共同点是：都把离散或局部输入单元转为统一维度的向量序列。区别是：

- 文本 token 通常通过整数 ID 查表；
- 图像 patch 通常通过线性投影或卷积得到向量。

**本轮常见面试题**

| 面试问题                                                  | 标准回答                                                                                             |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| 1. 为什么不能直接把 token ID 当作连续特征？                          | token ID 只是类别索引，数值差没有语义。直接输入会错误引入大小和距离关系，因此需要 Embedding 映射为可学习的连续向量。                             |
| 2. Embedding 的本质是什么？                                  | 一个形状为 $V\times d$ 的可训练查找表。输入 token ID 后取出对应行，得到 $d$ 维向量。                                         |
| 3. Embedding 查表和 one-hot 乘矩阵有什么关系？                    | 数学上等价。one-hot 乘矩阵会选出对应行，但实际实现直接按索引查表，避免构造巨大稀疏 one-hot。                                           |
| 4. 输入形状为 $[B,L]$，经过 Embedding 后是什么？                   | 变成 $[B,L,d]$，每个 token ID 被替换成一个 $d$ 维向量。                                                         |
| 5. Embedding 参数量如何计算？                                 | 主要是词表大小乘隐藏维度，即 $V\times d$；若有额外位置或片段 Embedding，还要分别相加。                                           |
| 6. Token Embedding 与 contextual representation 有什么区别？ | Token Embedding 是查表得到的初始静态向量；contextual representation 是经过 Transformer 后结合上下文得到的动态 hidden state。 |
| 7. `padding_idx` 是否能替代 attention mask？                | 不能。`padding_idx` 主要控制 PAD 对应的 Embedding 行；attention mask 负责屏蔽 PAD 在上下文计算中的影响，训练时还可能需要损失 mask。    |
| 8. LM Head 做什么？                                       | 把每个位置的 $d$ 维 hidden state 映射成长度为 $V$ 的 logits，从而为词表中每个候选 token 打分。                               |
| 9. Weight Tying 是什么？                                  | 输入 Embedding 与输出 LM Head 共享同一组 $V\times d$ 权重，可减少参数，并让输入和输出表示共享参数空间。                             |
| 10. 图像 Patch Embedding 与文本 Embedding 有何异同？            | 两者都把输入单元映射为统一维度的向量序列；文本常通过 token ID 查表，图像 patch 常通过线性层或卷积投影。                                     |

## 三、Logits、Softmax 与生成策略

**1. 大模型如何生成下一个 token？**

对于自回归语言模型，已经有 token 序列：

$$
x_1,x_2,\ldots,x_t
$$

模型预测下一个 token 的条件概率：

$$
P(x_{t+1}\mid x_1,\ldots,x_t)
$$

完整生成过程是：

```text
输入 prompt
→ 模型输出最后一个位置的 logits
→ 把 logits 转为候选概率
→ 按某种解码策略选出下一个 token
→ 把新 token 追加到输入
→ 重复，直到 EOS 或达到长度限制
```

**2. Logits 是什么？**

LM Head 输出 logits：

$$
z=[z_1,z_2,\ldots,z_V]
$$

它们是未归一化分数：

- 可以为负；
- 总和不需要为 1；
- 大小表示模型的相对偏好；
- 不能直接当作概率。

Softmax 把 logits 转为概率：

$$
p_i=
\frac{\exp(z_i)}
{\sum_{j=1}^{V}\exp(z_j)}
$$

得到：

$$
p_i\ge0,\qquad\sum_i p_i=1
$$

数值计算时通常先减去最大 logit，避免指数溢出：

$$
p_i=
\frac{\exp(z_i-\max(z))}
{\sum_j\exp(z_j-\max(z))}
$$

这不会改变 Softmax 结果，因为分子分母同时乘了同一个常数。

**3. 手算示例**

假设三个候选 token 的 logits 为：

```text
A: 2.0
B: 1.0
C: 0.0
```

Softmax 概率约为：

```text
A: 0.665
B: 0.245
C: 0.090
```

A 的概率最高，但 B、C 仍可能在采样时被选中。Greedy Search 则永远选择 A。

代码验证：

```python
import torch

logits = torch.tensor([2.0, 1.0, 0.0])
probs = torch.softmax(logits, dim=-1)

print(probs)
print("概率和:", probs.sum())
```

**4. Greedy Search**

Greedy Search 每一步都选择当前概率最大的 token：

$$
x_{t+1}=\arg\max_i p_i
$$

优点：

- 实现简单；
- 速度快；
- 相同输入与模型配置下结果确定；
- 适合短答案、结构化任务或创造性要求不高的场景。

缺点：

- 每一步局部最优不保证整段序列全局最优；
- 长文本可能重复、僵硬；
- 缺少多样性。

直观例子：

```text
第一步：
A 概率 0.55，B 概率 0.45
Greedy 选择 A

但后续：
A → 最佳完整序列概率 0.55 × 0.30 = 0.165
B → 最佳完整序列概率 0.45 × 0.90 = 0.405
```

第一步选 A 看起来更好，完整序列却可能不如从 B 开始。

**5. Beam Search**

Beam Search 在每一步保留多个得分最高的候选序列。若 beam size 为 $K$：

1. 从当前 $K$ 条序列分别扩展候选；
2. 计算扩展后序列的累计得分；
3. 只保留总体得分最高的 $K$ 条；
4. 直到结束，再选出最佳序列。

常见累计分数使用对数概率：

$$
\operatorname{score}(x_{1:T})
=
\sum_{t=1}^{T}
\log P(x_t\mid x_{<t})
$$

直接相乘很多小概率容易数值下溢，取对数后乘法变成加法。由于长序列会累加更多负对数概率，实际实现常加入长度惩罚。

Beam Search 的特点：

- 比 Greedy 更充分地搜索高概率序列；
- 常用于翻译、摘要等目标相对明确的任务；
- 计算和显存开销随 beam 数增加；
- 在开放式聊天中可能产生较保守、相似度高的答案；
- 它不是随机采样，默认仍偏向高概率序列。

**6. Multinomial Sampling**

采样不是固定选最大概率 token，而是按概率分布随机抽取：

```python
next_token = torch.multinomial(probs, num_samples=1)
```

概率高的 token 更容易被选中，但低概率 token 也有机会出现。因此采样能提高多样性，不过若直接从整个大词表采样，极低概率的奇怪 token 也可能被选中，所以通常结合 Temperature、Top-k 或 Top-p。

**7. Temperature**

Temperature 调整 Softmax 前 logits 的尺度：

$$
p_i(T)=
\frac{\exp(z_i/T)}
{\sum_j\exp(z_j/T)}
$$


![粘贴图片](/my-blog/resources/uploads/pasted-1785439666955.png)


- $T<1$：logits 差距被放大，分布更尖锐，输出更保守；
- $T=1$：保持原分布；
- $T>1$：logits 差距被缩小，分布更平坦，输出更多样；
- $T\to0$：理论上趋近于只选择最大 logit，但实际采样不要设置为 0。

Temperature 不会改变 logits 的排序，只改变概率差距。若使用 Hugging Face `generate()`，通常只有 `do_sample=True` 时 Temperature 才真正影响采样；Greedy Search 直接取最大值，缩放后最大值位置不变。

代码观察：

```python
import torch

logits = torch.tensor([4.0, 2.5, 1.5, 0.5])

for temperature in [0.5, 1.0, 2.0]:
    probs = torch.softmax(logits / temperature, dim=-1)
    print(f"T={temperature}: {probs.tolist()}")
```

**8. Top-k Sampling**

Top-k 只保留概率最高的 $k$ 个 token，其余 token 概率设为 0，再重新归一化：

$$
S_k=\operatorname{TopK}(p,k)
$$

然后从 $S_k$ 中采样。

特点：

- 候选数量固定；
- 实现简单；
- 当概率分布很尖时，$k$ 可能过大；
- 当概率分布很平时，$k$ 可能过小。

例如 `top_k=50` 表示每一步最多从概率最高的 50 个 token 中采样，不表示一定会生成第 50 个，也不表示最终文本只有 50 种可能。

**9. Top-p / Nucleus Sampling**

Top-p 先按概率从高到低排序，选择累计概率达到阈值 $p$ 的最小 token 集合：

$$
S_p=
\min\left\{
S:
\sum_{i\in S}p_i\ge p
\right\}
$$

再在集合中重新归一化并采样。


![粘贴图片](/my-blog/resources/uploads/pasted-1785439706699.png)


特点：

- 候选数量随分布动态变化；
- 模型很确定时，只保留少量 token；
- 模型不确定时，保留更多 token；
- 开放式生成中很常用。

Top-k 与 Top-p 可以同时使用。常见实现会先后应用多个 logits 处理器，最终从剩余候选中采样。两者同时设置时，候选范围通常会比单独使用更受限制。

**10. Repetition Penalty 与 No-repeat N-gram**

生成模型容易进入重复循环，常见控制方法有：

- `repetition_penalty`：降低已经出现过的 token 再次出现的相对分数；
- `no_repeat_ngram_size=n`：禁止出现已经生成过的相同 n-gram；
- frequency/presence 类惩罚：根据历史出现频率或是否出现调整分数。

惩罚太强会导致：

- 必要术语无法重复；
- 语法变差；
- 模型为了避开重复而改用不自然表达。

所以它是调节项，不是越大越好。

**11. 长度与停止条件**

常见参数：

| 参数 | 含义 |
|---|---|
| `max_new_tokens` | 最多生成多少个新 token，通常比 `max_length` 更直观 |
| `max_length` | 输入 token 与输出 token 的总长度上限 |
| `min_new_tokens` | 至少生成多少个新 token |
| `eos_token_id` | 生成该 token 时可以停止 |
| `stop_strings` | 生成指定字符串后停止，具体支持取决于版本和调用方式 |
| `early_stopping` | Beam Search 中控制何时停止搜索 |

不要混淆：

```text
max_new_tokens = 100：在 prompt 后最多再生成 100 个 token。
max_length = 100：prompt 与新生成内容合计最多 100 个 token。
```

**12. 训练阶段与生成阶段为什么不同？**

训练时，完整目标序列已知，可以采用 Teacher Forcing：

```text
输入：  [我, 爱, 机器, 学习]
目标：  [爱, 机器, 学习, EOS]
```

在 causal mask 约束下，每个位置只能利用当前位置之前的信息，但所有位置的预测可以在一次前向传播中并行计算。

生成时，下一个 token 尚不存在：

```text
第 1 步：prompt → token 1
第 2 步：prompt + token 1 → token 2
第 3 步：prompt + token 1 + token 2 → token 3
```

因此生成过程在时间维度上通常是串行的。KV Cache 可以避免每一步都重新计算全部历史 token 的 Key/Value，但不能消除“后一个 token 依赖前一个 token”的自回归顺序。

**13. Hugging Face 生成策略实验**

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
).to(device)
model.eval()

messages = [
    {"role": "user", "content": "用两句话解释什么是机器学习。"}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

def decode_new_tokens(output_ids):
    prompt_len = inputs["input_ids"].shape[1]
    new_ids = output_ids[:, prompt_len:]  # 只保留模型新生成的部分
    return tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0]

# Greedy：不采样，每一步选择最大概率 token。
greedy_ids = model.generate(
    **inputs,
    max_new_tokens=80,
    do_sample=False,
)
print("Greedy:\n", decode_new_tokens(greedy_ids))

# Sampling：设置随机种子后，同一环境中更容易复现实验。
set_seed(42)
sample_ids = model.generate(
    **inputs,
    max_new_tokens=80,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.05,
)
print("Sampling:\n", decode_new_tokens(sample_ids))
```

比较多组参数：

```python
settings = [
    {"name": "较保守", "temperature": 0.3, "top_p": 0.8, "top_k": 20},
    {"name": "均衡", "temperature": 0.7, "top_p": 0.9, "top_k": 50},
    {"name": "较发散", "temperature": 1.2, "top_p": 0.98, "top_k": 100},
]

for i, cfg in enumerate(settings):
    set_seed(100 + i)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=cfg["temperature"],
        top_p=cfg["top_p"],
        top_k=cfg["top_k"],
    )

    print("=" * 70)
    print(cfg["name"], cfg)
    print(decode_new_tokens(output_ids))
```

解释实验结果时，不要只看哪段“更好”，还要观察：

- 内容是否稳定；
- 是否出现重复；
- 是否偏离问题；
- 表达是否多样；
- 相同随机种子能否复现；
- 不同 prompt 是否需要不同参数。

**14. 常见参数组合**

| 任务倾向 | 常见思路 | 说明 |
|---|---|---|
| 分类、抽取、格式化输出 | Greedy 或低随机性 | 更强调稳定和格式一致 |
| 一般问答 | 中低 Temperature + Top-p | 保留一定表达多样性 |
| 创意写作 | 较高 Temperature + Top-p | 更发散，但事实可靠性可能下降 |
| 翻译、摘要 | Greedy 或 Beam Search | 具体效果依模型和任务评估决定 |
| 代码生成 | 通常较低 Temperature | 减少无意义分支，但可多次采样选优 |

这些不是固定法则。解码参数无法替代模型能力、提示设计、检索证据和结果评估。

**本轮常见面试题**

| 面试问题 | 标准回答 |
|---|---|
| 1. Logits 与概率有什么区别？ | Logits 是 LM Head 输出的未归一化分数，可以为负且总和不为 1；经过 Softmax 后才得到非负且总和为 1 的概率分布。 |
| 2. Greedy Search 的优缺点是什么？ | 每一步选最大概率 token，速度快、确定性强；但局部最优不保证完整序列最优，长文本可能重复、僵硬且缺少多样性。 |
| 3. Beam Search 如何工作？ | 每步保留若干条累计得分最高的候选序列，继续扩展并剪枝，最终选择整体得分较高的序列。它比 Greedy 搜索更充分，但计算开销更大。 |
| 4. Sampling 与 Greedy 的区别是什么？ | Greedy 固定选择最高概率 token；Sampling 按概率随机抽取，因此概率高的更常出现，但其他候选也有机会被选中。 |
| 5. Temperature 如何影响生成？ | 对 logits 除以 $T$ 后再 Softmax。$T<1$ 使分布更尖锐、输出更保守；$T>1$ 使分布更平坦、输出更多样。它不改变 logits 排序。 |
| 6. 为什么 `do_sample=False` 时 Temperature 通常不起作用？ | Greedy 只看最大 logit 的位置。正温度缩放不会改变排序，因此 argmax 不变；Temperature 主要用于改变采样概率。 |
| 7. Top-k 与 Top-p 的区别是什么？ | Top-k 固定保留概率最高的 $k$ 个 token；Top-p 保留累计概率达到阈值的最小集合，因此候选数量会随分布动态变化。 |
| 8. Top-k 和 Beam Search 是同一回事吗？ | 不是。Top-k 是每一步对可采样 token 的截断；Beam Search 维护多条完整候选序列，并按累计序列得分搜索。 |
| 9. `max_new_tokens` 与 `max_length` 有何区别？ | `max_new_tokens` 只限制新生成长度；`max_length` 限制输入和输出总 token 数。控制回答长度时通常前者更直观。 |
| 10. 为什么训练能并行、生成却通常串行？ | 训练时完整目标序列已知，可在 causal mask 下同时计算各位置损失；生成时第 $t+1$ 个 token 依赖实际生成的第 $t$ 个 token，因此时间步之间存在顺序依赖。 |
| 11. KV Cache 解决了什么问题？ | 缓存历史 token 的 Attention Key/Value，避免每一步重复计算全部历史表示，从而加速自回归推理；它不能让未来 token 提前产生。 |
| 12. 提高 Temperature 会让事实更正确吗？ | 不会。它只让概率分布更平坦、输出更多样，也可能增加偏离与错误。事实可靠性需要模型能力、证据和验证机制保障。 |

## 四、完整链路实验、排错与学习验收

**1. 建议的项目结构**

```text
day3/
├── day3_experiment.py
└── README.md
```

在已有虚拟环境中运行即可。若需要新建环境：

```bash
python -m venv .venv
```

这会在当前目录创建名为 `.venv` 的独立 Python 环境，避免 Day 3 依赖与系统 Python 或其他项目冲突。

Windows PowerShell 激活：

```powershell
.venv\Scripts\Activate.ps1
```

Windows CMD 激活：

```bat
.venv\Scripts\activate.bat
```

Linux/macOS 激活：

```bash
source .venv/bin/activate
```

安装依赖：

```bash
python -m pip install -U torch transformers accelerate sentencepiece
```

`accelerate` 可帮助 Transformers 处理设备与模型加载；本教程代码本身仍会显式判断 CPU 或 CUDA。

**2. 完整实验脚本**

将下面内容保存为 `day3_experiment.py`：

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TEXT = "大语言模型正在改变自然语言处理。"


def choose_device():
    """返回当前可用设备和适合该设备的模型数据类型。"""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def print_tokenizer_result(tokenizer, text):
    """展示文本、token、ID 和模型输入张量。"""
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    encoded = tokenizer(text, return_tensors="pt")

    print("\n[1] Tokenizer")
    print("原始文本:", text)
    print("tokens:", tokens)
    print("token IDs:", token_ids)
    print("input_ids shape:", tuple(encoded["input_ids"].shape))
    print("decode:", tokenizer.decode(token_ids))


def print_embedding_result(model, input_ids):
    """直接调用输入 Embedding，观察查表后的张量形状。"""
    embedding_layer = model.get_input_embeddings()
    embeddings = embedding_layer(input_ids)

    print("\n[2] Embedding")
    print("Embedding weight shape:", tuple(embedding_layer.weight.shape))
    print("Embedding output shape:", tuple(embeddings.shape))
    print("第一个 token 向量前 8 维:", embeddings[0, 0, :8])


@torch.no_grad()
def print_logits_result(model, tokenizer, inputs):
    """执行一次前向传播并查看最后位置的候选 token。"""
    outputs = model(**inputs)
    logits = outputs.logits
    last_logits = logits[:, -1, :]
    probs = torch.softmax(last_logits, dim=-1)

    top_probs, top_ids = torch.topk(probs, k=5, dim=-1)

    print("\n[3] LM Head 与 logits")
    print("logits shape:", tuple(logits.shape))
    print("最后位置概率 shape:", tuple(probs.shape))
    print("概率最高的 5 个候选:")

    for prob, token_id in zip(top_probs[0], top_ids[0]):
        token = tokenizer.decode([token_id.item()])
        print(f"token={token!r}, id={token_id.item()}, p={prob.item():.6f}")


@torch.no_grad()
def generate_text(model, tokenizer, device):
    """分别运行 Greedy 和 Sampling，比较输出差异。"""
    messages = [
        {"role": "user", "content": "用两句话解释 Tokenizer 的作用。"}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_len = inputs["input_ids"].shape[1]

    greedy_ids = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=False,
    )

    set_seed(42)
    sampling_ids = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.05,
    )

    greedy_text = tokenizer.decode(
        greedy_ids[0, prompt_len:],
        skip_special_tokens=True,
    )
    sampling_text = tokenizer.decode(
        sampling_ids[0, prompt_len:],
        skip_special_tokens=True,
    )

    print("\n[4] Generation")
    print("Greedy:\n", greedy_text)
    print("\nSampling:\n", sampling_text)


def main():
    device, dtype = choose_device()
    print("device:", device)
    print("dtype:", dtype)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    print_tokenizer_result(tokenizer, TEXT)

    inputs = tokenizer(TEXT, return_tensors="pt").to(device)
    print_embedding_result(model, inputs["input_ids"])
    print_logits_result(model, tokenizer, inputs)
    generate_text(model, tokenizer, device)


if __name__ == "__main__":
    main()
```

运行：

```bash
python day3_experiment.py
```

这条命令让当前 Python 解释器执行脚本。第一次运行会从模型仓库下载 tokenizer 和模型文件，时间取决于网络；后续通常会读取本地缓存。

**3. 你应该能解释的每一行核心输出**

假设得到：

```text
input_ids shape: (1, 9)
Embedding weight shape: (V, d)
Embedding output shape: (1, 9, d)
logits shape: (1, 9, V)
```

解释如下：

| 输出 | 含义 |
|---|---|
| `(1, 9)` | 一个样本，共 9 个 token |
| `(V, d)` | 词表中每个 token 都有一个 $d$ 维向量 |
| `(1, 9, d)` | 9 个 token 各自完成 Embedding 查表 |
| `(1, 9, V)` | 每个序列位置都对词表中 $V$ 个 token 给出 logits |
| `logits[:, -1, :]` | 取最后一个输入位置，用于分析下一个 token 的分布 |

注意：在标准 causal LM 前向输出中，位置 $t$ 的 logits 用于预测下一个 token。训练代码通常会对 labels 做一位 shift，模型库的损失实现会处理这一对应关系。

**4. 常见错误与排查**

| 现象                                              | 可能原因                    | 排查方法                                                       |
| ----------------------------------------------- | ----------------------- | ---------------------------------------------------------- |
| `ModuleNotFoundError: transformers`             | 当前解释器没有安装依赖             | 运行 `python -m pip show transformers`，确认安装位置与 VS Code 解释器一致 |
| 下载模型失败                                          | 网络、代理或模型站点连接问题          | 先在浏览器确认模型页面可访问，再检查终端代理与证书                                  |
| CUDA out of memory                              | 显存不足或其他进程占用             | 先运行 CPU；减少生成长度；关闭占显存进程；不要盲目重复运行多个模型实例                      |
| `expected all tensors to be on the same device` | 模型与输入不在同一设备             | 模型和 tokenizer 输出都 `.to(device)`                            |
| `Half` 类型在 CPU 上报错                              | CPU 对某些 FP16 运算支持不足     | CPU 使用 `torch.float32`，不要强行设置 FP16                         |
| 输出包含 prompt                                     | 解码了完整 `generate()` 返回序列 | 用 `output_ids[:, prompt_len:]` 截取新生成部分                     |
| 调了 Temperature 但结果不变                            | `do_sample=False`       | 采样参数需配合 `do_sample=True`                                   |
| batch padding 报错                                | tokenizer 没有 PAD token  | 检查 `tokenizer.pad_token`；必要时根据任务设置                         |
| 相同 Sampling 每次不同                                | 没固定随机种子                 | 使用 `set_seed()`；但硬件与算子差异仍可能造成细小不同                          |

**5. 完整知识链路**

```text
“机器学习”
↓ Tokenizer
tokens
↓ 词表映射
input_ids: [B, L]
↓ Embedding 查表
X: [B, L, d]
↓ Transformer
H: [B, L, d]
↓ LM Head
logits: [B, L, V]
↓ 取最后位置 + Softmax/解码处理
候选 token 分布
↓ Greedy / Beam / Sampling
下一个 token
↓ 追加到序列并重复
最终文本
```

**6. 今日精读论文**

以下论文均属于你指定的 `LLMForEverybody` 必读论文池。本日只精读与 Tokenizer、Embedding 和生成链路直接相关的部分，Attention 细节留到对应学习日系统展开。

| 论文                                                                                                                                                                  | 今日重点                                                                  | 与 Day 3 的对应关系                                     |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- | ------------------------------------------------- |
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762)                                                                                                       | Section 3.4 Embeddings and Softmax；结合 Figure 1 看输入 Embedding 与输出线性层位置 | 理解 Embedding 缩放、输出 Softmax，以及输入输出权重共享             |
| [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)                                                | Section 3，重点看 Input/Output Representations 与 Figure 2                 | 理解 WordPiece、Token/Segment/Position Embedding 的组合 |
| [Language Models are Unsupervised Multitask Learners（GPT-2）](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | Section 2.2 Input Representation                                      | 理解字节级输入表示为何能减少跨数据集预处理和 OOV 问题                     |

**建议精读顺序：**

```text
Transformer 3.4 → BERT Figure 2 与输入表示 → GPT-2 2.2
```

读完后应能回答：

- Transformer 为什么把 Embedding 乘以 $\sqrt{d_{\text{model}}}$？
- BERT 的输入为什么由三类 Embedding 相加？
- GPT-2 为什么从 Unicode 字符进一步走向字节级表示？
- 输入 Embedding 与输出 Softmax 权重为什么可以共享？

**7. 本轮学习验收**

完成下面任务才算真正学完 Day 3：

- 用同一 tokenizer 比较中文、英文、emoji 和生僻字符的 token 数；
- 手动画出一次 BPE 合并；
- 给定 $V$ 和 $d$，计算 Embedding 参数量和 FP16 存储量；
- 解释 `[B,L] → [B,L,d] → [B,L,V]`；
- 手算一个三分类 Softmax；
- 分别用 Greedy 和 Sampling 生成结果；
- 清楚说明 Temperature、Top-k、Top-p 各自改了什么；
- 口述“原始文本到最终文本”的完整链路。

**本轮常见面试题**

| 面试问题                                  | 标准回答                                                                                                                                  |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| 1. 请完整描述一句文本进入大模型后的流程。                | 文本先经 tokenizer 得到 token 和 ID；ID 经 Embedding 查表变成连续向量；Transformer 结合上下文产生 hidden states；LM Head 映射为词表 logits；再经解码策略选择下一个 token，循环直到停止。 |
| 2. 为什么 logits 的形状是 $[B,L,V]$？         | batch 中每条序列有 $L$ 个位置，每个位置都需要对词表中的 $V$ 个候选 token 打分，因此得到 $[B,L,V]$。                                                                    |
| 3. 推理时为什么通常只取最后位置 logits？             | 已有序列的前面位置对应过去的预测，当前要决定的是序列末尾之后的下一个 token，所以使用 `logits[:, -1, :]`。                                                                     |
| 4. 只把模型移到 GPU，输入不移会怎样？                | 模型参数和输入张量位于不同设备，运算会报 device mismatch；二者必须在同一设备上。                                                                                      |
| 5. 为什么 CPU 通常使用 FP32 而不是强行 FP16？      | 许多 CPU 对 FP16 算子支持和性能不理想，可能报错或更慢；GPU 上 FP16/BF16 才更常用于节省显存和加速。                                                                        |
| 6. 为什么 `generate()` 返回结果常包含输入 prompt？ | 对 causal LM，返回的是“原输入 token + 新生成 token”的完整序列。只要根据 prompt 长度切片即可得到新增部分。                                                                |
| 7. 固定随机种子后 Sampling 是否绝对可复现？          | 通常更容易复现，但不同硬件、库版本和非确定性算子仍可能造成差异。随机种子不是跨所有环境的绝对保证。                                                                                     |
| 8. Tokenizer 和模型可以随意混用吗？              | 通常不能。模型参数是按特定词表和 token ID 训练的，换 tokenizer 会导致同一 ID 表示不同 token，输入语义完全错位。                                                               |
| 9. 为什么更换 tokenizer 往往需要重新训练或调整模型？     | 词表、ID 与 Embedding 行是一一对应的。改变词表后，输入 Embedding 与输出 LM Head 的维度或含义会变化，需要重新初始化、映射或继续训练。                                                   |
| 10. 解码参数能否提升模型知识量？                    | 不能。解码只改变如何从已有概率分布中选 token，不会给模型增加新知识；它能影响稳定性、多样性和重复程度。                                                                                |

## Day 3 综合面试题

| 面试问题 | 标准回答 |
|---|---|
| 1. 为什么需要 Tokenizer？ | 神经网络接收数值张量，Tokenizer 将原始文本转为模型词表中的 token ID，并负责规范化、特殊 token、padding 和 mask 等输入处理。 |
| 2. 子词分词为什么比纯词级分词更适合大模型？ | 它能够复用词根、前后缀和常见片段，在控制词表大小的同时处理新词、拼写变化和多语言文本，避免词级词表无限增长。 |
| 3. BPE 的一句话核心是什么？ | 从细粒度单元开始，在整个训练语料上反复合并最常出现的相邻 token 对，逐步形成子词词表。 |
| 4. Byte-level BPE 为什么基本没有 OOV？ | 任意 Unicode 字符串都可以先编码为有限的字节序列，因此即使没有对应完整字符或词，也能退化为字节单元表示。 |
| 5. WordPiece、BPE、SentencePiece 如何区分？ | BPE 和 WordPiece 是构建子词的算法思路，前者偏高频合并，后者偏似然或词表收益；SentencePiece 是直接从原始文本训练和使用子词模型的工具体系，可采用 BPE 或 Unigram。 |
| 6. Token ID 为何不能表达语义距离？ | 它只是词表索引，编号顺序通常是人为或训练过程产生的，不具备连续数值的距离意义。 |
| 7. Embedding 层如何工作？ | 维护 $V\times d$ 的可训练矩阵，输入 ID 后按行索引，输出形状从 $[B,L]$ 变成 $[B,L,d]$。 |
| 8. Embedding 层为什么可以通过反向传播训练？ | 查出的矩阵行参与后续计算和损失，反向传播会把梯度传回被访问的行，使其逐渐学到对语言建模有用的表示。 |
| 9. 静态 Embedding 与上下文向量的差别是什么？ | 静态 Embedding 仅由 token ID 决定；上下文向量经过多层 Transformer 后还依赖周围 token，因此同一词在不同句子中的表示可以不同。 |
| 10. Weight Tying 的作用是什么？ | 共享输入 Embedding 和输出 LM Head 的词表权重，减少约 $V\times d$ 参数，并让输入与输出 token 表示共享参数空间。 |
| 11. LM Head 输出的 logits 表示什么？ | 对词表中每个 token 的未归一化打分，分数越大通常表示模型越偏好，但必须经 Softmax 才是概率。 |
| 12. Softmax 为什么要减去最大 logit？ | 为防止指数运算溢出；同时对所有 logits 减同一常数不会改变归一化后的概率。 |
| 13. Greedy Search 为什么可能不是全局最优？ | 它每一步只选当前最大概率 token，早期局部选择可能把后续带入较差路径，完整序列累计概率未必最高。 |
| 14. Beam Search 为什么使用对数概率？ | 序列概率是很多小概率连乘，容易数值下溢；取对数后变成求和，更稳定也更便于累计比较。 |
| 15. Temperature 是否会改变 token 排名？ | 正 Temperature 只缩放 logits 差异，不改变排序；它改变的是 Softmax 后的概率集中程度。 |
| 16. Top-p 相比 Top-k 的主要优势是什么？ | Top-p 根据当前概率分布动态决定候选数量，模型确定时集合小，不确定时集合大；Top-k 的候选数始终固定。 |
| 17. `do_sample=True` 表示什么？ | 表示从处理后的概率分布中随机采样，而不是固定执行 Greedy；Temperature、Top-k 和 Top-p 通常在此模式下发挥作用。 |
| 18. 为什么开放式生成常不用很大的 Beam Search？ | Beam Search 偏向高概率、相似的序列，可能让输出保守和重复；beam 数增大还会显著增加计算与内存开销。 |
| 19. 训练阶段的 Teacher Forcing 是什么？ | 训练每个位置时使用真实历史 token 作为条件，并把目标序列右移一位计算下一个 token 损失，因此多个位置可并行训练。 |
| 20. 为什么自回归推理难以在 token 维度完全并行？ | 第 $t+1$ 个 token 的输入依赖第 $t$ 个 token 的实际生成结果，未来 token 在前一步完成前并不存在。 |
| 21. KV Cache 与 Tokenizer 有什么关系？ | Tokenizer 决定序列 token 数；token 越多，缓存的历史位置越多。KV Cache 缓存 Attention 中的历史状态，用空间换取解码速度。 |
| 22. Tokenizer 为什么会影响上下文窗口的实际可用文本量？ | 上下文窗口按 token 数限制。同一段文字被切得越碎，就越快用完 token 预算，因此能容纳的实际字符或单词更少。 |
| 23. 为什么 tokenizer 与模型必须配套？ | 模型的 Embedding 第 $i$ 行对应训练时词表的第 $i$ 个 token。词表错位会让每个输入 ID 取到错误向量。 |
| 24. 一个模型输出重复文本，应优先从哪些方向分析？ | 检查模型本身、prompt、生成长度和训练数据，再尝试合理的 repetition penalty、Top-p、Temperature 或 no-repeat n-gram；不能只靠调高随机性。 |
| 25. 请用一句话串联 Day 3。 | Tokenizer 把文本变成离散 ID，Embedding 把 ID 变成连续向量，模型把向量变成词表 logits，解码策略再把 logits 逐步变回文本。 |

**官方资料与延伸阅读：**

- [Hugging Face Tokenizer API](https://huggingface.co/docs/transformers/main_classes/tokenizer)
- [Hugging Face Generation](https://huggingface.co/docs/transformers/main_classes/text_generation)
- [Hugging Face Generation Strategies](https://huggingface.co/docs/transformers/generation_strategies)
- [PyTorch `nn.Embedding`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
- [SentencePiece: A simple and language independent subword tokenizer and detokenizer](https://arxiv.org/abs/1808.06226)
- [Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909)
- [LLMForEverybody](https://github.com/luhengshiwo/LLMForEverybody)
