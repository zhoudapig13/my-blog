---
title: "『LLM学习笔记3』论文精读：Attention Is All You Need"
category: "internship"
tags:
  - "LLM"
date: "2026-08-03"
summary: "精读《Attention Is All You Need》，从循环模型的局限出发，系统拆解 Transformer 的编码器—解码器、自注意力、多头注意力、位置编码、训练方法与实验结果，并比较不同序列建模方式的复杂度。"
pdf: "https://arxiv.org/pdf/1706.03762"
pdfTitle: "Attention Is All You Need"
---

## 1. Introduction

**本节核心问题：为什么需要 Transformer？**

Transformer 出现之前，序列建模和序列转换任务主要依赖循环神经网络，尤其是 RNN、LSTM 和 GRU。这些模型通常采用 **Encoder–Decoder** 结构，广泛用于语言模型、机器翻译等任务。

所谓 **序列转换（sequence transduction）**，就是把一个序列映射成另一个序列，例如：

$$
\text{I love this movie}
\longrightarrow
\text{我喜欢这部电影}
$$

其中，Encoder 负责理解输入序列，Decoder 根据 Encoder 产生的表示，逐步生成输出序列。



**传统循环模型如何处理序列？**

对于序列中第 $t$ 个位置，循环神经网络根据当前位置的输入 $x_t$ 和前一个位置的隐藏状态 $h_{t-1}$，计算当前隐藏状态 $h_t$：

$$
h_t=f(h_{t-1},x_t)
$$

其中：

- $x_t$：第 $t$ 个位置的输入；
- $h_{t-1}$：模型处理到上一个位置时保存的信息；
- $h_t$：模型处理当前位置后得到的新状态；
- $f(\cdot)$：RNN、LSTM 或 GRU 所实现的状态更新函数。

以句子 “I love this movie” 为例，RNN 的处理顺序大致为：

$$
h_1
\rightarrow
h_2
\rightarrow
h_3
\rightarrow
h_4
$$

也就是：

$$
\text{I}
\rightarrow
\text{love}
\rightarrow
\text{this}
\rightarrow
\text{movie}
$$

计算 “love” 的表示之前，必须先处理 “I”；计算 “this” 之前，又必须先处理 “love”。

因此，循环模型具有明显的**顺序依赖**：

$$
h_t \text{ 依赖于 } h_{t-1}
$$

这会带来四个问题：

1. 序列中的位置必须按顺序计算；
2. 同一个样本内部很难并行处理所有位置；
3. 序列越长，连续计算步骤越多；
4. 长序列会增加训练时间和显存压力。



**“不能并行”到底是什么意思？**

需要区分两种并行：

- **不同样本之间的并行**：一个 batch 中的多个句子可以同时计算；
- **同一个句子内部的并行**：一个句子中的多个单词能否同时计算。

RNN 可以同时处理多个句子，但在每个句子内部，仍需要按照时间步逐个计算：

$$
x_1
\rightarrow
x_2
\rightarrow
x_3
\rightarrow
\cdots
\rightarrow
x_n
$$

因此，即使 GPU 擅长大规模并行计算，RNN 也无法充分利用这种能力。



**为什么已有的优化方法仍然不够？**

在 Transformer 之前，一些研究已经尝试使用矩阵分解、条件计算等方法提高循环模型的效率。

这些方法可以减少部分计算开销，但没有改变循环模型最根本的结构：

$$
h_t=f(h_{t-1},x_t)
$$

只要当前位置仍然依赖前一个位置，序列内部就仍然需要逐步计算。

所以，真正需要解决的核心问题是：

> 能否不再依赖循环结构，直接建立序列中不同位置之间的联系？



**Attention 在 Transformer 之前已经存在**

Attention 并不是 Transformer 首次提出的。

在早期的机器翻译模型中，Attention 通常与 RNN 一起使用。Decoder 在生成某个词时，可以通过 Attention 查看 Encoder 中不同位置的信息。

例如，将下面的英文翻译成中文：

$$
\text{The animal did not cross the street because it was tired.}
$$

当模型处理单词 “it” 时，需要判断它指的是 “animal” 还是 “street”。

Attention 可以让 “it” 直接关注前面的 “animal”，从而建立较远位置之间的联系。

但此前多数模型采用的是：

$$
\text{RNN}+\text{Attention}
$$

其中：

- RNN 仍然负责序列的主要表示；
- Attention 只是辅助模型寻找相关位置；
- 模型仍然没有摆脱循环计算。



**Transformer 的核心改变**

论文提出 Transformer，完全去除循环神经网络和卷积神经网络，只使用 Attention 建模序列关系。

其核心变化可以概括为：

$$
\text{RNN/CNN}+\text{Attention}
\quad\longrightarrow\quad
\text{Attention Only}
$$

Transformer 不再要求信息必须按照下面的方式逐步传递：

$$
x_1
\rightarrow
x_2
\rightarrow
x_3
\rightarrow
x_4
$$

而是让每个位置都可以直接观察其他位置：

$$
x_i
\longleftrightarrow
\{x_1,x_2,\ldots,x_n\}
$$

例如，在句子：

$$
\text{The cat that sat near the window was sleeping.}
$$

中，“cat”和“was sleeping”之间距离较远。

RNN 通常需要经过多个中间位置传递信息：

$$
\text{cat}
\rightarrow
\text{that}
\rightarrow
\text{sat}
\rightarrow
\cdots
\rightarrow
\text{sleeping}
$$

而 Self-Attention 可以让 “sleeping” 直接关注 “cat”，从而更容易捕捉长距离依赖。



**什么是全局依赖？**

全局依赖指的是：序列中的任意位置都可能与其他位置产生联系，无论它们相距多远。

假设输入序列为：

$$
X=(x_1,x_2,\ldots,x_n)
$$

Transformer 在计算第 $i$ 个位置的表示时，可以综合整个序列的信息：

$$
z_i
=
\sum_{j=1}^{n}
\alpha_{ij}v_j
$$

其中：

- $v_j$ 表示第 $j$ 个位置提供的信息；
- $\alpha_{ij}$ 表示第 $i$ 个位置对第 $j$ 个位置的关注程度；
- $\alpha_{ij}$ 越大，说明位置 $j$ 对位置 $i$ 越重要。

直观来看，第 $i$ 个词会对句子中的所有词分别分配一个权重，再将这些信息加权汇总。

需要注意：这个公式是对 Attention 思想的直观表达，具体计算形式将在后面的 Scaled Dot-Product Attention 中介绍。



**为什么 Transformer 更容易并行？**

Self-Attention 中所有位置之间的关系可以通过矩阵运算一次性计算。

RNN 的计算形式是：

$$
h_1
\rightarrow
h_2
\rightarrow
\cdots
\rightarrow
h_n
$$

Transformer 则可以将整个序列表示为矩阵：

$$
X=
\begin{bmatrix}
x_1\\
x_2\\
\vdots\\
x_n
\end{bmatrix}
$$

然后同时计算所有位置之间的相关性。

因此，Transformer 的优势主要体现在：

- 序列中的多个位置可以同时参与计算；
- 更适合 GPU 的大规模矩阵运算；
- 训练过程具有更高的并行度；
- 更容易建立长距离位置之间的联系。

这里强调的是**训练阶段的并行能力**。Transformer 的 Decoder 在实际生成文本时仍然通常需要逐词生成，不能简单理解成所有输出词都能同时生成。



**论文提出的主要贡献**

| 方面 | 主要内容 |
|---|---|
| 模型结构 | 提出完全基于 Attention 的 Transformer |
| 循环网络 | 移除 RNN、LSTM 和 GRU |
| 卷积网络 | 不依赖 CNN 建模序列关系 |
| 依赖建模 | 使用 Attention 建立全局依赖 |
| 并行能力 | 训练时可以同时处理序列中的多个位置 |
| 长距离关系 | 任意两个位置可以直接建立联系 |
| 训练效率 | 相较循环模型训练速度显著提高 |
| 实验结果 | 在机器翻译任务上取得当时新的最佳结果 |

论文指出，Transformer 在 8 张 NVIDIA P100 GPU 上训练约 12 小时，就能够在机器翻译任务中达到当时新的最佳性能。



**Introduction 的逻辑主线**

$$
\boxed{
\begin{aligned}
&\text{RNN/LSTM/GRU 是主流序列模型}\\
&\Downarrow\\
&\text{循环结构存在严格的前后依赖}\\
&\Downarrow\\
&\text{同一序列内部难以并行计算}\\
&\Downarrow\\
&\text{Attention 可以直接连接不同位置}\\
&\Downarrow\\
&\text{此前 Attention 仍然依附于 RNN}\\
&\Downarrow\\
&\text{提出完全基于 Attention 的 Transformer}
\end{aligned}
}
$$



**容易混淆的几个概念**

| 概念 | 通俗理解 |
|---|---|
| Sequence Modeling | 根据已有序列理解或预测后续内容 |
| Sequence Transduction | 将一个输入序列转换为另一个输出序列 |
| Sequential Computation | 必须完成前一步，才能进行下一步 |
| Attention | 根据相关程度，从不同位置提取信息 |
| Self-Attention | 一个序列内部的不同位置互相关注 |
| Global Dependency | 相距很远的位置也可以直接建立联系 |
| Parallelization | 多个位置可以同时进行计算 |



**一句话总结**

Transformer 的核心出发点是：用完全基于 Attention 的结构替代 RNN 和 CNN，减少序列计算中的顺序依赖，使模型能够更高效地并行训练，并直接建模序列中的全局关系和长距离依赖。

## 2. Background

**本节核心问题：在 Transformer 之前，人们如何减少序列模型中的顺序计算？Transformer 又进一步解决了什么问题？**

Transformer 并不是第一个尝试提高序列计算并行度的模型。在它之前，研究者已经提出了 Extended Neural GPU、ByteNet 和 ConvS2S 等模型。这些方法使用卷积神经网络代替传统循环神经网络，使序列中多个位置的表示可以并行计算。

不过，卷积虽然解决了部分顺序计算问题，却仍然难以高效建立相距很远的位置之间的联系。



**卷积模型如何处理序列？**

设输入序列长度为 $n$，卷积核大小为 $k$。

一个卷积层通常只能观察当前位置附近的有限范围。例如，当卷积核大小为 $3$ 时，第 $i$ 个位置主要接收：

$$
x_{i-1},\ x_i,\ x_{i+1}
$$

的信息。

因此，如果两个词相距很远，信息不能在一个普通卷积层中直接传递，而需要经过多层卷积。

例如，在句子：

> The book that I bought yesterday is interesting.

中，“book”和“is”之间隔着多个单词。若每一层卷积只能观察相邻位置，那么“book”的信息需要经过若干中间位置，才能传递到“is”。

其传播过程可以直观表示为：

$$
x_1
\rightarrow
x_2
\rightarrow
x_3
\rightarrow
\cdots
\rightarrow
x_n
$$

需要的卷积层数与两个位置之间的距离有关。



**什么是路径长度？**

论文特别关注两个位置之间的**路径长度（path length）**。

路径长度指的是：一个位置的信息传递到另一个位置，需要经过多少次网络运算。

假设句子中第 $1$ 个词的信息需要影响第 $10$ 个词：

- 如果必须经过第 $2,3,\ldots,9$ 个位置逐步传递，路径较长；
- 如果第 $10$ 个位置可以直接关注第 $1$ 个位置，路径长度就是常数级。

路径越短，模型通常越容易学习长距离依赖，因为信息和梯度不需要经过太多中间步骤。

可以把它类比成传话：

- **长路径**：信息依次经过很多人；
- **短路径**：发送者直接告诉接收者。

经过的人越多，信息越容易衰减、变形或丢失。



**不同卷积模型的路径长度**

论文提到两种典型情况。

1. **普通卷积 ConvS2S**

普通卷积只能连接局部位置。要使两个相距较远的位置建立联系，需要逐层扩大感受野。

其路径长度大致随位置距离线性增长：

$$
O(n)
$$

更准确地说，在论文的讨论中，ConvS2S 中两个任意位置建立联系所需的操作数量会随距离线性增加。

2. **ByteNet 的膨胀卷积**

ByteNet 使用膨胀卷积，使每一层可以跨越更大的距离，从而更快扩大感受野。

其最长路径可以降低到对数级：

$$
O(\log_k n)
$$

其中：

- $n$ 是序列长度；
- $k$ 是卷积核宽度。

对数级路径比线性路径短很多，但仍然需要经过多个中间层。



**Transformer 的关键区别**

Transformer 使用 Self-Attention，使序列中任意两个位置都可以在一个注意力层中直接建立联系。

假设输入序列为：

$$
X=(x_1,x_2,\ldots,x_n)
$$

对于任意位置 $x_i$，它都可以直接关注：

$$
x_1,x_2,\ldots,x_n
$$

因此，任意两个位置之间的最长路径长度为：

$$
O(1)
$$

这里的 $O(1)$ 表示路径长度不会随着序列长度 $n$ 的增加而增加。

例如，无论两个词相距 $2$ 个位置还是 $200$ 个位置，Self-Attention 都可以让它们在同一层中直接交互。



**三种结构的直观对比**

| 模型结构 | 信息传播方式 | 两个远距离位置之间的路径 |
|---|---|---|
| RNN | 按时间步逐个传递 | 随序列长度线性增加 |
| CNN | 通过多层局部卷积扩大范围 | 线性或对数增长 |
| Self-Attention | 任意位置直接建立联系 | 常数级 |

可以用下面的方式理解。

RNN：

$$
x_1
\rightarrow
x_2
\rightarrow
x_3
\rightarrow
x_4
$$

CNN：

$$
x_1
\rightarrow
\text{局部卷积层}
\rightarrow
\text{更高层卷积}
\rightarrow
x_4
$$

Self-Attention：

$$
x_1
\longleftrightarrow
x_4
$$



**Self-Attention 是什么？**

Self-Attention 有时也称为 **Intra-Attention**，中文可以理解为“序列内部的注意力”。

它的核心作用是：

> 让同一个序列中的不同位置相互建立联系，从而计算每个位置新的表示。

假设输入序列为：

$$
X=(x_1,x_2,\ldots,x_n)
$$

Self-Attention 在计算第 $i$ 个位置的新表示 $z_i$ 时，会参考整个序列：

$$
z_i
=
\sum_{j=1}^{n}
\alpha_{ij}v_j
$$

其中：

- $v_j$ 是第 $j$ 个位置提供的信息；
- $\alpha_{ij}$ 表示第 $i$ 个位置对第 $j$ 个位置的关注程度；
- 所有位置的信息按照注意力权重进行加权求和。

注意力权重通常满足：

$$
\sum_{j=1}^{n}\alpha_{ij}=1
$$

例如，在句子：

> The animal did not cross the street because it was tired.

中，计算 “it” 的表示时，模型可能给 “animal” 较大的注意力权重：

$$
\alpha_{\text{it},\text{animal}}
>
\alpha_{\text{it},\text{street}}
$$

于是模型更倾向于认为 “it” 指代 “animal”。

以上只是帮助理解 Self-Attention 的直观形式，具体的 Query、Key、Value 和缩放点积计算将在后续章节介绍。



**Self-Attention 在 Transformer 之前已经存在**

论文指出，Self-Attention 此前已经成功应用于多种任务，包括：

- 阅读理解；
- 抽象式文本摘要；
- 文本蕴含；
- 与任务无关的句子表示学习。

因此，Transformer 的贡献并不是首次提出 Self-Attention，而是将它提升为整个序列转换模型的主体结构。

此前常见的方式是：

$$
\text{RNN}+\text{Self-Attention}
$$

或者：

$$
\text{CNN}+\text{Attention}
$$

而 Transformer 采用：

$$
\text{Self-Attention}
+
\text{Feed-Forward Network}
$$

整个模型不再依赖序列对齐的循环结构或卷积结构。



**什么是 Sequence-Aligned Recurrence？**

论文中使用了 **sequence-aligned recurrence** 这一表达，可以理解为：

> 每个序列位置都对应一个循环计算步骤。

例如，长度为 $n$ 的序列需要依次进行 $n$ 个时间步：

$$
t=1,2,\ldots,n
$$

每个时间步计算：

$$
h_t=f(h_{t-1},x_t)
$$

序列位置和计算步骤严格对齐，因此无法跳过前面的时间步。

Transformer 去除了这种计算模式，使每个位置不再必须等待前一个位置完成计算。



**End-to-End Memory Networks 与 Transformer 的区别**

论文还提到 End-to-End Memory Networks。

这类模型使用基于注意力的记忆机制，而不是传统的序列对齐循环，在简单语言问答和语言模型任务上表现良好。

但它们仍然采用了**循环式注意力机制**：模型可能需要反复读取记忆，多次执行注意力操作。

Transformer 则将 Self-Attention 作为基础网络层，直接用于计算整个输入和输出序列的表示。



**Transformer 在本节中的核心创新定位**

根据作者的表述，Transformer 是第一个完全依靠 Self-Attention 计算输入和输出表示的序列转换模型，同时不使用：

- 序列对齐的 RNN；
- 卷积神经网络。

可以概括为：

$$
\boxed{
\text{Transformer}
=
\text{完全基于 Self-Attention 的序列转换模型}
}
$$

它的目标不仅是取消循环计算，还包括：

1. 提高序列内部计算的并行度；
2. 缩短长距离位置之间的信息传播路径；
3. 直接建立全局依赖；
4. 避免依赖多层卷积逐步扩大感受野。



**Self-Attention 是否没有缺点？**

Self-Attention 可以让所有位置两两交互，但这也会带来一个问题。

对于长度为 $n$ 的序列，需要计算每一对位置之间的关系，因此关系数量约为：

$$
n\times n=n^2
$$

其计算复杂度会包含：

$$
O(n^2)
$$

这一项。

当序列特别长时，全局 Self-Attention 的计算量和显存占用会明显增加。

论文在本节暂时没有展开复杂度公式，而是在后面的 **Why Self-Attention** 部分详细比较 Self-Attention、RNN 和 CNN 的计算复杂度。



**为什么多头注意力可能有帮助？**

论文指出，Self-Attention 通过加权平均汇总多个位置的信息，可能降低有效分辨率。

通俗来说，如果只进行一次加权平均，不同类型的信息可能混合在一起。

例如，一个单词可能同时需要关注：

- 主语；
- 谓语；
- 指代对象；
- 位置关系；
- 语义相关词。

单个注意力结果需要把这些关系压缩到同一个表示中。

Transformer 使用 Multi-Head Attention 缓解这一问题，让不同注意力头可以从不同表示子空间和不同位置提取信息。

可以直观理解为：

$$
\text{Head}_1:\text{关注语法关系}
$$

$$
\text{Head}_2:\text{关注指代关系}
$$

$$
\text{Head}_3:\text{关注语义关系}
$$

不过，这些具体功能并不是人工预先规定的，而是模型在训练中自行学习的。

Multi-Head Attention 的正式定义将在第 3.2.2 节介绍。



**本节涉及的模型对比**

| 模型 | 基础结构 | 是否可以并行处理序列位置 | 长距离关系的传播特点 |
|---|---|---:|---|
| RNN/LSTM/GRU | 循环网络 | 较弱 | 需要逐时间步传递 |
| ConvS2S | 普通卷积 | 可以 | 路径随距离线性增长 |
| ByteNet | 膨胀卷积 | 可以 | 路径可以缩短到对数级 |
| End-to-End Memory Network | 循环式注意力 | 部分可以 | 通过多次访问记忆建立关系 |
| Transformer | Self-Attention | 可以 | 任意位置可在一层内直接联系 |



**本节逻辑主线**

$$
\boxed{
\begin{aligned}
&\text{RNN 的顺序计算限制并行}\\
&\Downarrow\\
&\text{CNN 可以并行计算所有位置}\\
&\Downarrow\\
&\text{但远距离位置仍需经过多层卷积}\\
&\Downarrow\\
&\text{Self-Attention 可以直接连接任意位置}\\
&\Downarrow\\
&\text{此前 Self-Attention 多作为辅助模块}\\
&\Downarrow\\
&\text{Transformer 完全依靠 Self-Attention}\\
&\Downarrow\\
&\text{不再使用序列对齐的 RNN 或 CNN}
\end{aligned}
}
$$

**一句话总结**

Transformer 之前的卷积模型虽然提高了并行计算能力，但远距离信息仍需要经过多层网络才能相互影响；Transformer 使用 Self-Attention，让任意两个位置可以在一层中直接建立联系，并首次构建了完全不依赖序列对齐循环和卷积的序列转换模型。


## 3. Model Architecture

**本节核心问题：Transformer 的整体结构是什么？数据如何在 Encoder、Decoder、Attention 和前馈神经网络之间流动？**

Transformer 仍然采用经典的 **Encoder–Decoder** 框架，但不再使用 RNN 或 CNN，而是使用以下模块构建整个模型：

- Multi-Head Attention；
- Position-wise Feed-Forward Network；
- Residual Connection；
- Layer Normalization；
- Positional Encoding。


![粘贴图片](/my-blog/resources/uploads/pasted-1785778861683.png)


整体流程可以概括为：

$$
\text{Input Tokens}
\rightarrow
\text{Embedding + Positional Encoding}
\rightarrow
\text{Encoder}
\rightarrow
\text{Decoder}
\rightarrow
\text{Linear}
\rightarrow
\text{Softmax}
$$

其中：

- Encoder 负责理解输入序列；
- Decoder 根据 Encoder 的输出，逐步生成目标序列；
- Attention 负责不同位置之间的信息交互；
- FFN 负责对每个位置的特征进行进一步加工。



**Encoder–Decoder 的基本任务**

设输入序列为：

$$
X=(x_1,x_2,\ldots,x_n)
$$

Encoder 将输入序列转换为一组连续表示：

$$
Z=(z_1,z_2,\ldots,z_n)
$$

其中，$z_i$ 是第 $i$ 个输入 token 经过多层 Encoder 处理后得到的表示。

可以将 $Z$ 理解为：

> Encoder 对整个输入句子的理解结果。

Decoder 在给定 $Z$ 的条件下，逐步生成输出序列：

$$
Y=(y_1,y_2,\ldots,y_m)
$$

生成第 $t$ 个输出时，Decoder 使用：

- Encoder 产生的输入表示 $Z$；
- 已经生成的输出 $y_1,\ldots,y_{t-1}$。

其条件概率可以写为：

$$
p(y_t\mid y_1,y_2,\ldots,y_{t-1},Z)
$$

这种根据已有输出继续预测下一个输出的方式称为 **自回归生成（Auto-Regressive Generation）**。

例如，将：

> I love machine learning.

翻译为：

> 我喜欢机器学习。

Decoder 的生成过程是：

$$
\text{我}
\rightarrow
\text{喜欢}
\rightarrow
\text{机器}
\rightarrow
\text{学习}
$$

生成“机器”时，Decoder 可以使用“我”和“喜欢”，但不能提前使用尚未生成的“学习”。



**3.1 Encoder and Decoder Stacks**

原始 Transformer 使用：

$$
N=6
$$

即 Encoder 和 Decoder 都由 6 层堆叠而成。

这里的“相同层”是指每一层的结构相同，但不同层拥有各自独立的参数，并不共享参数。



**Encoder 的整体结构**

每个 Encoder Layer 包含两个子层：

1. Multi-Head Self-Attention；
2. Position-wise Feed-Forward Network。

一个 Encoder Layer 的信息流为：

$$
\text{输入}
\rightarrow
\text{Multi-Head Self-Attention}
\rightarrow
\text{Add \& Norm}
\rightarrow
\text{Feed-Forward Network}
\rightarrow
\text{Add \& Norm}
\rightarrow
\text{输出}
$$

完整 Encoder 由 6 个这样的结构连续堆叠：

$$
H^{(0)}
\rightarrow
H^{(1)}
\rightarrow
H^{(2)}
\rightarrow
\cdots
\rightarrow
H^{(6)}
$$

其中：

- $H^{(0)}$ 是输入 Embedding 与 Positional Encoding 的和；
- $H^{(6)}$ 是 Encoder 的最终输出。


![粘贴图片](/my-blog/resources/uploads/pasted-1785778981019.png)




**Encoder Self-Attention 在做什么？**

Encoder 的 Self-Attention 允许输入序列中的每个位置观察所有位置。

假设输入句子是：

> The animal did not cross the street because it was tired.

当模型计算 “it” 的表示时，可以同时观察：

- animal；
- cross；
- street；
- tired；
- 句子中的其他 token。

模型可能发现 “it” 与 “animal” 的关系更强，因此在更新 “it” 的表示时，给予 “animal” 更大的注意力权重。

Self-Attention 的 Query、Key 和 Value 都来自同一个输入序列：

$$
Q=XW^Q
$$

$$
K=XW^K
$$

$$
V=XW^V
$$

因为 $Q$、$K$ 和 $V$ 都由 $X$ 产生，所以称为 **Self-Attention**，即序列内部的注意力。

需要注意：虽然 $Q$、$K$、$V$ 来自同一个输入 $X$，但它们使用不同的投影矩阵，因此通常并不相等。



**Residual Connection：残差连接**

每个 Attention 子层和 FFN 子层外部都使用残差连接。Attention 负责让不同 token 交换信息，FFN 负责让每个 token 单独加工刚收到的信息。

对于一个子层 $\operatorname{Sublayer}(\cdot)$，首先计算：

$$
x+\operatorname{Sublayer}(x)
$$

其中：

- $x$ 是子层原始输入；
- $\operatorname{Sublayer}(x)$ 是子层新计算出的信息。

可以将残差连接理解为：

$$
\text{新表示}
=
\text{原始信息}
+
\text{新学到的信息}
$$

例如，如果某个 Attention 层暂时没有学到非常有效的信息，残差连接仍然可以保留原来的输入，而不会让原始信息完全丢失。

残差连接还有助于梯度在深层网络中传播，使多层 Transformer 更容易训练。



**Layer Normalization：层归一化**

原始 Transformer 在残差相加后使用 Layer Normalization：

$$
\operatorname{LayerNorm}
\left(
x+\operatorname{Sublayer}(x)
\right)
$$

完整顺序是：

$$
x
\rightarrow
\operatorname{Sublayer}(x)
\rightarrow
x+\operatorname{Sublayer}(x)
\rightarrow
\operatorname{LayerNorm}
$$

Layer Normalization 会对单个样本中某个 token 的特征维度进行归一化。

假设某个 token 的表示为：

$$
x=(x_1,x_2,\ldots,x_d)
$$

其均值为：

$$
\mu=
\frac{1}{d}
\sum_{i=1}^{d}x_i
$$

方差为：

$$
\sigma^2=
\frac{1}{d}
\sum_{i=1}^{d}(x_i-\mu)^2
$$

归一化后的结果可以表示为：

$$
\operatorname{LayerNorm}(x)
=
\gamma
\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}
+\beta
$$

其中，$\gamma$ 和 $\beta$ 是可学习参数。

论文第 3.1 节主要说明了 LayerNorm 的使用位置，没有展开它的完整公式。这里给出公式是为了帮助理解其作用。



**为什么所有子层的输出维度必须相同？**

残差连接需要进行：

$$
x+\operatorname{Sublayer}(x)
$$

矩阵相加要求两者具有相同维度。

因此，原始 Transformer 中：

- Embedding 输出维度；
- Attention 输出维度；
- FFN 最终输出维度；

都统一设置为：

$$
d_{\text{model}}=512
$$

假设序列长度为 $n$，则每层 Encoder 的输入和输出形状均为：

$$
n\times512
$$



**Decoder 的整体结构**

Decoder 同样由 6 个 Decoder Layer 堆叠而成。

每个 Decoder Layer 包含三个子层：

1. Masked Multi-Head Self-Attention；
2. Encoder–Decoder Attention；
3. Position-wise Feed-Forward Network。

一个 Decoder Layer 的信息流为：

$$
\text{目标序列输入}
\rightarrow
\text{Masked Self-Attention}
\rightarrow
\text{Add \& Norm}
$$

$$
\rightarrow
\text{Encoder--Decoder Attention}
\rightarrow
\text{Add \& Norm}
$$

$$
\rightarrow
\text{Feed-Forward Network}
\rightarrow
\text{Add \& Norm}
\rightarrow
\text{输出}
$$

与 Encoder 相比，Decoder 多出了：

- Masked Self-Attention 中的因果掩码；
- 读取 Encoder 输出的 Encoder–Decoder Attention。


![粘贴图片](/my-blog/resources/uploads/pasted-1785779070656.png)




**为什么 Decoder Self-Attention 要使用 Mask？**

训练时，完整的目标句子是已知的。

假设目标序列为：

$$
(\text{我},\text{喜欢},\text{机器},\text{学习})
$$

如果不加任何限制，那么模型在预测“我”时，可能直接看到后面的“喜欢”“机器”和“学习”。

这相当于在做题时提前看到了答案。

为了保持自回归性质，Decoder 使用 Mask，使第 $i$ 个位置只能关注：

$$
1,2,\ldots,i
$$

不能关注未来位置：

$$
i+1,i+2,\ldots,m
$$

以长度为 4 的序列为例，Mask 可以表示为：

$$
M=
\begin{bmatrix}
0 & -\infty & -\infty & -\infty\\
0 & 0 & -\infty & -\infty\\
0 & 0 & 0 & -\infty\\
0 & 0 & 0 & 0
\end{bmatrix}
$$

Attention 分数加上 Mask：

$$
S_{\text{masked}}
=
\frac{QK^{\mathsf T}}{\sqrt{d_k}}+M
$$

再经过 Softmax：

$$
A=
\operatorname{softmax}
\left(
\frac{QK^{\mathsf T}}{\sqrt{d_k}}+M
\right)
$$

由于：

$$
\exp(-\infty)=0
$$

被屏蔽位置的注意力权重会变为 0。

例如，第二个位置只能看到第一个和第二个位置：

$$
[\alpha_{21},\alpha_{22},0,0]
$$



**Outputs Shifted Right 是什么意思？**

Transformer 结构图中，Decoder 的输入标记为：

> Outputs shifted right

意思是：将真实目标序列整体向右移动一位，再作为 Decoder 输入。

假设真实目标序列为：

$$
(\text{我},\text{喜欢},\text{机器学习},\text{<EOS>})
$$

Decoder 输入为：

$$
(\text{<BOS>},\text{我},\text{喜欢},\text{机器学习})
$$

训练目标仍然是：

$$
(\text{我},\text{喜欢},\text{机器学习},\text{<EOS>})
$$

对应关系如下：

| Decoder 当前输入 | 当前需要预测的 token |
|---|---|
| `<BOS>` | 我 |
| `<BOS>, 我` | 喜欢 |
| `<BOS>, 我, 喜欢` | 机器学习 |
| `<BOS>, 我, 喜欢, 机器学习` | `<EOS>` |

其中：

- `<BOS>` 表示句子开始；
- `<EOS>` 表示句子结束。

Shifted Right 和 Mask 共同保证：

> 模型在预测第 $i$ 个 token 时，只能使用前面的真实 token，不能看到当前答案和未来答案。



**Encoder–Decoder Attention 在做什么？**

Decoder 的第二个 Attention 子层用于读取 Encoder 的最终输出。

在 Encoder–Decoder Attention 中：

$$
Q=H_{\text{decoder}}W^Q
$$

$$
K=H_{\text{encoder}}W^K
$$

$$
V=H_{\text{encoder}}W^V
$$

即：

- Query 来自 Decoder；
- Key 来自 Encoder；
- Value 来自 Encoder。

它的作用是：

> Decoder 在生成当前目标词时，从输入序列中寻找最相关的信息。

例如，将：

> I ate an apple.

翻译为：

> 我吃了一个苹果。

当 Decoder 生成“苹果”时，Query 会与 Encoder 中所有输入位置的 Key 进行匹配，并可能给 “apple” 较大的注意力权重。

当 Decoder 生成“吃”时，则可能重点关注 “ate”。

因此，Encoder–Decoder Attention 建立了输入序列与输出序列之间的对应关系。



**Transformer 中三种 Attention 的区别**

| Attention 类型 | Query 来源 | Key 来源 | Value 来源 | 是否屏蔽未来 |
|---|---|---|---|---|
| Encoder Self-Attention | Encoder | Encoder | Encoder | 否 |
| Decoder Self-Attention | Decoder | Decoder | Decoder | 是 |
| Encoder–Decoder Attention | Decoder | Encoder | Encoder | 否 |

可以简单记忆为：

- Encoder Self-Attention：输入序列内部互相看；
- Decoder Self-Attention：输出序列只能向前看；
- Encoder–Decoder Attention：Decoder 回头查看 Encoder。



**3.2 Attention**

论文将 Attention 描述为：

> 将一个 Query 和一组 Key–Value 对映射为一个输出。

可以将 Query、Key 和 Value 类比为查资料：

- Query：你当前想查询的问题；
- Key：每条资料的标题或关键词；
- Value：每条资料真正包含的内容。

Attention 首先比较 Query 与不同 Key 的匹配程度，然后根据匹配程度对 Value 进行加权求和。

其直观形式为：

$$
\operatorname{Output}
=
\sum_{j=1}^{n}\alpha_jv_j
$$

其中：

- $v_j$ 是第 $j$ 个 Value；
- $\alpha_j$ 是第 $j$ 个 Value 对当前 Query 的重要程度。

注意力权重通常满足：

$$
\alpha_j\geq0
$$

以及：

$$
\sum_{j=1}^{n}\alpha_j=1
$$

权重越大，对应 Value 对最终输出的贡献越大。



**Query、Key 和 Value 的直观含义**

对于一个 token，可以产生三个不同的向量：

$$
q_i=x_iW^Q
$$

$$
k_i=x_iW^K
$$

$$
v_i=x_iW^V
$$

它们承担不同角色：

| 向量 | 作用 | 通俗理解 |
|---|---|---|
| Query | 表示当前 token 想寻找什么信息 | 我需要找什么 |
| Key | 表示当前 token 可以被怎样匹配 | 我具有什么标签 |
| Value | 表示当前 token 真正提供的内容 | 我能提供什么信息 |

例如，计算代词 “it” 的新表示时：

- “it”的 Query 表示它当前需要寻找指代对象；
- “animal”和“street”的 Key 分别表示它们是否适合作为指代对象；
- 匹配完成后，再提取对应的 Value。



**3.2.1 Scaled Dot-Product Attention**

Transformer 使用的 Attention 称为：

> Scaled Dot-Product Attention，即缩放点积注意力。

公式为：

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^{\mathsf T}}{\sqrt{d_k}}
\right)V
$$


![粘贴图片](/my-blog/resources/uploads/pasted-1785779180412.png)


整个计算过程可以拆成四步。



**第一步：计算 Query 和 Key 的匹配分数**

$$
S=QK^{\mathsf T}
$$

其中，$S$ 称为 Attention Score Matrix。

矩阵中的第 $i$ 行、第 $j$ 列表示：

$$
S_{ij}=q_i\cdot k_j
$$

即第 $i$ 个 Query 与第 $j$ 个 Key 的点积。

一般来说：

- 点积越大，匹配程度越高；
- 点积越小，匹配程度越低。



**第二步：对点积结果进行缩放**

$$
\hat{S}
=
\frac{QK^{\mathsf T}}{\sqrt{d_k}}
$$

其中，$d_k$ 是 Query 和 Key 的维度。



**第三步：使用 Softmax 得到注意力权重**

$$
A=
\operatorname{softmax}(\hat{S})
$$

Softmax 通常沿每一行进行，因此：

$$
\sum_{j=1}^{n}A_{ij}=1
$$

矩阵 $A$ 中的每一行表示：

> 某个 Query 对所有 Key 分配的注意力权重。



**第四步：对 Value 进行加权求和**

$$
O=AV
$$

最终得到：

$$
O=
\operatorname{softmax}
\left(
\frac{QK^{\mathsf T}}{\sqrt{d_k}}
\right)V
$$

完整流程可以写为：

$$
QK^{\mathsf T}
\rightarrow
\frac{QK^{\mathsf T}}{\sqrt{d_k}}
\rightarrow
\operatorname{softmax}
\rightarrow
\operatorname{softmax}
\left(
\frac{QK^{\mathsf T}}{\sqrt{d_k}}
\right)V
$$



**Q、K、V 的矩阵维度**

设：

- Query 数量为 $n_q$；
- Key 和 Value 数量为 $n_k$；
- Query 和 Key 的维度为 $d_k$；
- Value 的维度为 $d_v$。

则：

$$
Q\in\mathbb{R}^{n_q\times d_k}
$$

$$
K\in\mathbb{R}^{n_k\times d_k}
$$

$$
V\in\mathbb{R}^{n_k\times d_v}
$$

因为：

$$
K^{\mathsf T}
\in
\mathbb{R}^{d_k\times n_k}
$$

所以：

$$
QK^{\mathsf T}
\in
\mathbb{R}^{n_q\times n_k}
$$

注意力权重矩阵的形状也是：

$$
A\in
\mathbb{R}^{n_q\times n_k}
$$

最终输出为：

$$
O=AV
\in
\mathbb{R}^{n_q\times d_v}
$$



**一个简单的 Attention 数值例子**

假设某个 Query 与三个 Key 的缩放后分数为：

$$
[2,1,0]
$$

经过 Softmax 后，大约得到：

$$
[0.665,0.245,0.090]
$$

假设三个 Value 分别为：

$$
v_1,\quad v_2,\quad v_3
$$

则 Attention 输出为：

$$
o
=
0.665v_1
+
0.245v_2
+
0.090v_3
$$

说明当前 Query 与第一个 Key 最匹配，因此第一个 Value 对输出的贡献最大。



**为什么要除以 $\sqrt{d_k}$？**

假设 Query 和 Key 中的每个分量相互独立，并满足：

$$
\mathbb{E}[q_i]=\mathbb{E}[k_i]=0
$$

$$
\operatorname{Var}(q_i)=\operatorname{Var}(k_i)=1
$$

它们的点积为：

$$
q\cdot k
=
\sum_{i=1}^{d_k}q_ik_i
$$

该点积的方差大约为：

$$
\operatorname{Var}(q\cdot k)=d_k
$$

因此，随着 $d_k$ 增大，点积的绝对值通常也会增大。

如果直接将很大的数输入 Softmax，例如：

$$
\operatorname{softmax}([20,1,0])
$$

输出会非常接近：

$$
[1,0,0]
$$

Softmax 会进入饱和区域，大部分位置的梯度变得非常小，不利于模型学习。

除以 $\sqrt{d_k}$ 后：

$$
\operatorname{Var}
\left(
\frac{q\cdot k}{\sqrt{d_k}}
\right)
\approx1
$$

因此，缩放操作的核心作用是：

> 控制点积结果的数值范围，避免 Softmax 过早饱和，使梯度更加稳定。



**Dot-Product Attention 与 Additive Attention**

论文比较了两种常见 Attention。

| Attention 类型 | 匹配函数 | 特点 |
|---|---|---|
| Additive Attention | 使用小型前馈神经网络计算匹配分数 | 表达方式灵活 |
| Dot-Product Attention | 直接计算 Query 与 Key 的点积 | 适合矩阵并行计算 |
| Scaled Dot-Product Attention | 点积后除以 $\sqrt{d_k}$ | 缓解高维点积过大问题 |

Additive Attention 与 Dot-Product Attention 的理论复杂度相近。

但 Dot-Product Attention 可以直接使用高度优化的矩阵乘法，因此在实际计算中通常：

- 速度更快；
- 显存利用效率更高。



**3.2.2 Multi-Head Attention**


![粘贴图片](/my-blog/resources/uploads/pasted-1785779257004.png)


如果只使用单个 Attention，模型会将不同位置的信息汇总到同一个加权平均结果中。

但是，一个词可能同时涉及多种关系。

例如，在句子：

> The animal did not cross the street because it was tired.

单词 “it” 可能同时需要关注：

- “animal”：指代关系；
- “tired”：语义关系；
- “because”：因果关系；
- “cross”：动作关系。

单头 Attention 需要将这些关系混合在同一个表示中。

Multi-Head Attention 的核心思想是：

> 将 Query、Key 和 Value 投影到多个不同的表示子空间，在每个子空间中独立执行 Attention。

第 $i$ 个注意力头为：

$$
\operatorname{head}_i
=
\operatorname{Attention}
\left(
QW_i^Q,
KW_i^K,
VW_i^V
\right)
$$

多个注意力头的输出先进行拼接：

$$
\operatorname{Concat}
\left(
\operatorname{head}_1,
\operatorname{head}_2,
\ldots,
\operatorname{head}_h
\right)
$$

然后经过输出投影矩阵 $W^O$：

$$
\operatorname{MultiHead}(Q,K,V)
=
\operatorname{Concat}
\left(
\operatorname{head}_1,\ldots,\operatorname{head}_h
\right)W^O
$$



**Multi-Head Attention 的参数矩阵**

每个注意力头都有独立的投影矩阵：

$$
W_i^Q
\in
\mathbb{R}^{d_{\text{model}}\times d_k}
$$

$$
W_i^K
\in
\mathbb{R}^{d_{\text{model}}\times d_k}
$$

$$
W_i^V
\in
\mathbb{R}^{d_{\text{model}}\times d_v}
$$

拼接后的结果通过：

$$
W^O
\in
\mathbb{R}^{hd_v\times d_{\text{model}}}
$$

重新映射回模型维度。



**原始 Transformer 的多头设置**

原始 Transformer 使用：

$$
h=8
$$

即 8 个注意力头。

模型维度为：

$$
d_{\text{model}}=512
$$

每个头的 Query、Key 和 Value 维度为：

$$
d_k=d_v=\frac{d_{\text{model}}}{h}
$$

代入数值：

$$
d_k=d_v=\frac{512}{8}=64
$$

因此，每个头输出 64 维表示。

8 个头拼接后：

$$
8\times64=512
$$

所以：

$$
\operatorname{Concat}
\left(
\operatorname{head}_1,\ldots,\operatorname{head}_8
\right)
\in
\mathbb{R}^{n\times512}
$$

最终仍然保持 $d_{\text{model}}=512$，便于后续残差相加。



**多头注意力的直观理解**

可以把多个注意力头理解成多个人从不同角度阅读同一句话。

例如：

$$
\begin{aligned}
\operatorname{head}_1 &: \text{可能关注主谓关系}\\
\operatorname{head}_2 &: \text{可能关注指代关系}\\
\operatorname{head}_3 &: \text{可能关注语义相似性}\\
\operatorname{head}_4 &: \text{可能关注位置关系}
\end{aligned}
$$

但需要注意：

> 每个头具体学习什么关系，并不是人工提前规定的，而是模型根据训练任务自行学习的。

多头注意力的优势是允许模型同时关注：

- 不同位置；
- 不同特征子空间；
- 不同类型的句法或语义关系。



**为什么使用 8 个头不会使计算量变成 8 倍？**

如果单头 Attention 直接使用完整维度：

$$
d_k=512
$$

则每次注意力计算都在 512 维空间中进行。

多头注意力虽然进行了 8 次 Attention，但每个头只有：

$$
d_k=64
$$

所以每个头的计算规模更小。

8 个 64 维头的总计算量，与一个 512 维单头 Attention 的计算量大致处于同一数量级。



**3.2.3 Applications of Attention in the Transformer**

Transformer 使用 Multi-Head Attention 的方式共有三种。

**第一种：Encoder Self-Attention**

$$
Q=K=V=\text{上一层 Encoder 的输出}
$$

每个输入位置都可以关注所有输入位置。

作用是建模输入序列内部的关系。



**第二种：Decoder Masked Self-Attention**

$$
Q=K=V=\text{上一层 Decoder 的输出}
$$

但通过 Mask 屏蔽未来位置。

第 $i$ 个位置只能关注：

$$
1,2,\ldots,i
$$

作用是建模已经生成的目标序列，同时保持自回归性质。



**第三种：Encoder–Decoder Attention**

$$
Q=\text{Decoder 上一子层的输出}
$$

$$
K=V=\text{Encoder 的最终输出}
$$

作用是让 Decoder 在生成每个目标 token 时，从输入序列中提取相关信息。



**3.3 Position-wise Feed-Forward Networks**

每个 Encoder Layer 和 Decoder Layer 中都包含一个前馈神经网络：

$$
\operatorname{FFN}(x)
=
\max(0,xW_1+b_1)W_2+b_2
$$

由于：

$$
\max(0,z)=\operatorname{ReLU}(z)
$$

因此也可以写为：

$$
\operatorname{FFN}(x)
=
\operatorname{ReLU}(xW_1+b_1)W_2+b_2
$$

原始 Transformer 中，FFN 的维度变化为：

$$
512
\rightarrow
2048
\rightarrow
512
$$

即：

$$
d_{\text{model}}=512
$$

$$
d_{\text{ff}}=2048
$$

第一层线性变换将特征从 512 维扩展到 2048 维：

$$
h=
xW_1+b_1
$$

经过 ReLU：

$$
h'=
\operatorname{ReLU}(h)
$$

再通过第二层线性变换压回 512 维：

$$
y=
h'W_2+b_2
$$



**为什么 Attention 后面还需要 FFN？**

Attention 主要负责：

> 让不同 token 之间交换和汇总信息。

FFN 主要负责：

> 对每个 token 自己的特征进行非线性加工。

可以简单理解为：

- Attention：不同位置互相交流；
- FFN：每个位置独立消化和处理交流后的信息。

Attention 的输出主要是不同 Value 的加权组合。

如果只有 Attention，模型的非线性表达能力会受到限制。FFN 中的 ReLU 为模型引入了额外的非线性变换能力。



**Position-wise 是什么意思？**

Position-wise 表示：

> 同一个 FFN 分别作用于序列中的每个位置，不在不同位置之间进行信息混合。

假设输入矩阵为：

$$
X=
\begin{bmatrix}
x_1\\
x_2\\
x_3
\end{bmatrix}
$$

则：

$$
\operatorname{FFN}(X)
=
\begin{bmatrix}
\operatorname{FFN}(x_1)\\
\operatorname{FFN}(x_2)\\
\operatorname{FFN}(x_3)
\end{bmatrix}
$$

在同一个 Transformer Layer 内：

- 所有位置共享相同的 $W_1,W_2,b_1,b_2$；
- 不同位置分别独立进行计算。

但是，不同 Transformer Layer 使用不同的 FFN 参数。

论文指出，Position-wise FFN 也可以看成两个卷积核大小为 1 的卷积层。



**3.4 Embeddings and Softmax**

神经网络不能直接处理离散 token，因此需要先通过 Embedding 将 token 映射为连续向量。

假设词表大小为：

$$
|\mathcal{V}|
$$

Embedding 矩阵为：

$$
E
\in
\mathbb{R}^{|\mathcal{V}|\times d_{\text{model}}}
$$

原始 Transformer 中：

$$
d_{\text{model}}=512
$$

因此，每个 token 都会被映射为一个 512 维向量：

$$
e_i\in\mathbb{R}^{512}
$$



**Decoder 如何预测下一个 token？**

Decoder 最后一层输出为：

$$
H_{\text{decoder}}
\in
\mathbb{R}^{m\times d_{\text{model}}}
$$

通过一个线性变换映射到词表空间：

$$
\operatorname{logits}_t
=
h_tW_{\text{vocab}}+b
$$

其中：

$$
W_{\text{vocab}}
\in
\mathbb{R}^{d_{\text{model}}\times|\mathcal{V}|}
$$

因此：

$$
\operatorname{logits}_t
\in
\mathbb{R}^{|\mathcal{V}|}
$$

Logits 表示模型对词表中每个 token 给出的原始分数。

再通过 Softmax 得到概率：

$$
p(y_t=j)
=
\frac{
\exp(\operatorname{logits}_{t,j})
}{
\sum_{k=1}^{|\mathcal{V}|}
\exp(\operatorname{logits}_{t,k})
}
$$

模型通常选择概率较大的 token 作为下一个输出，或者使用 Beam Search 等方式进行解码。



**Weight Sharing：权重共享**

论文让以下三部分共享同一个权重矩阵：

1. Encoder 输入 Embedding；
2. Decoder 输入 Embedding；
3. Decoder 输出前的线性变换。

这种方法通常称为：

> Weight Tying，即权重绑定或权重共享。

权重共享可以：

- 减少参数数量；
- 让输入词表示与输出词分类空间保持联系；
- 提高参数利用效率。

论文还对 Embedding 乘以：

$$
\sqrt{d_{\text{model}}}
$$

因此，输入表示为：

$$
X_{\text{embedding}}
=
\sqrt{d_{\text{model}}}
\cdot E[X]
$$

在加入位置编码前，提高 token 语义表示的信号强度，避免它在数值尺度上被位置编码淹没。随后再与 Positional Encoding 相加。



**3.5 Positional Encoding**

Transformer 没有 RNN，也没有 CNN。

因此，模型结构本身无法天然感知 token 的顺序。

例如：

> 我喜欢你

和：

> 你喜欢我

包含相似的 token，但语义完全不同。

如果没有位置信息，Self-Attention 只能看到 token 内容，无法可靠区分：

- 哪个 token 在前；
- 哪个 token 在后；
- 两个 token 相距多远。

因此，Transformer 将位置编码加入词嵌入：

$$
X_{\text{input}}
=
X_{\text{embedding}}
+
PE
$$

Positional Encoding 的维度必须与 Embedding 相同：

$$
PE
\in
\mathbb{R}^{n\times d_{\text{model}}}
$$

这样二者才能直接相加。



**正弦与余弦位置编码**

论文使用固定的正弦和余弦函数构造位置编码。

对于偶数维：

$$
PE_{(pos,2i)}
=
\sin
\left(
\frac{pos}
{10000^{2i/d_{\text{model}}}}
\right)
$$

对于奇数维：

$$
PE_{(pos,2i+1)}
=
\cos
\left(
\frac{pos}
{10000^{2i/d_{\text{model}}}}
\right)
$$

其中：

- $pos$ 表示 token 在序列中的位置；
- $i$ 表示特征维度的编号；
- $d_{\text{model}}$ 表示模型维度。

例如：

- $pos=0$ 表示第一个位置；
- $pos=1$ 表示第二个位置；
- 不同的 $i$ 对应不同频率。

每个位置都会得到一个独特的位置向量。



**为什么不同维度要使用不同频率？**

不同维度上的正弦和余弦函数具有不同的变化速度。

某些维度变化较快，适合区分相邻位置；某些维度变化较慢，适合表示较大范围的位置关系。

可以类比钟表：

- 秒针变化快；
- 分针变化较慢；
- 时针变化最慢。

多个不同频率结合，就可以同时表示不同尺度的位置信息。

论文中的波长按照几何级数变化，大致从：

$$
2\pi
$$

增加到：

$$
10000\cdot2\pi
$$



**为什么选择正弦和余弦函数？**

作者希望模型能够较容易地学习相对位置关系。

根据三角函数公式：

$$
\sin(a+b)
=
\sin a\cos b
+
\cos a\sin b
$$

$$
\cos(a+b)
=
\cos a\cos b
-
\sin a\sin b
$$

对于固定偏移量 $k$，位置 $pos+k$ 的编码可以由位置 $pos$ 的正弦和余弦编码线性表示。

因此，模型可能比较容易学习：

- 前一个位置；
- 后一个位置；
- 相距 $k$ 个位置；
- 其他相对位置关系。

论文将这一点作为选择正弦位置编码的动机，并没有给出严格的理论证明。



**固定位置编码与可学习位置编码**

论文也尝试了可学习的位置 Embedding：

$$
PE_{pos}
=
\text{一个可训练的向量}
$$

实验结果表明：

> 固定正弦位置编码与可学习位置编码取得了几乎相同的效果。

作者最终选择固定的正弦位置编码，是因为它可能具有更好的长度外推能力，即可能应用于训练阶段没有出现过的更长序列。



**完整 Encoder 数据流**

输入 token 序列为：

$$
X=(x_1,x_2,\ldots,x_n)
$$

首先计算 Embedding 并加入位置编码：

$$
H^{(0)}
=
\sqrt{d_{\text{model}}}
\operatorname{Embedding}(X)
+
PE
$$

随后经过 6 层 Encoder：

$$
H^{(l)}
=
\operatorname{EncoderLayer}^{(l)}
\left(
H^{(l-1)}
\right),
\qquad
l=1,2,\ldots,6
$$

最终得到：

$$
Z=H^{(6)}
$$

$Z$ 将作为所有 Decoder Layer 中 Encoder–Decoder Attention 的 Key 和 Value 来源。



**完整 Decoder 数据流**

首先将目标序列右移：

$$
Y_{\text{shifted}}
=
(\text{<BOS>},y_1,y_2,\ldots,y_{m-1})
$$

计算 Embedding 并加入位置编码：

$$
S^{(0)}
=
\sqrt{d_{\text{model}}}
\operatorname{Embedding}
\left(
Y_{\text{shifted}}
\right)
+
PE
$$

随后经过 6 层 Decoder：

$$
S^{(l)}
=
\operatorname{DecoderLayer}^{(l)}
\left(
S^{(l-1)},Z
\right),
\qquad
l=1,2,\ldots,6
$$

最终通过 Linear 和 Softmax：

$$
P(Y)
=
\operatorname{softmax}
\left(
S^{(6)}W_{\text{vocab}}+b
\right)
$$

得到每个位置对下一个 token 的预测概率。



**Transformer 各模块的分工**

| 模块 | 主要作用 |
|---|---|
| Token Embedding | 将离散 token 转换为连续向量 |
| Positional Encoding | 注入 token 的顺序和位置信息 |
| Encoder Self-Attention | 建模输入序列内部的全局关系 |
| Decoder Masked Self-Attention | 建模已有输出，并屏蔽未来信息 |
| Encoder–Decoder Attention | 让 Decoder 从输入序列中读取信息 |
| Multi-Head Attention | 从多个表示子空间学习不同关系 |
| Position-wise FFN | 独立加工每个位置的特征 |
| Residual Connection | 保留原始信息并帮助梯度传播 |
| Layer Normalization | 稳定中间表示与训练过程 |
| Linear Layer | 将 Decoder 表示映射到词表空间 |
| Softmax | 得到下一个 token 的概率分布 |



**容易混淆的概念**

| 概念 | 含义 |
|---|---|
| Self-Attention | Query、Key、Value 来自同一个序列 |
| Encoder–Decoder Attention | Query 来自 Decoder，Key 和 Value 来自 Encoder |
| Mask | 禁止 Decoder 当前位置查看未来 token |
| Shifted Right | 将目标序列右移一位作为 Decoder 输入 |
| Multi-Head | 在多个表示子空间中并行执行 Attention |
| Position-wise FFN | 对每个 token 独立使用同一个前馈网络 |
| Positional Encoding | 为没有循环结构的 Transformer 注入顺序信息 |
| Auto-Regressive | 根据已经生成的 token 预测下一个 token |



**本节逻辑主线**

$$
\boxed{
\begin{aligned}
&\text{Input Token Embedding + Positional Encoding}\\
&\Downarrow\\
&\text{Encoder Self-Attention 建模输入内部关系}\\
&\Downarrow\\
&\text{Encoder FFN 加工各位置特征}\\
&\Downarrow\\
&\text{得到输入序列的完整表示}\\
&\Downarrow\\
&\text{Decoder Masked Self-Attention 读取历史输出}\\
&\Downarrow\\
&\text{Encoder--Decoder Attention 查询输入信息}\\
&\Downarrow\\
&\text{Decoder FFN 加工各位置特征}\\
&\Downarrow\\
&\text{Linear + Softmax 预测下一个 Token}
\end{aligned}
}
$$

**一句话总结**

Transformer 通过堆叠 Multi-Head Attention、Position-wise FFN、残差连接和 LayerNorm 构建 Encoder 与 Decoder。Encoder 使用 Self-Attention 理解完整输入；Decoder 使用 Masked Self-Attention 读取已有输出，再通过 Encoder–Decoder Attention 查询输入信息，最后经 Linear 和 Softmax 逐步生成目标序列。


## 4. Why Self-Attention

**本节核心问题：为什么 Transformer 选择 Self-Attention，而不是继续使用 RNN 或 CNN？**

论文从三个角度比较 Self-Attention、循环网络和卷积网络：

1. 每层的计算复杂度；
2. 可以并行计算到什么程度；
3. 序列中两个位置之间的最大路径长度。

这三个角度分别对应三个现实问题：

- 一层网络需要进行多少计算？
- 序列中的所有 token 能不能同时计算？
- 相距很远的 token 需要经过多少层或多少步骤才能交换信息？



**比较不同结构时使用的符号**

| 符号 | 含义 |
|---|---|
| $n$ | 序列长度，即 token 数量 |
| $d$ | 每个 token 的表示维度 |
| $k$ | 卷积核大小 |
| $r$ | Restricted Self-Attention 中每个位置关注的局部范围 |

例如，一个句子经过分词后包含 128 个 token，每个 token 表示为 512 维向量，则：

$$
n=128,\qquad d=512
$$



**论文给出的整体比较**

| Layer Type | 每层计算复杂度 | 最少顺序操作数 | 最大路径长度 |
|---|---:|---:|---:|
| Self-Attention | $O(n^2d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(nd^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(knd^2)$ | $O(1)$ | $O(\log_k n)$ |
| Restricted Self-Attention | $O(rnd)$ | $O(1)$ | $O(n/r)$ |

这张表不能只看“计算复杂度”一列。Transformer 的优势主要来自：

- Self-Attention 可以高度并行；
- 任意两个位置可以在一层内直接交互；
- 长距离依赖的信息传播路径很短。



**第一项比较：每层计算复杂度**

Self-Attention 的主要计算来自：

$$
QK^{\mathsf T}
$$

假设：

$$
Q,K\in\mathbb{R}^{n\times d}
$$

计算 $QK^{\mathsf T}$ 时，需要比较序列中每一对 token，因此会产生一个：

$$
n\times n
$$

的注意力分数矩阵。

所以 Self-Attention 的复杂度为：

$$
O(n^2d)
$$

其中：

- $n^2$ 来自所有 token 两两比较；
- $d$ 来自每次比较涉及的向量维度。

直观来说，如果序列长度扩大为原来的两倍，注意力关系数量大约变为原来的四倍：

$$
n^2\longrightarrow(2n)^2=4n^2
$$

这也是标准 Self-Attention 处理超长序列时计算量和显存占用较大的原因。



**RNN 的计算复杂度**

在循环网络中，每个时间步通常需要进行隐藏状态之间的矩阵变换：

$$
h_t=f(h_{t-1},x_t)
$$

如果隐藏状态维度为 $d$，一个 $d$ 维向量乘以一个 $d\times d$ 的矩阵，主要复杂度为：

$$
O(d^2)
$$

序列共有 $n$ 个时间步，因此一层循环网络的总复杂度约为：

$$
O(nd^2)
$$



**什么时候 Self-Attention 比 RNN 计算量更小？**

比较两者：

$$
\text{Self-Attention}:O(n^2d)
$$

$$
\text{RNN}:O(nd^2)
$$

两者相除：

$$
\frac{n^2d}{nd^2}
=
\frac{n}{d}
$$

因此，当：

$$
n<d
$$

时，Self-Attention 的理论计算量小于 RNN。

在论文所讨论的机器翻译任务中，句子的 token 数量通常小于表示维度，因此这一条件经常成立。

例如，当：

$$
n=128,\qquad d=512
$$

时：

$$
\frac{n}{d}
=
\frac{128}{512}
=
\frac{1}{4}
$$

从主要复杂度项看，Self-Attention 的计算量约为循环层的四分之一。

不过，这只是忽略常数项和具体实现后的理论比较。真实速度还会受到硬件、矩阵实现、显存访问等因素影响。



**CNN 的计算复杂度**

普通卷积层需要在每个位置使用大小为 $k$ 的卷积核，并对 $d$ 维输入和输出进行变换，因此复杂度约为：

$$
O(knd^2)
$$

其中：

- $n$ 表示需要处理的序列位置数量；
- $k$ 表示每个位置观察多少个相邻位置；
- $d^2$ 来自输入维度和输出维度之间的矩阵变换。

与 RNN 相比，卷积网络虽然可以并行处理所有位置，但通常还要乘上卷积核大小 $k$。



**第二项比较：最少顺序操作数**

这里衡量的不是总计算量，而是：

> 完成一层计算时，最少必须依次执行多少个步骤？



**RNN 为什么需要 $O(n)$ 次顺序操作？**

RNN 的当前状态依赖前一个状态：

$$
h_t=f(h_{t-1},x_t)
$$

因此：

$$
h_1
\rightarrow
h_2
\rightarrow
h_3
\rightarrow
\cdots
\rightarrow
h_n
$$

计算 $h_2$ 之前必须先得到 $h_1$，计算 $h_3$ 之前必须先得到 $h_2$。

长度为 $n$ 的序列至少需要：

$$
O(n)
$$

次顺序操作。

即使 GPU 有大量并行计算单元，也不能同时计算同一个样本中的所有时间步。



**Self-Attention 为什么只需要 $O(1)$ 次顺序操作？**

Self-Attention 将整个序列组织成矩阵：

$$
X=
\begin{bmatrix}
x_1\\
x_2\\
\vdots\\
x_n
\end{bmatrix}
$$

然后统一计算：

$$
Q=XW^Q,\qquad
K=XW^K,\qquad
V=XW^V
$$

再通过矩阵乘法计算：

$$
\operatorname{Attention}(Q,K,V)
=
\operatorname{softmax}
\left(
\frac{QK^{\mathsf T}}{\sqrt{d_k}}
\right)V
$$

所有位置的 Attention 可以在同一个矩阵运算中并行完成，因此一层 Self-Attention 所需的顺序操作数为：

$$
O(1)
$$

这里的 $O(1)$ 不表示“只进行一次乘法”，而是表示：

> 顺序计算步骤不会随着序列长度 $n$ 增加。

序列越长，矩阵会越大，总计算量仍然会增加；但不同位置不需要像 RNN 一样依次等待。



**CNN 为什么也可以达到 $O(1)$ 的顺序操作数？**

同一个卷积层中的所有位置也可以同时计算。

例如，对于序列：

$$
(x_1,x_2,\ldots,x_n)
$$

卷积层可以并行计算：

$$
z_1,z_2,\ldots,z_n
$$

因此，单个卷积层的顺序操作数同样是：

$$
O(1)
$$

不过，CNN 要让相距很远的 token 发生信息交换，通常需要堆叠多层卷积，这就引出了第三项比较。



**第三项比较：最大路径长度**

最大路径长度指：

> 序列中任意两个位置要实现信息交互，最多需要经过多少个计算步骤或网络层？

假设句子中第一个词需要影响最后一个词：

$$
x_1\longrightarrow x_n
$$

如果它们可以直接交换信息，路径就很短；如果信息必须经过大量中间位置或中间层，路径就很长。

路径长度之所以重要，是因为前向信息和反向梯度都需要沿着这些路径传播。

一般而言：

> 路径越短，模型越容易学习长距离依赖。



**RNN 的最大路径长度为什么是 $O(n)$？**

RNN 中，第一个位置的信息要影响最后一个位置，需要逐步传递：

$$
h_1
\rightarrow
h_2
\rightarrow
h_3
\rightarrow
\cdots
\rightarrow
h_n
$$

因此最远的两个位置之间，需要经过大约 $n$ 个步骤：

$$
O(n)
$$

例如：

> The animal that was standing near the busy road was tired.

若前面的 “animal” 要影响后面的 “was tired”，信息必须经过许多中间时间步。

虽然 LSTM 和 GRU 可以缓解长距离信息丢失问题，但信息路径本身依然很长。



**Self-Attention 的最大路径长度为什么是 $O(1)$？**

在一层全局 Self-Attention 中，每个位置都可以直接关注所有位置：

$$
x_i
\longleftrightarrow
x_j
$$

无论两个 token 相隔 2 个位置还是 200 个位置，它们都可以在同一个 Attention 层中直接建立联系。

因此最大路径长度为：

$$
O(1)
$$

例如，在句子：

> The animal that was standing near the busy road was tired.

计算 “tired” 的表示时，可以直接给 “animal” 较高的注意力权重：

$$
\alpha_{\text{tired},\text{animal}}
$$

不需要通过每个中间词逐步传递。



**CNN 的最大路径长度为什么更长？**

普通卷积每层只能观察局部区域。

例如，卷积核大小为：

$$
k=3
$$

则一个位置在一层中只能直接访问自己和附近位置。

第一层中：

$$
x_i
\longrightarrow
\{x_{i-1},x_i,x_{i+1}\}
$$

堆叠第二层后，感受野才会进一步扩大。

因此，如果两个 token 相距很远，就必须堆叠多层卷积，才能让它们相互影响。

对于连续卷积核，建立远距离联系可能需要：

$$
O(n/k)
$$

层。

如果采用膨胀卷积，使每层的感受野快速扩大，则最长路径可以降低到：

$$
O(\log_k n)
$$

但仍然长于 Self-Attention 的：

$$
O(1)
$$



**用“传话”理解路径长度**

假设 A 要把消息告诉相距很远的 D。

RNN 类似：

$$
A\rightarrow B\rightarrow C\rightarrow D
$$

信息需要经过多个人逐步传递。

CNN 类似：

$$
A\rightarrow\text{局部小组}
\rightarrow\text{更大范围的小组}
\rightarrow D
$$

通过逐层扩大交流范围建立联系。

Self-Attention 类似：

$$
A\rightarrow D
$$

A 可以直接把信息告诉 D。

这就是为什么 Self-Attention 更适合建立长距离依赖。



**为什么短路径有助于学习长距离依赖？**

深度学习依靠反向传播更新参数。

如果两个相关位置之间路径很长，梯度需要经过很多中间步骤：

$$
\frac{\partial L}{\partial h_n}
\rightarrow
\frac{\partial L}{\partial h_{n-1}}
\rightarrow
\cdots
\rightarrow
\frac{\partial L}{\partial h_1}
$$

在传播过程中，梯度可能逐渐变小或变得不稳定。

如果路径较短，信息和梯度需要经过的中间运算更少，因此模型更容易学习远距离位置之间的关系。

需要注意，这并不意味着 Self-Attention 一定能够完美学会所有长距离关系，而是它在网络结构上提供了更直接的信息通道。



**Self-Attention 在所有情况下都更省计算吗？**

不是。

Self-Attention 的复杂度为：

$$
O(n^2d)
$$

其中包含序列长度的平方项。

当序列非常长时：

$$
n^2
$$

会迅速增大。

例如：

- 普通句子可能只有几十或几百个 token；
- 长文档、音频、视频或高分辨率图像可能包含成千上万个位置。

此时，全局 Self-Attention 需要计算所有位置之间的关系，计算量和显存占用都会很高。

所以，Self-Attention 的主要权衡是：

| 优势 | 代价 |
|---|---|
| 所有位置可以并行计算 | 注意力矩阵大小为 $n\times n$ |
| 任意位置可以直接交互 | 长序列下复杂度为 $O(n^2d)$ |
| 长距离路径短 | 显存占用随 $n^2$ 增长 |



**Restricted Self-Attention 是什么？**

为降低长序列中的计算量，可以让每个位置只关注附近大小为 $r$ 的区域，而不是关注全部 $n$ 个位置。

全局 Self-Attention 中，第 $i$ 个位置可以关注：

$$
x_1,x_2,\ldots,x_n
$$

Restricted Self-Attention 中，第 $i$ 个位置只关注附近位置，例如：

$$
x_{i-r/2},\ldots,x_i,\ldots,x_{i+r/2}
$$

此时，每个位置只与大约 $r$ 个位置计算 Attention。

复杂度从：

$$
O(n^2d)
$$

降低为：

$$
O(rnd)
$$

如果：

$$
r\ll n
$$

计算量会显著降低。



**Restricted Self-Attention 的代价**

局部注意力不能让任意两个位置在一层中直接交互。

一个位置的信息每层只能传播大约 $r$ 个位置，因此远距离信息需要经过多层传递。

其最大路径长度变为：

$$
O(n/r)
$$

因此，Restricted Self-Attention 做出的权衡是：

$$
\boxed{
\text{降低计算量}
\quad\Longleftrightarrow\quad
\text{增加长距离信息传播路径}
}
$$

论文将限制注意力范围作为未来可能研究的方向。后来许多长序列 Transformer 确实采用了局部注意力、稀疏注意力等思想，但这些后续工作不属于本论文第 4 节的内容。



**Separable Convolution：可分离卷积**

论文还讨论了可分离卷积。

普通卷积的复杂度约为：

$$
O(knd^2)
$$

可分离卷积可以将复杂度降低为：

$$
O(knd+nd^2)
$$

其中：

- $O(knd)$ 对应沿序列位置进行局部卷积；
- $O(nd^2)$ 对应特征维度之间的线性变换。

论文指出，即使令：

$$
k=n
$$

使卷积能够覆盖完整序列，可分离卷积的复杂度也与下面两部分组合相近：

$$
\text{Self-Attention}
+
\text{Position-wise FFN}
$$

这进一步支持了 Transformer 使用 Self-Attention 和 FFN 组合的设计。



**Self-Attention 是否更具有可解释性？**

论文认为，Self-Attention 可能带来一定的可解释性优势。

因为 Attention 会产生权重矩阵：

$$
A=
\operatorname{softmax}
\left(
\frac{QK^{\mathsf T}}{\sqrt{d_k}}
\right)
$$

其中：

$$
A_{ij}
$$

可以观察为第 $i$ 个位置对第 $j$ 个位置的注意力权重。

论文附录中的可视化显示，一些不同的注意力头似乎学习到了不同类型的关系，例如：

- 长距离依赖；
- 指代关系；
- 某些句法结构；
- 某些语义联系。

例如，某个头可能让代词 “its” 重点关注对应的名词。

不过，更准确的表述应该是：

> Attention 权重可以帮助观察模型的信息聚合行为，但不能简单等同于模型完整的推理过程或严格的因果解释。

论文在这里主要报告的是观察到的现象，而不是证明每个注意力头都有固定、清晰的语言学功能。



**为什么 Transformer 最终选择 Self-Attention？**

从论文的比较来看，Self-Attention 同时满足以下特点：

| 需求 | Self-Attention 的表现 |
|---|---|
| 降低顺序计算 | 所有位置可以并行处理 |
| 建模长距离关系 | 任意位置可以直接建立联系 |
| 缩短传播路径 | 最大路径长度为 $O(1)$ |
| 适应常见句子长度 | 当 $n<d$ 时，复杂度通常优于循环层 |
| 提供可观察关系 | 可以可视化不同位置之间的注意力权重 |

其主要局限是：

$$
O(n^2d)
$$

的计算复杂度会在长序列中形成瓶颈。



**四类结构的通俗对比**

| 结构 | token 之间如何交流 | 能否并行 | 长距离联系 | 主要问题 |
|---|---|---:|---:|---|
| RNN | 按顺序逐步传递 | 较弱 | 路径长 | 训练难以并行 |
| CNN | 通过局部卷积逐层传播 | 可以 | 需要多层 | 感受野逐层扩大 |
| Self-Attention | 所有位置直接两两交互 | 可以 | 一层即可 | 长序列计算量大 |
| Restricted Self-Attention | 只与附近位置交互 | 可以 | 需要多层传播 | 全局联系不再直接 |



**本节逻辑主线**

$$
\boxed{
\begin{aligned}
&\text{比较每层计算复杂度}\\
&\Downarrow\\
&\text{当 }n<d\text{ 时，Self-Attention 通常更有优势}\\
&\Downarrow\\
&\text{比较顺序操作数}\\
&\Downarrow\\
&\text{Self-Attention 可以并行计算所有位置}\\
&\Downarrow\\
&\text{比较最大路径长度}\\
&\Downarrow\\
&\text{任意位置可以在一层内直接交互}\\
&\Downarrow\\
&\text{更容易建模长距离依赖}\\
&\Downarrow\\
&\text{但长序列下需要承担 }O(n^2d)\text{ 的代价}
\end{aligned}
}
$$

**一句话总结**

论文选择 Self-Attention，是因为它既能并行处理序列中的所有位置，又能让任意两个 token 在一层内直接交换信息，从而显著缩短长距离依赖的传播路径；它的主要代价是注意力矩阵随序列长度平方增长，因此在超长序列中计算和显存开销较大。


## 5. Training

**本节核心问题：原始 Transformer 使用什么数据、硬件、优化器、学习率策略和正则化方法完成训练？**

这一节介绍 Transformer 的训练方案，主要包括：

- 训练数据与分词方式；
- Batch 的组织方式；
- 硬件配置与训练时长；
- Adam 优化器；
- Warmup 学习率策略；
- Dropout 与 Label Smoothing。

需要注意，本节重点不是模型内部结构，而是回答：

> 已经搭建好的 Transformer，具体应该怎样训练，才能稳定收敛并取得较好的泛化性能？



**5.1 Training Data and Batching：训练数据与批处理**

论文分别在两个机器翻译任务上训练 Transformer：

| 任务 | 训练数据规模 | 分词方式 | 词表大小 |
|---|---:|---|---:|
| WMT 2014 英语→德语 | 约 450 万个句子对 | Byte-Pair Encoding，BPE | 约 37,000 |
| WMT 2014 英语→法语 | 约 3,600 万个句子对 | WordPiece | 约 32,000 |

所谓“句子对”，是指一条源语言句子与对应的目标语言翻译，例如：

$$
\text{I like machine learning.}
\longleftrightarrow
\text{Ich mag maschinelles Lernen.}
$$

英语到德语任务中，源语言和目标语言共享一个约 37,000 个 token 的 BPE 词表。



**为什么不直接按照完整单词构建词表？**

如果把每个完整单词都作为独立 token，会遇到两个问题：

1. 词表可能非常庞大；
2. 很多低频词和新词无法被词表覆盖。

BPE 和 WordPiece 会将单词拆成更小的子词单元。

例如，一个低频单词可以被拆成：

$$
\text{unbelievable}
\rightarrow
\text{un}+\text{believ}+\text{able}
$$

这样，即使完整单词没有在训练集中频繁出现，模型仍然可以利用已经学过的子词表示它。

这里的具体拆分只是帮助理解的示例，并不是论文给出的实际分词结果。



**为什么英语和德语可以共享词表？**

论文在英语到德语任务中使用共享的源语言—目标语言词表。

共享词表意味着：

> Encoder 输入 token 与 Decoder 输入、输出 token 来自同一个 token 集合。

这也为前面介绍的 Weight Tying 提供了条件，因为 Encoder Embedding、Decoder Embedding 和输出投影矩阵需要具有兼容的词表维度。

共享词表还可以让两种语言中相同或相近的符号共享表示，例如：

- 数字；
- 标点；
- 专有名词；
- 拼写相近的子词。



**Transformer 如何组织一个 Batch？**

论文没有规定每个 Batch 固定包含多少个句子，而是按照 token 数量组织 Batch。

每个 Batch 大约包含：

$$
25000
$$

个源语言 token，以及：

$$
25000
$$

个目标语言 token。

也就是：

$$
\text{每个 Batch}
\approx
25000\text{ 个 source tokens}
+
25000\text{ 个 target tokens}
$$



**为什么使用 token 数量，而不是固定句子数量？**

不同句子的长度可能差异很大。

假设两个 Batch 都包含 64 个句子：

- Batch A 中每个句子平均只有 10 个 token；
- Batch B 中每个句子平均有 100 个 token。

那么二者实际计算量相差约 10 倍。

如果固定 token 总数，则：

- 句子较短时，一个 Batch 可以容纳更多句子；
- 句子较长时，一个 Batch 自动减少句子数量。

这样能使不同 Batch 的计算量相对稳定。

可以直观表示为：

$$
\text{短句}
\Rightarrow
\text{一个 Batch 中句子更多}
$$

$$
\text{长句}
\Rightarrow
\text{一个 Batch 中句子更少}
$$



**为什么还要按照相近长度进行分组？**

同一个 Batch 中，不同长度的句子通常要补齐到相同长度。

例如：

$$
\text{句子 A 长度}=5
$$

$$
\text{句子 B 长度}=20
$$

为了组成矩阵，句子 A 可能需要补充 15 个 `<PAD>`：

$$
[x_1,x_2,x_3,x_4,x_5,
\underbrace{\text{PAD},\ldots,\text{PAD}}_{15\text{ 个}}]
$$

这些 Padding 会占用显存和计算资源，但不会提供有效信息。

因此，论文把长度相近的句子放入同一个 Batch，可以减少无效 Padding。

其核心目的可以概括为：

$$
\boxed{
\text{相近长度分组}
+
\text{固定 token 总量}
\Rightarrow
\text{减少 Padding 并稳定计算量}
}
$$



**5.2 Hardware and Schedule：硬件与训练时长**

论文使用一台配备 8 张 NVIDIA P100 GPU 的机器进行训练。

| 模型 | 单个训练 Step 用时 | 总训练 Step | 总训练时间 |
|---|---:|---:|---:|
| Transformer Base | 约 0.4 秒 | 100,000 | 约 12 小时 |
| Transformer Big | 约 1.0 秒 | 300,000 | 约 3.5 天 |

Base 模型规模较小，每个 Step 的计算速度更快。

Big 模型规模更大，因此：

- 每个 Step 用时更长；
- 训练 Step 更多；
- 总训练时间也更长。

这里的一个 Step 指：

> 模型读取一个 Batch，完成前向传播、计算损失、反向传播并更新一次参数。

因此：

$$
100000\text{ steps}
$$

表示模型共进行了约 100,000 次参数更新。



**为什么 Transformer 的训练效率是重要贡献？**

虽然 Transformer 中 Self-Attention 需要计算 token 两两之间的关系，但同一个序列中的所有位置可以并行处理。

相比 RNN：

$$
h_1
\rightarrow
h_2
\rightarrow
\cdots
\rightarrow
h_n
$$

Transformer 可以通过矩阵运算同时处理多个位置，因此更适合 GPU。

论文强调，Base 模型只需要约 12 小时训练，就能取得很有竞争力的机器翻译结果，这说明 Transformer 不仅效果好，也具有较高的训练效率。



**5.3 Optimizer：Adam 优化器**

论文使用 Adam 优化器，其超参数为：

$$
\beta_1=0.9
$$

$$
\beta_2=0.98
$$

$$
\epsilon=10^{-9}
$$

Adam 会同时维护梯度的一阶矩估计和二阶矩估计。

设第 $t$ 步的梯度为：

$$
g_t
$$

一阶矩估计为：

$$
m_t
=
\beta_1m_{t-1}
+
(1-\beta_1)g_t
$$

二阶矩估计为：

$$
v_t
=
\beta_2v_{t-1}
+
(1-\beta_2)g_t^2
$$

可以直观理解为：

- $m_t$ 记录近期梯度的大致方向；
- $v_t$ 记录近期梯度平方的大小；
- Adam 根据不同参数的梯度尺度，自适应调整更新幅度。

论文在这一节只给出了 Adam 的超参数，没有重新推导 Adam 的完整算法。上述公式用于帮助理解这些参数的含义。



**三个 Adam 参数分别是什么意思？**

| 参数 | 论文设置 | 直观作用 |
|---|---:|---|
| $\beta_1$ | $0.9$ | 控制一阶矩，即梯度方向的平滑程度 |
| $\beta_2$ | $0.98$ | 控制二阶矩，即梯度平方的平滑程度 |
| $\epsilon$ | $10^{-9}$ | 防止分母过小或除以零，提高数值稳定性 |

这里的 $\beta_2=0.98$ 与 Adam 常见默认值 $0.999$ 不同，这是原始 Transformer 采用的特定设置。



**Transformer 的学习率公式**

论文没有使用恒定学习率，而是让学习率随着训练 Step 动态变化：

$$
\operatorname{lrate}
=
d_{\text{model}}^{-1/2}
\cdot
\min
\left(
\operatorname{step\_num}^{-1/2},
\operatorname{step\_num}
\cdot
\operatorname{warmup\_steps}^{-3/2}
\right)
$$

其中：

- $d_{\text{model}}$：模型表示维度；
- $\operatorname{step\_num}$：当前训练步数；
- $\operatorname{warmup\_steps}$：预热步数；
- $\min(\cdot)$：选择两个表达式中较小的一个。

原始 Transformer 使用：

$$
\operatorname{warmup\_steps}=4000
$$

Base 模型中：

$$
d_{\text{model}}=512
$$



**为什么公式中有两个不同的学习率变化趋势？**

公式内部有两项：

$$
\operatorname{step\_num}^{-1/2}
$$

和：

$$
\operatorname{step\_num}
\cdot
\operatorname{warmup\_steps}^{-3/2}
$$

第一项随着 Step 增大而减小：

$$
\operatorname{step\_num}^{-1/2}
=
\frac{1}{\sqrt{\operatorname{step\_num}}}
$$

第二项与当前 Step 成正比，因此随着 Step 线性增大。

取二者的较小值，就会形成：

1. 前期线性上升；
2. 到达峰值；
3. 后期按照平方根倒数逐渐下降。



**Warmup 阶段：学习率线性增加**

设：

$$
s=\operatorname{step\_num}
$$

$$
w=\operatorname{warmup\_steps}
$$

当：

$$
s\leq w
$$

时，较小的一项是：

$$
s\cdot w^{-3/2}
$$

所以学习率为：

$$
\operatorname{lrate}(s)
=
d_{\text{model}}^{-1/2}
\cdot
s
\cdot
w^{-3/2}
$$

其中只有 $s$ 在变化，因此学习率随着训练步数线性增加：

$$
\operatorname{lrate}(s)\propto s
$$

这就是 Warmup，即学习率预热。



**为什么训练开始时不直接使用较大学习率？**

训练刚开始时：

- 参数仍接近随机初始化；
- Attention 权重尚不稳定；
- 梯度方向可能波动较大；
- 直接使用较大学习率容易造成参数更新过猛。

Warmup 让学习率从较小值逐渐上升，相当于让模型先缓慢适应训练数据。

可以类比开车：

> 刚启动时先缓慢加速，而不是一开始就踩到底。

这是对 Warmup 作用的直观理解。论文明确给出了学习率先线性上升的策略，但没有在本节提供严格理论证明。



**Warmup 之后：学习率逐渐衰减**

当：

$$
s\geq w
$$

时，较小的一项变为：

$$
s^{-1/2}
$$

因此：

$$
\operatorname{lrate}(s)
=
d_{\text{model}}^{-1/2}
\cdot
s^{-1/2}
$$

也就是：

$$
\operatorname{lrate}(s)
\propto
\frac{1}{\sqrt{s}}
$$

随着训练继续进行，学习率逐渐减小。

后期减小学习率，可以让模型在已经找到较好参数区域后，使用更小步长进行精细调整，降低在最优区域附近来回震荡的风险。



**学习率在哪一步达到最大值？**

两部分在：

$$
s=w
$$

时相等，因此学习率在 Warmup 结束时达到峰值。

将：

$$
s=w
$$

代入公式：

$$
\operatorname{lrate}_{\max}
=
d_{\text{model}}^{-1/2}
\cdot
w^{-1/2}
$$

也就是：

$$
\operatorname{lrate}_{\max}
=
\frac{1}
{\sqrt{d_{\text{model}}\cdot w}}
$$

对于 Base 模型：

$$
d_{\text{model}}=512
$$

$$
w=4000
$$

因此峰值学习率约为：

$$
\operatorname{lrate}_{\max}
\approx
6.99\times10^{-4}
$$

这个数值由论文公式计算得到。



**学习率变化过程**

整个过程可以概括为：

$$
\boxed{
\text{较小学习率启动}
\rightarrow
\text{前 4000 步线性升高}
\rightarrow
\text{第 4000 步达到峰值}
\rightarrow
\text{之后按 }s^{-1/2}\text{ 衰减}
}
$$

其分段形式为：

$$
\operatorname{lrate}(s)
=
\begin{cases}
d_{\text{model}}^{-1/2}
s
w^{-3/2},
&
s\leq w
\\[6pt]
d_{\text{model}}^{-1/2}
s^{-1/2},
&
s\geq w
\end{cases}
$$

其中：

$$
w=4000
$$



**为什么还要乘以 $d_{\text{model}}^{-1/2}$？**

学习率公式中包含：

$$
d_{\text{model}}^{-1/2}
=
\frac{1}{\sqrt{d_{\text{model}}}}
$$

它使学习率根据模型表示维度进行缩放：

- $d_{\text{model}}$ 越大，整体学习率尺度越小；
- $d_{\text{model}}$ 越小，整体学习率尺度越大。

论文给出了这一公式，但没有在本节详细推导其理论来源。

因此，最稳妥的理解是：

> 该项负责根据模型宽度统一调整学习率的整体数值尺度。



**5.4 Regularization：正则化**

正则化的目标是减少过拟合，提高模型在未见数据上的表现。

论文在本节明确介绍了：

- Dropout；
- Label Smoothing。

原文写道使用“三类正则化”，但本节实际以标题形式展开的是 Residual Dropout 和 Label Smoothing；其中 Dropout 又被应用在两类位置。这里按照原文明确描述的内容整理，不额外补充一个未被说明的第三项。



**第一类：Residual Dropout**

论文在每个子层的输出上应用 Dropout，然后再与子层输入进行残差相加并完成 LayerNorm。

原来的结构可以写为：

$$
\operatorname{LayerNorm}
\left(
x+\operatorname{Sublayer}(x)
\right)
$$

加入 Dropout 后，可以表示为：

$$
\operatorname{LayerNorm}
\left(
x+
\operatorname{Dropout}
\left(
\operatorname{Sublayer}(x)
\right)
\right)
$$

也就是：

$$
\text{子层输出}
\rightarrow
\text{Dropout}
\rightarrow
\text{残差相加}
\rightarrow
\text{LayerNorm}
$$



**Dropout 在做什么？**

训练过程中，Dropout 会随机把部分神经元输出设置为 0。

假设：

$$
p_{\text{drop}}=0.1
$$

表示训练时大约有 10% 的输出被随机丢弃。

这样可以防止模型过度依赖少量特征或固定路径。

可以通俗理解为：

> 训练时随机让部分信息通道暂时失效，迫使模型学习更加分散、更加稳健的表示。

原始 Transformer Base 使用：

$$
P_{\text{drop}}=0.1
$$



**Dropout 被应用在哪些位置？**

论文明确提到了两类位置。

第一类是每个子层的输出，包括：

- Multi-Head Attention 输出；
- FFN 输出。

在执行残差连接之前使用 Dropout：

$$
x+
\operatorname{Dropout}
\left(
\operatorname{Sublayer}(x)
\right)
$$

第二类是 Embedding 与 Positional Encoding 相加后的结果。

原本输入为：

$$
H^{(0)}
=
\sqrt{d_{\text{model}}}E[X]+PE
$$

加入 Dropout 后可以表示为：

$$
H^{(0)}
=
\operatorname{Dropout}
\left(
\sqrt{d_{\text{model}}}E[X]+PE
\right)
$$

Encoder 和 Decoder 的输入端都使用这一处理。



**为什么称为 Residual Dropout？**

因为 Dropout 被施加在子层输出上，并且位于残差相加之前：

$$
x+
\operatorname{Dropout}
\left(
\operatorname{Sublayer}(x)
\right)
$$

即使子层中的部分输出被丢弃，原始输入 $x$ 仍然可以通过残差路径保留下来。

因此，模型不会因为 Dropout 而完全失去该位置的原始信息。



**第二类：Label Smoothing**

普通分类训练通常使用 One-Hot 标签。

假设词表中有 5 个候选 token，正确答案是第 3 个，则目标分布为：

$$
[0,0,1,0,0]
$$

它要求模型对正确 token 给出概率 1，对其余 token 给出概率 0。

Label Smoothing 不再使用如此绝对的目标分布，而是为其他类别保留少量概率。

论文使用：

$$
\epsilon_{\text{ls}}=0.1
$$

一种常见的 Label Smoothing 写法是：

$$
q_{\text{correct}}
=
1-\epsilon_{\text{ls}}
$$

$$
q_{\text{other}}
=
\frac{\epsilon_{\text{ls}}}{K-1}
$$

其中 $K$ 是类别数量。

例如，当：

$$
K=5
$$

$$
\epsilon_{\text{ls}}=0.1
$$

时，目标分布可以写成：

$$
[0.025,0.025,0.9,0.025,0.025]
$$

需要注意：这是帮助理解 Label Smoothing 的一种常见实现形式，论文在本节只给出了 $\epsilon_{\text{ls}}=0.1$，没有展开具体分配公式。



**Label Smoothing 为什么有用？**

如果使用严格 One-Hot 标签，模型会被鼓励将正确 token 的概率不断推向 1：

$$
p_{\text{correct}}\rightarrow1
$$

这可能使模型变得过度自信。

Label Smoothing 则告诉模型：

> 正确答案最重要，但不需要对其他所有候选都保持绝对的零概率。

这样可以：

- 降低模型过度自信；
- 改善泛化能力；
- 缓解过拟合；
- 让输出概率分布更加平滑。



**为什么 Label Smoothing 会损害 Perplexity？**

论文指出，Label Smoothing：

- 会使 Perplexity 变差；
- 但会提高 Accuracy 和 BLEU。

这是因为模型被训练成不再对正确答案给出极端高概率。

例如，原本模型可能输出：

$$
p_{\text{correct}}=0.99
$$

使用 Label Smoothing 后，模型可能输出：

$$
p_{\text{correct}}=0.90
$$

如果 Perplexity 按真实正确 token 的概率计算，那么较低的正确类别概率可能使 Perplexity 数值变差。

但模型不再过度自信，通常会获得更好的泛化能力，因此在最终翻译质量指标 BLEU 上反而有所提升。

所以这里体现了一个重要区别：

> 更低的训练概率损失或 Perplexity，不一定自动等价于更好的最终翻译质量。



**训练方案汇总**

| 训练环节 | 论文设置 |
|---|---|
| 英德数据规模 | 约 450 万句对 |
| 英法数据规模 | 约 3,600 万句对 |
| 英德分词 | BPE，共享约 37,000 词表 |
| 英法分词 | WordPiece，约 32,000 词表 |
| Batch 大小 | 约 25,000 个源 token 和 25,000 个目标 token |
| 硬件 | 单机 8 张 NVIDIA P100 |
| Base 训练 | 100,000 Steps，约 12 小时 |
| Big 训练 | 300,000 Steps，约 3.5 天 |
| 优化器 | Adam |
| $\beta_1$ | $0.9$ |
| $\beta_2$ | $0.98$ |
| $\epsilon$ | $10^{-9}$ |
| Warmup Steps | $4000$ |
| Base Dropout | $0.1$ |
| Label Smoothing | $0.1$ |



**本节逻辑主线**

$$
\boxed{
\begin{aligned}
&\text{使用大规模平行语料和子词分词}\\
&\Downarrow\\
&\text{按照相近长度和 token 总数组织 Batch}\\
&\Downarrow\\
&\text{使用 8 张 P100 并行训练}\\
&\Downarrow\\
&\text{采用 Adam 更新模型参数}\\
&\Downarrow\\
&\text{前 4000 步线性 Warmup}\\
&\Downarrow\\
&\text{之后按平方根倒数衰减学习率}\\
&\Downarrow\\
&\text{通过 Dropout 和 Label Smoothing 抑制过拟合}
\end{aligned}
}
$$

**一句话总结**

原始 Transformer 使用大规模机器翻译语料和基于 token 数量的动态 Batch，在 8 张 P100 GPU 上通过 Adam 训练；学习率先经过 4000 步线性 Warmup，再按照训练步数的平方根倒数衰减，同时使用 Dropout 和 Label Smoothing 提高训练稳定性与泛化能力。

## 6. Results

**本节核心问题：Transformer 的实际效果如何？哪些结构设计真正有效？模型能否迁移到机器翻译以外的任务？**

论文从三个方面验证 Transformer：

1. 在英语—德语和英语—法语机器翻译任务上，与已有模型比较；
2. 通过消融实验分析模型深度、宽度、注意力头数、Dropout、Label Smoothing 和位置编码；
3. 将 Transformer 应用于英语成分句法分析，检验其任务泛化能力。



**6.1 Machine Translation：机器翻译结果**

论文使用 WMT 2014 机器翻译基准，评估以下两个任务：

$$
\text{English}\rightarrow\text{German}
$$

$$
\text{English}\rightarrow\text{French}
$$

主要评价指标为 BLEU。



**BLEU 是什么？**

BLEU 是机器翻译中常用的自动评价指标，主要比较：

> 模型生成的译文与参考译文之间，有多少相同的局部词组。

它不仅比较单个词，还会比较连续的多个词，即不同长度的 n-gram。

例如，参考译文是：

> 我喜欢机器学习

模型生成：

> 我喜欢机器学习

二者高度一致，BLEU 较高。

如果模型生成：

> 机器我学习喜欢

虽然包含相似的词，但词序和局部词组不匹配，因此 BLEU 会明显降低。

BLEU 通常还包含长度惩罚，防止模型通过生成特别短的句子获得虚假的高匹配率。

这里是帮助理解 BLEU 的补充说明；论文在本节直接报告 BLEU，没有展开其具体公式。



**Transformer 的主要翻译结果**

| 模型 | 英语→德语 BLEU | 英语→法语 BLEU |
|---|---:|---:|
| Transformer Base | 27.3 | 38.1 |
| Transformer Big | **28.4** | **41.8** |

在英语到德语任务中，Transformer Big 获得：

$$
\operatorname{BLEU}=28.4
$$

相比此前最佳结果，包括集成模型，提高超过：

$$
2.0\ \operatorname{BLEU}
$$

论文中的典型对比包括：

| 模型 | 英语→德语 BLEU |
|---|---:|
| GNMT + RL | 24.6 |
| ConvS2S | 25.16 |
| MoE | 26.03 |
| GNMT + RL Ensemble | 26.30 |
| ConvS2S Ensemble | 26.36 |
| Transformer Base | 27.3 |
| Transformer Big | **28.4** |

这说明原始 Transformer 的单模型结果已经超过了部分此前的集成模型。



**英语到法语结果中的文本差异**

论文摘要和 Table 2 报告 Transformer Big 在英语到法语任务上取得：

$$
41.8\ \operatorname{BLEU}
$$

但上传版本第 6.1 节的正文中出现了：

$$
41.0\ \operatorname{BLEU}
$$

二者存在不一致。

由于摘要、Table 2 以及论文广泛引用的最终结果均为 41.8，本笔记后续按照：

$$
41.8\ \operatorname{BLEU}
$$

记录，但保留这一原文差异，避免把来源内部的不一致悄悄忽略。



**效果提升是否依赖更高的训练成本？**

论文不仅比较翻译效果，也比较估算的训练 FLOPs。

Table 2 中报告：

| 模型 | 训练计算量 |
|---|---:|
| Transformer Base | 约 $3.3\times10^{18}$ FLOPs |
| Transformer Big | 约 $2.3\times10^{19}$ FLOPs |

一些此前模型的训练开销达到：

$$
10^{20}\sim10^{21}\ \text{FLOPs}
$$

例如，部分集成模型虽然翻译效果较强，但训练成本明显高于 Transformer。

因此，论文的核心结论不是简单的“模型更大所以效果更好”，而是：

> Transformer 在取得更高 BLEU 的同时，训练计算成本仍显著低于许多此前的强模型和集成模型。



**论文如何估算训练成本？**

论文使用以下信息估算训练 FLOPs：

- 训练时间；
- 使用的 GPU 数量；
- 单张 GPU 持续单精度浮点计算能力。

可以直观写成：

$$
\text{Training FLOPs}
\approx
\text{训练秒数}
\times
\text{GPU 数量}
\times
\text{单张 GPU 的 FLOPs/s}
$$

这是一种估算值，并不是对每一次实际矩阵运算进行精确统计。



**Checkpoint Averaging：检查点平均**

论文在最终测试时没有直接使用最后一个训练检查点，而是对训练末期的多个 Checkpoint 进行参数平均。

Base 模型平均最后 5 个 Checkpoint：

$$
\theta_{\text{avg}}
=
\frac{1}{5}
\sum_{i=1}^{5}\theta_i
$$

这些 Checkpoint 之间间隔约 10 分钟。

Big 模型平均最后 20 个 Checkpoint：

$$
\theta_{\text{avg}}
=
\frac{1}{20}
\sum_{i=1}^{20}\theta_i
$$

其中：

- $\theta_i$ 表示第 $i$ 个 Checkpoint 中的模型参数；
- $\theta_{\text{avg}}$ 表示平均后的模型参数。



**为什么要进行 Checkpoint Averaging？**

训练后期，模型参数通常会在一个较优区域附近轻微震荡。

例如，连续几个 Checkpoint 的参数可能分别位于：

$$
\theta_1,\theta_2,\theta_3,\theta_4,\theta_5
$$

单独使用某一个 Checkpoint，可能受到某个 Batch 或某次更新造成的偶然波动影响。

对多个 Checkpoint 取平均，可以获得较平滑、较稳定的参数：

$$
\theta_{\text{avg}}
=
\frac{\theta_1+\theta_2+\theta_3+\theta_4+\theta_5}{5}
$$

需要注意：

> Checkpoint Averaging 不是把 5 个模型分别推理后再平均输出。

它是在推理之前，把同一次训练过程中多个时间点的参数进行平均，最终仍然只得到一个模型。

因此，它与独立训练多个模型的 Ensemble 不完全相同。



**Beam Search：束搜索**

推理时，论文使用：

$$
\operatorname{beam\ size}=4
$$

普通贪心解码在每一步只保留当前概率最高的 token：

$$
y_t
=
\arg\max_y
p(y\mid y_{<t},X)
$$

但某一步局部概率最高的选择，并不一定能组成整体概率最高的完整句子。

Beam Search 会同时保留若干条候选序列。

当：

$$
\operatorname{beam\ size}=4
$$

时，每一步保留当前得分最高的 4 条候选路径。

例如，在某一步模型可能保留：

1. “我喜欢”；
2. “我热爱”；
3. “本人喜欢”；
4. “我很喜欢”。

下一步再分别扩展这些候选，最后选择完整句子得分较高的结果。



**Length Penalty：长度惩罚**

论文使用长度惩罚参数：

$$
\alpha=0.6
$$

生成序列的概率是多个条件概率的乘积：

$$
p(Y\mid X)
=
\prod_{t=1}^{m}
p(y_t\mid y_{<t},X)
$$

取对数后：

$$
\log p(Y\mid X)
=
\sum_{t=1}^{m}
\log p(y_t\mid y_{<t},X)
$$

由于每个概率通常小于 1，序列越长，累积对数概率通常越小，因此模型容易偏向较短的输出。

Length Penalty 用来修正这种长度偏好，使较长但合理的翻译不会因为 token 数量多而受到过度惩罚。

论文采用：

$$
\alpha=0.6
$$

这些解码超参数通过开发集实验确定。



**最大输出长度**

论文将最大输出长度设置为：

$$
\text{Maximum Output Length}
=
\text{Input Length}+50
$$

例如，输入包含 30 个 token，则模型最多生成：

$$
30+50=80
$$

个 token。

如果提前生成 `<EOS>`，则会提前结束，不需要强制生成到最大长度。



**翻译实验的核心结论**

| 结论 | 含义 |
|---|---|
| Base 模型已经很强 | 单个 Base Transformer 超过多个此前模型和集成结果 |
| Big 模型进一步提升 | 增大模型容量可以继续改善 BLEU |
| 训练成本较低 | 相比部分传统强模型，需要的训练 FLOPs 更少 |
| 并行结构有效 | 不使用 RNN 和 CNN 仍能取得更强翻译质量 |
| 单模型竞争力强 | Transformer Big 超过此前多个 Ensemble |



**6.2 Model Variations：模型变体与消融实验**

论文在 WMT 2014 英语到德语任务的开发集 newstest2013 上进行消融实验。

所有结果均在开发集上测量，主要报告：

- Perplexity，PPL；
- BLEU；
- 参数量。

消融实验的目的不是追求最佳测试结果，而是回答：

> Transformer 中哪些结构和超参数真正重要？



**Base 模型的默认配置**

| 参数 | Base 配置 |
|---|---:|
| Encoder / Decoder 层数 $N$ | 6 |
| 模型维度 $d_{\text{model}}$ | 512 |
| FFN 隐藏维度 $d_{\text{ff}}$ | 2048 |
| 注意力头数 $h$ | 8 |
| 每头 Key 维度 $d_k$ | 64 |
| 每头 Value 维度 $d_v$ | 64 |
| Dropout $P_{\text{drop}}$ | 0.1 |
| Label Smoothing $\epsilon_{\text{ls}}$ | 0.1 |
| 训练步数 | 100K |
| 开发集 PPL | 4.92 |
| 开发集 BLEU | 25.8 |
| 参数量 | 约 6500 万 |

原始 Transformer Base 可以概括为：

$$
N=6,\qquad
d_{\text{model}}=512,\qquad
d_{\text{ff}}=2048,\qquad
h=8
$$

并且：

$$
d_k=d_v=\frac{512}{8}=64
$$



**Perplexity：困惑度是什么？**

Perplexity 常用于衡量语言模型对正确 token 的预测能力。

可以将其写成平均交叉熵的指数：

$$
\operatorname{PPL}
=
\exp
\left(
-\frac{1}{T}
\sum_{t=1}^{T}
\log p(y_t\mid y_{<t},X)
\right)
$$

直观上：

- PPL 越低，模型给正确 token 分配的平均概率越高；
- PPL 越高，模型对正确输出越不确定。

例如：

$$
\operatorname{PPL}=5
$$

可以非常粗略地理解成，模型平均像是在约 5 个候选之间犹豫。

但 PPL 与 BLEU 衡量的内容不同：

- PPL 评价 token 概率分布；
- BLEU 评价最终生成译文与参考译文的匹配程度。

因此，PPL 更低不一定保证 BLEU 一定更高，尤其是在使用 Label Smoothing 时。



**消融一：注意力头数是否越多越好？**

论文在总计算量大致保持不变的情况下改变注意力头数。

| 注意力头数 $h$ | 每头维度 $d_k=d_v$ | PPL | BLEU |
|---:|---:|---:|---:|
| 1 | 512 | 5.29 | 24.9 |
| 4 | 128 | 5.00 | 25.5 |
| 8 | 64 | 4.92 | 25.8 |
| 16 | 32 | 4.91 | 25.8 |
| 32 | 16 | 5.01 | 25.4 |

单头注意力的 BLEU 为：

$$
24.9
$$

Base 模型的 BLEU 为：

$$
25.8
$$

两者相差：

$$
25.8-24.9=0.9
$$

说明将表示拆分到多个注意力头中确实有效。

但头数增加到 32 时，性能又下降到：

$$
25.4
$$

因此，结论不是“注意力头越多越好”，而是：

> 多头注意力需要在头数与每个头的表示维度之间取得平衡。

当总维度固定时：

$$
d_k=\frac{d_{\text{model}}}{h}
$$

头数越多，每个头的维度越小。

例如：

$$
h=32
\Rightarrow
d_k=16
$$

每个头的表达空间过小，可能限制其建模能力。



**为什么单头 Attention 表现更差？**

单头 Attention 只能在一个表示空间中完成信息匹配和加权汇总。

多头 Attention 则可以在不同投影子空间中并行计算：

$$
\operatorname{head}_i
=
\operatorname{Attention}
\left(
QW_i^Q,KW_i^K,VW_i^V
\right)
$$

不同头可以学习不同类型的关系，例如：

- 语法关系；
- 语义关系；
- 指代关系；
- 长距离依赖。

论文的实验说明，将全部维度放入一个注意力头并不是最优设计。



**消融二：Key 的维度是否重要？**

论文降低每个注意力头的 Key 维度 $d_k$，发现模型效果下降。

| $d_k$ | PPL | BLEU | 参数量 |
|---:|---:|---:|---:|
| 16 | 5.16 | 25.1 | 约 5800 万 |
| 32 | 5.01 | 25.4 | 约 6000 万 |
| Base：64 | 4.92 | 25.8 | 约 6500 万 |

Attention 的匹配分数来自：

$$
QK^{\mathsf T}
$$

其中，Query 与 Key 负责判断：

> 当前位置与其他位置的匹配程度有多高？

当 $d_k$ 过小时，Query 和 Key 中可以保存的匹配信息变少，可能难以充分表达复杂的相似性或兼容性关系。

论文据此提出：

> 判断 Query 与 Key 的兼容性并不是一个简单问题，未来可能需要比普通点积更复杂的匹配函数。

需要注意，这只是作者根据消融结果提出的推测，并不是严格理论结论。



**消融三：网络深度是否重要？**

论文改变 Encoder 和 Decoder 堆叠层数 $N$。

| 层数 $N$ | PPL | BLEU | 参数量 |
|---:|---:|---:|---:|
| 2 | 6.11 | 23.7 | 约 3600 万 |
| 4 | 5.19 | 25.3 | 约 5000 万 |
| 6 | 4.92 | 25.8 | 约 6500 万 |
| 8 | 4.88 | 25.5 | 约 8000 万 |

从 2 层增加到 6 层时，BLEU 明显提高：

$$
23.7\rightarrow25.8
$$

说明足够的网络深度很重要。

但增加到 8 层后，开发集 BLEU 没有继续提升，而是变为：

$$
25.5
$$

因此，在当前训练设置下，更深不一定自动带来更好结果。

可能原因包括：

- 更深模型需要不同的优化设置；
- 当前训练步数可能不足；
- 模型容量增大后更容易过拟合；
- Post-Norm 深层结构的优化难度可能增加。

最后一点属于后来的理解，不是论文在本节明确给出的解释。论文在这里主要报告实验现象。



**消融四：模型维度是否重要？**

论文改变模型表示维度 $d_{\text{model}}$。

| $d_{\text{model}}$ | 每头维度 | PPL | BLEU | 参数量 |
|---:|---:|---:|---:|---:|
| 256 | 32 | 5.75 | 24.5 | 约 2800 万 |
| 512 | 64 | 4.92 | 25.8 | 约 6500 万 |
| 1024 | 128 | 4.66 | 26.0 | 约 1.68 亿 |

增大模型维度后，性能总体提升：

$$
d_{\text{model}}:256\rightarrow512\rightarrow1024
$$

对应 BLEU：

$$
24.5\rightarrow25.8\rightarrow26.0
$$

模型维度越大，每个 token 可以保存的特征越多，但参数量和计算成本也会明显增加。

例如：

$$
d_{\text{model}}=1024
$$

时参数量达到约：

$$
168\text{M}
$$

因此，这是效果与计算成本之间的权衡。



**消融五：FFN 隐藏维度是否重要？**

Base 模型使用：

$$
d_{\text{ff}}=2048
$$

论文将其改为较小或较大的值。

| $d_{\text{ff}}$ | PPL | BLEU | 参数量 |
|---:|---:|---:|---:|
| 1024 | 5.12 | 25.4 | 约 5300 万 |
| 2048 | 4.92 | 25.8 | 约 6500 万 |
| 4096 | 4.75 | 26.2 | 约 9000 万 |

当 FFN 隐藏维度从：

$$
1024\rightarrow2048\rightarrow4096
$$

增加时，BLEU 从：

$$
25.4\rightarrow25.8\rightarrow26.2
$$

说明 FFN 并不是简单的辅助模块，它的容量会显著影响模型表现。

FFN 的维度变化为：

$$
d_{\text{model}}
\rightarrow
d_{\text{ff}}
\rightarrow
d_{\text{model}}
$$

更大的 $d_{\text{ff}}$ 为每个 token 提供了更大的中间特征空间，可以进行更丰富的非线性特征变换。



**消融六：Dropout 是否重要？**

Base 模型使用：

$$
P_{\text{drop}}=0.1
$$

论文还测试了：

$$
P_{\text{drop}}=0
$$

和：

$$
P_{\text{drop}}=0.2
$$

| Dropout | PPL | BLEU |
|---:|---:|---:|
| 0.0 | 5.77 | 24.6 |
| 0.1 | 4.92 | 25.8 |
| 0.2 | 4.95 | 25.5 |

完全移除 Dropout 后，BLEU 明显下降：

$$
25.8\rightarrow24.6
$$

说明模型出现了更严重的过拟合。

但 Dropout 提高到 0.2 时，效果也略低于 0.1，说明正则化过强也可能使模型难以充分学习。

因此，Dropout 的效果通常表现为：

$$
\text{过小}
\Rightarrow
\text{容易过拟合}
$$

$$
\text{过大}
\Rightarrow
\text{可能欠拟合}
$$

$$
\text{适中}
\Rightarrow
\text{泛化表现较好}
$$



**消融七：Label Smoothing 对 PPL 和 BLEU 的影响**

Base 模型使用：

$$
\epsilon_{\text{ls}}=0.1
$$

论文还比较了不使用和使用更强 Label Smoothing 的情况。

| $\epsilon_{\text{ls}}$ | PPL | BLEU |
|---:|---:|---:|
| 0.0 | 4.67 | 25.3 |
| 0.1 | 4.92 | 25.8 |
| 0.2 | 5.47 | 25.7 |

不使用 Label Smoothing 时，PPL 反而更低：

$$
4.67<4.92
$$

但 BLEU 更差：

$$
25.3<25.8
$$

提高到：

$$
\epsilon_{\text{ls}}=0.2
$$

后，PPL 进一步变差到：

$$
5.47
$$

但 BLEU 仍然达到：

$$
25.7
$$

这说明：

> PPL 和最终生成质量并不完全一致。

Label Smoothing 会阻止模型把正确类别概率推得过于接近 1，因此可能使按照正确 token 概率计算的 PPL 变差。

但模型的输出分布更加平滑、不过度自信，最终翻译质量可能反而提高。



**消融八：固定位置编码还是可学习位置编码？**

论文将正弦位置编码替换为可学习位置 Embedding。

| 位置编码 | PPL | BLEU |
|---|---:|---:|
| 正弦、余弦位置编码 | 4.92 | 25.8 |
| 可学习位置 Embedding | 4.92 | 25.7 |

二者表现几乎相同：

$$
25.8\approx25.7
$$

这说明 Transformer 的性能并不依赖固定正弦函数本身。

论文最终保留正弦位置编码，主要原因是作者认为它可能更容易外推到训练阶段未见过的更长序列。

这里的实验只证明二者在当前任务上的表现接近，没有直接证明正弦位置编码的长度外推能力一定更强。



**Transformer Big 的配置**

论文最终的 Big 模型配置为：

| 参数 | Transformer Big |
|---|---:|
| 层数 $N$ | 6 |
| $d_{\text{model}}$ | 1024 |
| $d_{\text{ff}}$ | 4096 |
| 注意力头数 $h$ | 16 |
| Dropout | 0.3 |
| 训练步数 | 300K |
| 开发集 PPL | 4.33 |
| 开发集 BLEU | 26.4 |
| 参数量 | 约 2.13 亿 |

与 Base 相比，Big 模型主要扩大了：

- 模型维度；
- FFN 隐藏维度；
- 注意力头数；
- 参数量；
- 训练步数。

其参数量从约：

$$
65\text{M}
$$

增加到：

$$
213\text{M}
$$

开发集 BLEU 从：

$$
25.8
$$

提高到：

$$
26.4
$$



**消融实验得到的整体结论**

| 变量 | 论文观察 |
|---|---|
| 注意力头数 | 单头较差，但头数也不是越多越好 |
| Key 维度 | $d_k$ 太小会损害匹配能力 |
| 网络深度 | 过浅明显较差，继续加深不一定持续提升 |
| 模型宽度 | 更大的 $d_{\text{model}}$ 通常效果更好 |
| FFN 宽度 | 更大的 $d_{\text{ff}}$ 可以明显改善性能 |
| Dropout | 对避免过拟合非常重要 |
| Label Smoothing | 可能使 PPL 变差，但改善 BLEU |
| 位置编码 | 固定与可学习位置编码效果接近 |
| 模型规模 | 更大模型总体更好，但计算和参数成本也更高 |



**6.3 English Constituency Parsing：英语成分句法分析**

为了验证 Transformer 是否只能用于机器翻译，论文还将其应用于英语成分句法分析。



**什么是成分句法分析？**

成分句法分析的目标是把一个句子转换成树状结构。

例如：

> The cat sleeps.

可以被分析为：

$$
\text{S}
\rightarrow
\text{NP}+\text{VP}
$$

其中：

- S：完整句子；
- NP：名词短语；
- VP：动词短语。

进一步可以展开为：

$$
\text{NP}
\rightarrow
\text{Det}+\text{Noun}
$$

$$
\text{VP}
\rightarrow
\text{Verb}
$$

因此，模型输入是一个普通句子，输出则是表示句法结构的序列。

该任务与机器翻译相比有几个特殊困难：

- 输出必须满足较强的树结构约束；
- 输出序列通常显著长于输入句子；
- 小数据训练条件下，传统 Seq2Seq RNN 表现不够理想。



**句法分析实验使用的模型**

论文使用一个 4 层 Transformer：

$$
N=4
$$

模型维度为：

$$
d_{\text{model}}=1024
$$

这与机器翻译 Base 模型的配置不同。



**两种训练设置**

论文测试了两种数据规模。

| 设置 | 训练数据 | 词表大小 |
|---|---:|---:|
| WSJ Only | 约 40,000 个训练句子 | 16,000 |
| Semi-Supervised | 约 1,700 万个句子 | 32,000 |

WSJ Only 使用 Penn Treebank 中 Wall Street Journal 部分的数据。

Semi-Supervised 设置则加入规模更大的高置信度自动标注语料和 BerkeleyParser 语料。



**句法分析的解码设置**

由于句法树的线性化输出通常明显长于原始句子，论文将最大输出长度设置为：

$$
\text{Maximum Output Length}
=
\text{Input Length}+300
$$

同时使用：

$$
\operatorname{beam\ size}=21
$$

长度惩罚参数为：

$$
\alpha=0.3
$$

与机器翻译相比：

- Beam Size 从 4 增加到 21；
- 最大输出长度从输入长度加 50，增加到加 300。

这是因为句法分析输出受到更强的结构约束，并且候选输出通常更长。



**句法分析使用什么指标？**

成分句法分析使用：

$$
F_1
$$

指标。

F1 综合考虑 Precision 和 Recall：

$$
F_1
=
\frac{2PR}{P+R}
$$

其中：

- Precision：模型预测出的句法成分中，有多少是正确的；
- Recall：参考答案中的正确句法成分，有多少被模型找到了。

例如：

$$
P=0.90,\qquad R=0.80
$$

则：

$$
F_1
=
\frac{2\times0.90\times0.80}{0.90+0.80}
\approx0.847
$$

F1 越高，说明预测句法树与参考句法树越接近。



**WSJ Only 结果**

| 模型 | WSJ 23 F1 |
|---|---:|
| Vinyals & Kaiser 等 | 88.3 |
| Petrov 等 | 90.4 |
| Zhu 等 | 90.4 |
| Dyer 等 | 91.7 |
| Transformer，4 层 | **91.3** |

Transformer 在只使用约 40,000 个 WSJ 训练句子的情况下，取得：

$$
F_1=91.3
$$

虽然略低于 Dyer 等人的 91.7，但已经超过 BerkeleyParser 对应的 90.4。

论文特别强调，RNN Seq2Seq 模型此前在小数据设置中没有达到同样水平，而 Transformer 在缺少大量任务特定调优的情况下依然表现较好。



**Semi-Supervised 结果**

| 模型 | WSJ 23 F1 |
|---|---:|
| Zhu 等 | 91.3 |
| Huang & Harper | 91.3 |
| McClosky 等 | 92.1 |
| Vinyals & Kaiser 等 | 92.1 |
| Transformer，4 层 | **92.7** |

在半监督设置中，Transformer 达到：

$$
F_1=92.7
$$

超过了表中此前的半监督结果：

$$
92.1
$$

但论文表格中还有：

| 方法类别 | 模型 | F1 |
|---|---|---:|
| Multi-task | Luong 等 | 93.0 |
| Generative | Dyer 等 | 93.3 |

因此，Transformer 并没有超过表格中的所有方法。

论文的准确表述是：

> Transformer 的结果超过了多数此前报告的模型，但仍低于 Recurrent Neural Network Grammar 等最强结果。



**为什么这个实验很重要？**

机器翻译与句法分析的输出性质差异较大。

机器翻译：

$$
\text{自然语言序列}
\rightarrow
\text{另一种自然语言序列}
$$

句法分析：

$$
\text{自然语言序列}
\rightarrow
\text{结构化句法表示}
$$

如果 Transformer 只在翻译上有效，可能说明它只是针对机器翻译任务进行了特殊优化。

句法分析实验说明：

- Transformer 可以处理具有强结构约束的输出；
- 可以处理比输入明显更长的输出；
- 在小数据设置中也具有较强竞争力；
- 模型架构具有跨任务泛化能力。



**Results 部分的总体结论**

$$
\boxed{
\begin{aligned}
&\text{Transformer 在英德翻译上取得 }28.4\text{ BLEU}\\
&\Downarrow\\
&\text{超过此前单模型和多个集成模型}\\
&\Downarrow\\
&\text{英法翻译达到表中报告的 }41.8\text{ BLEU}\\
&\Downarrow\\
&\text{训练成本低于许多此前强模型}\\
&\Downarrow\\
&\text{消融实验验证多头、模型宽度、FFN 和正则化的重要性}\\
&\Downarrow\\
&\text{固定与可学习位置编码表现接近}\\
&\Downarrow\\
&\text{在英语成分句法分析中同样取得有竞争力的结果}\\
&\Downarrow\\
&\text{说明 Transformer 不只适用于机器翻译}
\end{aligned}
}
$$

**一句话总结**

Transformer 不仅在英德和英法机器翻译任务上以较低训练成本取得当时领先结果，消融实验还表明多头注意力、足够的模型宽度、较大的 FFN 和适当正则化都很重要；其在英语成分句法分析中的表现进一步证明，该架构能够泛化到具有强结构约束的其他序列转换任务。
