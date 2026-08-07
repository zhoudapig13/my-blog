---
title: "『LLM学习笔记1』损失函数、优化器和梯度"
category: "internship"
tags:
  - "LLM"
date: "2026-07-30"
summary: "系统梳理深度学习中的损失函数、优化器与梯度机制，涵盖 MSE、交叉熵、KL 散度、Softmax、SGD、AdamW、梯度累积与裁剪，并结合代码和常见问题理解训练稳定性。"
pdf: ""
pdfTitle: ""
---

## 一、损失函数——模型究竟在优化什么

**1. MSE：回归任务中最常见的平方误差**

均方误差为：

$$
L_{\mathrm{MSE}}=\frac{1}{N}\sum_{i=1}^{N}(\hat y_i-y_i)^2
$$

它的梯度与误差成正比：

$$
\frac{\partial L}{\partial \hat y_i}=\frac{2}{N}(\hat y_i-y_i)
$$

因此，预测偏得越远，修正力度越大。若假设观测满足 $y=\hat y+\epsilon$，且噪声 $\epsilon\sim\mathcal N(0,\sigma^2)$、方差固定，那么最小化高斯负对数似然与最小化 MSE 等价。MSE 的主要问题是平方项会放大离群点：误差从 $2$ 增加到 $10$，损失会从 $4$ 增加到 $100$。数据中异常值很多时，可考虑 MAE 或 Huber Loss。

补充：
$$ \mathrm{MAE}=\frac{1}{n}\sum_{i=1}^{n}|y_i-\hat y_i| $$
$$ L_\delta(e)= \begin{cases} \frac{1}{2}e^2, & |e|\leq \delta,\\[4pt] \delta\left(|e|-\frac{1}{2}\delta\right), & |e|>\delta \end{cases} $$
**为什么分类通常不用 MSE？** MSE 当然可以做分类，但它没有充分利用“类别概率分布”的结构。以二分类为例，若 $p=\sigma(z)$（sigmoid函数），MSE 的梯度中会额外出现 $p(1-p)$：

$$
\frac{\partial L_{\mathrm{MSE}}}{\partial z}\propto (p-y)p(1-p)
$$

当模型非常自信却预测错误时，$p$ 可能接近 $0$ 或 $1$，此时 $p(1-p)$ 很小，梯度反而容易变弱。交叉熵对 logits 的梯度更直接，通常能更有效地纠正“自信但错误”的预测。

- MSE在分类任务中最大的问题是经过sigmoid后，多乘了一个 $p(1−p)$，导致模型越自信时梯度越小，尤其是“自信但错误”的情况学习效率很低；交叉熵直接优化概率分布，因此更适合分类。

**2. Softmax：把 logits 转成多分类概率**

对于 $C$ 个类别，模型先输出任意实数 logits $z_1,\ldots,z_C$，（logits表示未归一化分数）Softmax 将其转化为和为 $1$ 的概率：

$$
p_i=\frac{e^{z_i}}{\sum_{j=1}^{C}e^{z_j}}
$$

Softmax 只关心 logits 之间的相对差异。给所有 logits 同时减去同一个常数，输出概率不变，因此实际计算时常减去最大值：

$$
p_i=\frac{e^{z_i-z_{\max}}}{\sum_j e^{z_j-z_{\max}}}
$$

这样可以避免 $e^{z_i}$ 过大造成数值溢出。Softmax 适合“每个样本只能属于一个类别”的单标签多分类；多标签分类中，各类别可以同时为真，通常应使用独立 Sigmoid 加 BCE，而不是 Softmax。

**3. 交叉熵：让正确类别获得更高概率**

目标分布为 $y$、预测分布为 $p$ 时，交叉熵为：

$$
H(y,p)=-\sum_{i=1}^{C}y_i\log p_i
$$

若标签采用 one-hot 编码，真实类别为 $k$，只有 $y_k=1$，交叉熵可化简为：

$$
L=-\log p_k
$$

这意味着模型只需为“分给真实类别的概率”负责。若 $p_k=0.9$，损失约为 $0.105$；若 $p_k=0.01$，损失约为 $4.605$。模型对错误类别越自信，惩罚越强。

交叉熵也等价于分类分布的负对数似然。训练分类器时最小化交叉熵，本质上是在最大化训练标签出现的似然。

**4. Softmax 与交叉熵为什么通常合并计算？**

令 $p=\mathrm{softmax}(z)$，交叉熵为 $L=-\sum_i y_i\log p_i$，对 logits 求导可得到一个极其简洁的结果：

$$
\frac{\partial L}{\partial z_i}=p_i-y_i
$$

含义非常直观：某类别预测概率高于目标值，就把对应 logit 往下调；预测概率低于目标值，就把对应 logit 往上调。

工程上合并计算还有两个关键原因：

- **数值稳定：** 直接先算 Softmax 再取 $\log$，可能出现概率下溢到 $0$，随后产生 $\log 0$。融合实现会使用 `logsumexp` 技巧，避免中间概率的精度损失。
- **计算简洁：** 框架可以直接从 logits 计算损失和梯度，不必显式保存完整的 Softmax Jacobian。

因此，PyTorch 的 `F.cross_entropy(logits, target)` 接收的是**原始 logits**，不要提前手动执行 Softmax。它内部相当于稳定版的 `LogSoftmax + NLLLoss`。

```python
import torch
import torch.nn.functional as F

logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, 1.2, 0.3]])
target = torch.tensor([0, 2])

loss_builtin = F.cross_entropy(logits, target)
log_probs = F.log_softmax(logits, dim=-1)
loss_manual = -log_probs.gather(1, target[:, None]).mean()

print(loss_builtin.item())
print(loss_manual.item())  # 两者应基本一致
```

**5. KL 散度与交叉熵的关系**

KL 散度衡量目标分布 $p$ 与近似分布 $q$ 的差异：

$$
D_{\mathrm{KL}}(p\|q)=\sum_i p_i\log\frac{p_i}{q_i}
$$

交叉熵满足：

$$
H(p,q)=H(p)+D_{\mathrm{KL}}(p\|q)
$$

其中 $H(p)$ 只由目标分布决定，其中$H(P)$是自身分布的信息熵，是一个固定跟的常数。训练时目标 $p$ 固定，因此最小化交叉熵 $H(p,q)$ 与最小化 $D_{\mathrm{KL}}(p\|q)$ 对模型参数来说具有相同的最优解。两者的差别是交叉熵还包含一个与模型无关的常数 $H(p)$。

其中交叉熵：$$ H(p,q)=-\sum_i p_i\log q_i $$
需要注意三点：

- KL 散度不对称，通常有 $D_{\mathrm{KL}}(p\|q)\neq D_{\mathrm{KL}}(q\|p)$。
- 监督分类中，$p$ 是真实标签分布，$q$ 是模型预测；知识蒸馏中，$p$ 常是教师模型的软分布，$q$ 是学生模型分布。
- KL 散度不是严格意义上的距离，因为它不对称，也不满足三角不等式。

**6. 语言模型为什么使用 token-level 交叉熵？**

自回归语言模型把一段文本的联合概率分解为一连串条件概率：一句话的概率 = 每一个 token 出现概率的乘积。

$$
P(x_1,\ldots,x_T)=\prod_{t=1}^{T}P(x_t\mid x_{<t})
$$

取负对数后，序列级负对数似然就变成各位置 token 交叉熵之和：

$$
\mathcal L=-\frac{1}{N_{\mathrm{tok}}}\sum_{b,t}m_{b,t}\log P_\theta(x_{b,t+1}\mid x_{b,\leq t})
$$

其中 $m_{b,t}$ 是有效 token 的掩码，Padding 位置不计入损失。模型在位置 $t$ 输出整个词表上的 logits，用它预测位置 $t+1$ 的真实 token，因此每个有效位置都是一次词表多分类，最终再对所有有效 token 求和或平均。

“token-level”不代表各 token 相互独立。每个位置的概率都以此前 token 为条件，依赖关系已经编码在上下文表示中。训练阶段通常使用 teacher forcing，即把真实前缀送入模型；推理阶段则需要把模型自己生成的 token 继续作为上下文。

- 每个 token 是一个分类问题。

```python
# input_ids: [batch_size, sequence_length]
# 位置 t 的输出用于预测位置 t+1
inputs = input_ids[:, :-1]
targets = input_ids[:, 1:]
logits = model(inputs)                 # [B, T-1, vocab_size]

loss = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    targets.reshape(-1),
    ignore_index=pad_token_id,
)
```

- 这段代码实现了 **GPT 类自回归语言模型训练中的 token-level 交叉熵损失计算**。
- 首先，`input_ids` 是经过 tokenizer 转换后的 token 序列，形状为 `[batch_size, sequence_length]`。由于语言模型采用“预测下一个 token”的训练方式，因此将输入序列向左移动一位：`inputs = input_ids[:, :-1]` 作为模型输入，将右移一位的 `targets = input_ids[:, 1:]` 作为真实预测目标，即模型在位置 $t$ 的输出用于预测位置 $t+1$ 的 token。
- 模型前向传播后得到 `logits`，其形状为 `[B, T-1, vocab_size]`，表示每个位置对整个词表的预测分数。随后通过 `reshape(-1, vocab_size)` 将 batch 维和序列维合并，使每个 token 预测都转换成一次独立的多分类任务，再使用 `cross_entropy` 计算预测分布与真实 token 之间的交叉熵损失。
- 其中 `ignore_index=pad_token_id` 用于忽略 padding token，避免补齐位置参与梯度计算。最终得到的 loss 就是所有有效 token 预测误差的平均值，用于优化语言模型参数。

若平均 token 负对数似然为 $\mathcal L$，困惑度通常定义为：

$$
\mathrm{PPL}=e^{\mathcal L}
$$

困惑度越低，表示模型平均给真实 token 分配的概率越高。但不同 tokenizer 会改变 token 划分方式，因此不同分词体系下的 PPL 不宜直接横向比较。

**7. 常见任务与损失函数速查**

| 任务 | 常见输出 | 常用损失 | 关键提醒 |
|---|---|---|---|
| 回归 | 任意实数 | MSE、MAE、Huber | MSE 对离群点敏感 |
| 二分类 | 1 个 logit | BCEWithLogitsLoss | 不要先手动 Sigmoid |
| 单标签多分类 | $C$ 个 logits | CrossEntropyLoss | 不要先手动 Softmax |
| 多标签分类 | $C$ 个独立 logits | BCEWithLogitsLoss | 每类独立判断，可同时为真 |
| 知识蒸馏 | 软概率分布 | KL / 软标签交叉熵 | 注意温度与 KL 方向 |
| 自回归语言模型 | 每个位置的词表 logits | token-level Cross Entropy | 标签右移，忽略 Padding |

**第一轮自检：** 你应当能直接说出：交叉熵为何比 MSE 更适合分类；`CrossEntropyLoss` 为什么接收 logits；语言模型的序列似然如何分解为 token 交叉熵；KL 与交叉熵只差哪一项。


| 问题                                            | 标准回答                                                                                                                                                                                                                                                                                                                                   |
| --------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. 为什么分类任务通常使用交叉熵，而不是 MSE？**                | 分类任务本质上是在学习类别概率分布，交叉熵可以直接衡量真实分布与预测分布之间的差异。MSE 虽然也能用于分类，但在二分类中，若预测概率为 $p=\sigma(z)$，其对 logit 的梯度为 $\frac{\partial L_{\mathrm{MSE}}}{\partial z}\propto(p-y)p(1-p)$。当模型非常自信但预测错误时，$p$ 接近 $0$ 或 $1$，此时 $p(1-p)$ 很小，梯度会明显减弱，模型纠错速度较慢。交叉熵对 logit 的梯度为 $\frac{\partial L_{\mathrm{CE}}}{\partial z}=p-y$，不会额外乘上 $p(1-p)$，因此通常具有更直接、更稳定的优化信号。 |
| **2. CrossEntropyLoss 为什么直接接收 logits，而不是概率？** | logits 是模型输出的未归一化类别分数。PyTorch 的 `CrossEntropyLoss` 内部已经组合了 `LogSoftmax` 和 `NLLLoss`，因此输入应当是 logits，而不需要提前执行 Softmax。其计算等价于 $L=-\log\frac{e^{z_y}}{\sum_{j=1}^{C}e^{z_j}}=-z_y+\log\sum_{j=1}^{C}e^{z_j}$。框架会利用 log-sum-exp 技巧进行稳定计算，避免直接计算指数和对数时出现数值上溢、下溢或精度损失。                                                                        |
| **3. 语言模型的序列损失为什么可以分解为 token-level 交叉熵？**     | 根据概率链式法则，自回归语言模型可以将整个序列的联合概率分解为每个 token 的条件概率乘积：$P(x_1,\dots,x_T)=\prod_{t=1}^{T}P(x_t\mid x_{<t})$。对联合概率取负对数后，乘积会转化为求和：$-\log P(x_1,\dots,x_T)=-\sum_{t=1}^{T}\log P(x_t\mid x_{<t})$。因此，序列级负对数似然等于各个位置 token 交叉熵之和。每个位置都相当于一次词表大小为 `vocab_size` 的多分类任务。                                                                              |
| **4. 为什么 token-level loss 不代表 token 之间相互独立？** | token-level 只说明损失在每个 token 位置分别计算，不代表模型假设各个 token 相互独立。语言模型建模的是条件概率 $P(x_t\mid x_{<t})$，当前 token 的预测依赖前面所有 token。例如预测“苹果”时，模型会使用“我喜欢吃”作为上下文。Transformer 通过 self-attention 将上下文依赖编码到隐藏表示中，因此各个 token 之间仍然存在依赖关系。                                                                                                                        |
| **5. KL 散度和交叉熵有什么关系？**                        | 交叉熵、信息熵和 KL 散度之间满足 $H(p,q)=H(p)+D_{\mathrm{KL}}(p\Vert q)$。其中，$p$ 是真实分布，$q$ 是模型预测分布，$H(p)$ 只由真实分布决定。在监督学习中，训练数据确定后，$H(p)$ 与模型参数无关，是一个常数。因此，最小化交叉熵 $H(p,q)$ 等价于最小化 KL 散度 $D_{\mathrm{KL}}(p\Vert q)$，也就是让预测分布 $q$ 尽可能接近真实分布 $p$。                                                                                                        |
| **6. 为什么 GPT 不直接预测整个句子，而是逐 token 预测？**        | 一个句子可能由大量不同的 token 组合而成，直接把所有可能句子作为类别会产生极其庞大的输出空间。自回归分解将复杂的序列生成问题转化为一系列“根据已有上下文预测下一个 token”的词表多分类问题。这样既可以共享同一个输出词表，也可以通过条件概率 $P(x_t\mid x_{<t})$ 建模不同长度的序列。                                                                                                                                                                            |
| **7. token-level loss 的具体计算过程是什么？**           | 对长度为 $T$ 的序列，通常使用前 $T-1$ 个 token 作为输入，使用后 $T-1$ 个 token 作为预测目标。模型在每个位置输出一个 `vocab_size` 维的 logits，并与该位置的真实下一个 token 计算交叉熵。带 mask 的平均损失可以写为 $\mathcal{L}=-\frac{1}{N_{\mathrm{tok}}}\sum_{b,t}m_{b,t}\log P_\theta(x_{b,t+1}\mid x_{b,\leq t})$。其中，$m_{b,t}$ 表示该位置是否为有效 token，Padding 位置不会计入损失。                                         |
| **8. 什么是 teacher forcing？训练和推理有什么区别？**        | teacher forcing 指训练时使用真实的历史 token 作为模型上下文。例如训练模型预测“苹果”时，输入真实前缀“我喜欢吃”，即学习 $P(\text{苹果}\mid\text{我喜欢吃})$。由于 Transformer 可以并行计算所有位置，训练时能够一次性得到整个序列的 token-level loss。推理时没有真实后续文本，只能把模型上一步生成的 token 继续作为下一步输入，因此推理过程通常是逐 token 自回归生成的。                                                                                                     |

## 二、优化器——拿到梯度以后怎么更新参数

**1. SGD：沿当前 mini-batch 梯度反方向前进**

$$
\theta_{t+1}=\theta_t-\eta g_t
$$

其中 $\eta$ 是学习率。mini-batch 梯度只是全数据梯度的有噪声估计，这种噪声可能带来震荡，却也可能帮助模型离开尖锐区域。SGD 状态少、额外显存开销低，经过充分调参后常有不错的泛化能力；它的缺点是对学习率敏感，在不同方向曲率差异很大时容易“之”字形前进。

**2. Momentum：对历史梯度做惯性累积**

一种常见写法为：

$$
v_t=\mu v_{t-1}+g_t,\qquad \theta_{t+1}=\theta_t-\eta v_t
$$

$\mu$ 通常接近 $0.9$。如果多个连续 batch 的梯度方向一致，动量会逐步积累，加快该方向的移动；若梯度在某个方向来回摆动，正负梯度会互相抵消，从而减少震荡。不同教材或框架对 $v_t$ 的缩放约定可能不同，但核心都是“使用历史梯度的指数加权信息”。

**3.Adam：一阶矩、二阶矩与自适应步长**

设第 $t$ 步的梯度为 $g_t$。Adam 同时维护梯度的一阶矩和平方梯度的二阶矩：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

其中：

- $m_t$ 是历史梯度的指数加权平均，主要反映近期稳定的更新方向；
- $v_t$ 是平方梯度的指数加权平均，主要反映每个参数的历史梯度尺度；
- $g_t^2$ 表示对梯度逐元素平方；
- 常用设置为 $\beta_1=0.9$、$\beta_2=0.999$。

一阶矩 $m_t$ 的作用类似 Momentum。如果连续多个 batch 的梯度方向一致，这些梯度会逐渐累积，使模型沿该方向更快前进；如果梯度在某个方向来回摆动，正负梯度会部分抵消，从而减少震荡。

二阶矩 $v_t$ 用来调节每个参数的实际更新步长：

- 某个参数的历史梯度长期较大时，$v_t$ 较大，该参数的有效步长会被压低；
- 某个参数的梯度较小或较稀疏时，$v_t$ 较小，该参数可以获得相对更大的有效步长。

严格来说，$v_t$ 是平方梯度的指数加权平均，不等同于统计学中的方差。

由于 Adam 通常初始化为：

$$
m_0=0,\qquad v_0=0
$$

训练初期的 $m_t$ 和 $v_t$ 会受到初始值 $0$ 的影响，估计结果偏小，因此需要做偏差修正：

$$
\hat m_t=\frac{m_t}{1-\beta_1^t}
$$

$$
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$

Adam 最终的参数更新公式为：

$$
\theta_{t+1}
=
\theta_t-\eta
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

其中：

- $\theta_t$ 是当前模型参数；
- $\eta$ 是基础学习率；
- $\hat m_t$ 决定主要更新方向；
- $\sqrt{\hat v_t}$ 根据历史梯度尺度调节每个参数的步长；
- $\epsilon$ 是防止分母为 $0$ 的小常数。

Adam 的核心可以概括为：

> 一阶矩决定“往哪里走”，二阶矩决定“每个参数走多远”。

Adam 通常收敛较快，对高维参数和稀疏梯度较友好，因此在 Transformer 训练中非常常见。它的代价是每个参数除了保存参数值和梯度，还需要额外保存 $m_t$、$v_t$ 两份优化器状态，因此显存开销明显高于普通 SGD。

Weight Decay：权重衰减

Weight Decay 中文通常译为 **权重衰减**。它的基本思想是：训练过程中持续把模型参数向 $0$ 缩小，从而限制权重过度增大。

一次权重衰减可以写成：

$$
\theta\leftarrow(1-\eta\lambda)\theta
$$

其中：

- $\eta$ 是学习率；
- $\lambda$ 是权重衰减系数；
- 因为 $1-\eta\lambda<1$，所以参数会逐渐缩小。

**为什么 SGD 中 L2 正则与 Weight Decay 等价？**

在原损失中加入 L2 正则项：

$$
L'=L+\frac{\lambda}{2}\|\theta\|_2^2
$$

对参数求导后，梯度变为：

$$
g'_t=g_t+\lambda\theta_t
$$

将其代入 SGD 更新：

$$
\theta_{t+1}
=
\theta_t-\eta(g_t+\lambda\theta_t)
$$

展开后：

$$
\theta_{t+1}
=
(1-\eta\lambda)\theta_t-\eta g_t
$$

其中：

- $(1-\eta\lambda)\theta_t$ 负责直接缩小参数；
- $-\eta g_t$ 负责根据任务损失更新参数。

因此，在普通 SGD 中，把 L2 正则加入损失，与直接执行 Weight Decay 在数学上等价。

**为什么在 Adam 中不等价？**

如果在 Adam 中直接把 L2 正则梯度加入任务梯度：

$$
g'_t=g_t+\lambda\theta_t
$$

那么 $\lambda\theta_t$ 会和任务梯度一起进入 Adam 的一阶矩和二阶矩：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g'_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)(g'_t)^2
$$

最终，这部分正则梯度也会被 Adam 的自适应分母处理：

$$
\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

这会导致原本希望统一执行的权重衰减，被 Adam 重新缩放：

- 某个参数的历史梯度较大，$\sqrt{\hat v_t}$ 较大，正则作用可能被压得较小；
- 某个参数的历史梯度较小，$\sqrt{\hat v_t}$ 较小，正则作用可能相对更强。

因此，不同参数实际受到的衰减强度不再统一。原本单纯的“缩小权重”，被 Adam 的一阶矩、二阶矩和逐参数缩放机制重新加工了。

所以在 Adam 中：

> L2 正则和标准 Weight Decay 不再等价。

**4.AdamW：将权重衰减与梯度更新解耦**

AdamW 将两件事分开处理：

1. 只使用任务梯度计算 Adam 更新；
2. 单独对参数执行 Weight Decay。

AdamW 的参数更新公式为：

$$
\theta_{t+1}
=
(1-\eta\lambda)\theta_t
-
\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

其中：

- 第一项 $(1-\eta\lambda)\theta_t$ 直接按固定比例缩小参数；
- 第二项按照 Adam 的方式学习任务损失。

这样，Weight Decay 不会进入 $m_t$ 和 $v_t$，也不会被 $\sqrt{\hat v_t}$ 逐参数缩放。这就是 AdamW 中“解耦”的核心。

可以这样记：

> Adam + L2：把正则项加入梯度，再交给 Adam 一起加工。  
> AdamW：Adam 负责根据任务梯度学习，Weight Decay 单独负责缩小权重。

Adam 与 AdamW 对比

| 对比项 | Adam | AdamW |
| --- | --- | --- |
| 自适应梯度更新 | 有 | 有 |
| 一阶矩与二阶矩 | 有 | 有 |
| 权重衰减处理 | 如果直接使用 L2，正则梯度会进入 $m_t$ 和 $v_t$ | Weight Decay 与梯度更新分开 |
| 衰减是否受自适应分母影响 | 会 | 不会 |
| 不同参数的衰减强度 | 可能不统一 | 按统一比例衰减 |
| Transformer 中的使用 | 常见 | 更常见 |
| 优化器状态显存 | 较高 | 与 Adam 基本相同 |

在 Transformer 训练中，通常只对权重矩阵执行 Weight Decay，而把以下参数排除：

- bias；
- LayerNorm 的缩放参数；
- RMSNorm 的缩放参数。

这些参数通常是一维参数，主要负责特征平移或尺度调节，把它们持续向 $0$ 收缩未必有益。这属于常见工程实践，不是绝对的数学定律。

**Adam 的优点是收敛快、对学习率不太敏感，并能为不同参数自动调整步长，适合高维和稀疏梯度任务；缺点是需要额外保存一阶矩和二阶矩，显存开销较大，而且直接加入 L2 正则时，权重衰减会被自适应机制干扰。AdamW 保留了 Adam 的快速、自适应更新，同时把权重衰减单独处理，正则化含义更清楚，通常更容易调参、泛化效果更好，因此 Transformer 中更常用；它的缺点是显存开销仍与 Adam 相近，并且还需要额外选择合适的权重衰减系数。**

面试问题

| 面试问题                                     | 标准回答                                                                                                               |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Adam 的核心思想是什么？**                       | Adam 同时维护梯度的一阶矩和平方梯度的二阶矩。一阶矩用于估计近期主要更新方向，二阶矩用于估计不同参数的历史梯度尺度，从而实现逐参数自适应步长。                                          |
| **Adam 为什么需要偏差修正？**                      | 因为 $m_0=v_0=0$，训练初期的一阶矩和二阶矩会被初始值拉向 $0$，从而产生低估。通过除以 $1-\beta_1^t$ 和 $1-\beta_2^t$ 可以修正这种初始化偏差。                      |
| **Adam 的优点和代价是什么？**                      | Adam 通常收敛较快，对高维参数和稀疏梯度友好，适合 Transformer。代价是每个参数还要保存一阶矩和二阶矩两份状态，因此显存开销高于 SGD。                                       |
| **Weight Decay 的中文和作用是什么？**              | Weight Decay 中文是“权重衰减”，作用是在每次更新时把参数按一定比例向 $0$ 缩小，限制参数过度增大。                                                         |
| **为什么 SGD 中 L2 正则与 Weight Decay 等价？**    | L2 正则产生的梯度为 $\lambda\theta_t$。代入 SGD 后，更新可以展开为 $\theta_{t+1}=(1-\eta\lambda)\theta_t-\eta g_t$，其中第一项正好是按固定比例缩小参数。  |
| **为什么在 Adam 中 L2 正则和 Weight Decay 不等价？** | 因为 L2 正则梯度 $\lambda\theta_t$ 会进入 Adam 的一阶矩、二阶矩，并被自适应分母逐参数缩放，导致不同参数受到的实际衰减强度不再统一。                                   |
| **AdamW 如何解决这个问题？**                      | AdamW 将任务梯度更新与权重衰减解耦。任务梯度只用于计算 Adam 更新，Weight Decay 单独按照 $(1-\eta\lambda)$ 缩小参数，因此不会进入 $m_t$ 和 $v_t$，也不会受到自适应分母干扰。 |
| **AdamW 与 Adam 是不是只差一个名字？**              | 不是。AdamW 改变了权重衰减进入参数更新的方式。Adam 加 L2 会把正则项当作梯度处理，而 AdamW 将权重衰减作为独立的参数收缩步骤。                                          |


**5. Muon：针对隐藏层矩阵参数的“正交化动量”优化器**

Muon 的名称通常解释为 **MomentUm Orthogonalized by Newton–Schulz**。它主要面向神经网络隐藏层中的二维权重矩阵，而不是把每个标量坐标独立处理。

- 普通 SGD 或 Adam 得到的梯度也是一个矩阵。有时这个梯度矩阵会被少数几个特别强的方向支配，导致很多神经元都朝着相似方向更新，其他较弱但有用的方向几乎得不到学习。

对某个矩阵参数 $W$，Muon 先对梯度形成动量矩阵 $M_t$，再对 $M_t$ 做近似正交化。若对 $M_t$ 做奇异值分解：

$$
M_t=U\Sigma V^\top
$$
正交化的作用：

- 强方向不会完全压制弱方向，多个方向能够更加均衡地参与更新。
- 注意：**Muon正交化的是更新矩阵，不是把模型权重 WWW 本身变成正交矩阵。** 官方实现将 Momentum 产生的二维更新矩阵替换为其近似的半正交极分解因子。

Muon 想得到的核心方向可理解为极分解中的 $UV^\top$。这一步会削弱不同奇异方向之间巨大的尺度差异，使更新不再被少数特别大的奇异值完全支配。**实际实现通常不直接做昂贵的完整 SVD，而是使用若干次 Newton–Schulz 型矩阵迭代来近似该结果。**

概念化更新可以写成：

$$
W_{t+1}=W_t-\eta\cdot \mathrm{Orthogonalize}(M_t)
$$

Muon 与 AdamW 的主要区别是：AdamW 主要进行逐元素的自适应缩放；Muon 利用权重和梯度的矩阵结构，对整个更新矩阵的奇异方向进行处理。原始用法通常让 Muon 负责隐藏层二维矩阵，而 embedding、输出头、bias、归一化增益等参数继续交给 AdamW，因此工程上往往是“Muon + 辅助 AdamW”，而不是全模型只用一个 Muon。

Muon 是较新的优化方向，理解其核心思想比死背某个版本的全部公式更重要：**先积累动量，再把矩阵更新近似正交化，让各主要方向获得更均衡的更新。** 它可能提高某些训练任务的优化效率，但也会带来额外矩阵乘法、分布式通信与超参数适配成本，不能简单理解为在所有任务上无条件替代 AdamW。

**6. 五种优化器对比**

| 优化器 | 核心机制 | 优点 | 主要代价/风险 | 常见场景 |
|---|---|---|---|---|
| SGD | 当前梯度 | 状态少、显存低、机制简单 | 对学习率敏感，收敛可能较慢 | 视觉任务、充分调参的训练 |
| Momentum | 历史梯度惯性 | 加速一致方向，减少震荡 | 多一份动量状态 | SGD 的常见增强版 |
| Adam | 一阶矩 + 二阶矩 | 收敛快、逐坐标自适应 | 状态显存大，需注意超参数 | NLP、Transformer、稀疏梯度 |
| AdamW | Adam + 解耦权重衰减 | 正则化含义更清楚，LLM 中常用 | 仍有较高状态显存 | Transformer 预训练与微调 |
| Muon | 动量矩阵正交化 | 利用矩阵结构，平衡奇异方向 | 额外矩阵计算，适用参数受限 | 隐藏层二维权重的探索性训练 |

**第二轮自检：** 你应当能不看笔记写出 Adam 的 $m_t$、$v_t$、偏差修正和更新式；能用一句话指出 Adam 与 AdamW 的本质差别；能说明 Muon 为什么通常只处理隐藏层二维矩阵。

常见优化器面试题

| 面试问题                                         | 标准回答                                                                                                                                                                                                          |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. SGD 的核心更新机制是什么？**                       | SGD 使用当前 mini-batch 计算出的梯度更新参数，公式为 $\theta_{t+1}=\theta_t-\eta g_t$。它只依据当前梯度，不保存历史梯度状态。mini-batch 梯度是全量梯度的有噪声估计，因此更新过程中可能出现震荡。                                                                                |
| **2. SGD 的主要优缺点是什么？**                        | SGD 的优点是机制简单、优化器状态少、显存开销低，经过充分调参后通常具有不错的泛化能力；缺点是对学习率敏感，在不同方向曲率差异较大时容易“之”字形震荡，收敛速度可能较慢。                                                                                                                        |
| **3. Momentum 相比 SGD 做了什么改进？**               | Momentum 在当前梯度之外，还会累积历史梯度形成动量：$v_t=\mu v_{t-1}+g_t$，$\theta_{t+1}=\theta_t-\eta v_t$。当多个 batch 的梯度方向一致时，动量会逐渐累积并加速前进；当梯度来回变化时，正负方向会部分抵消，从而减少震荡。                                                               |
| **4. Momentum 的代价是什么？**                      | Momentum 需要为每个参数额外保存一份动量状态，因此比普通 SGD 占用更多显存。同时，它仍然需要合理设置学习率和动量系数 $\mu$，通常 $\mu$ 接近 $0.9$。                                                                                                                     |
| **5. Adam 的核心思想是什么？**                        | Adam 同时维护梯度的一阶矩和平方梯度的二阶矩。一阶矩用于估计近期主要更新方向，二阶矩用于估计不同参数的历史梯度尺度，从而为每个参数设置自适应的有效步长。可以概括为：一阶矩决定“往哪里走”，二阶矩决定“每个参数走多远”。                                                                                               |
| **6. 写出 Adam 的一阶矩和二阶矩。**                     | Adam 的一阶矩为 $m_t=\beta_1m_{t-1}+(1-\beta_1)g_t$；二阶矩为 $v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2$。其中 $g_t^2$ 表示对梯度逐元素平方，常用设置为 $\beta_1=0.9$、$\beta_2=0.999$。                                                          |
| **7. Adam 为什么需要偏差修正？**                       | 因为 Adam 通常初始化为 $m_0=v_0=0$，训练初期的一阶矩和二阶矩会受到初始值影响而偏向 $0$。因此需要修正为 $\hat m_t=\frac{m_t}{1-\beta_1^t}$ 和 $\hat v_t=\frac{v_t}{1-\beta_2^t}$。                                                                       |
| **8. 写出 Adam 的参数更新公式。**                      | Adam 的参数更新为 $\theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$。其中 $\hat m_t$ 提供更新方向，$\sqrt{\hat v_t}$ 对每个参数的步长进行自适应缩放，$\epsilon$ 用于防止分母为 $0$。                                              |
| **9. Adam 的主要优缺点是什么？**                       | Adam 的优点是收敛较快，能够逐参数调整步长，对高维参数和稀疏梯度较友好，因此常用于 NLP 和 Transformer；缺点是需要为每个参数保存一阶矩和二阶矩，优化器状态显存较大，而且需要合理设置学习率、$\beta_1$ 和 $\beta_2$ 等超参数。                                                                           |
| **10. Adam 和 AdamW 的本质区别是什么？**               | Adam 如果直接加入 L2 正则，会把正则梯度 $\lambda\theta_t$ 一起放入一阶矩、二阶矩和自适应缩放过程；AdamW 则将权重衰减与梯度更新解耦，任务梯度用于 Adam 更新，权重衰减单独负责缩小参数。                                                                                               |
| **11. AdamW 每一步具体做哪两件事？**                    | 第一，只使用任务梯度计算 Adam 的自适应更新；第二，单独按照 $(1-\eta\lambda)$ 缩小需要衰减的权重。其更新公式为 $\theta_{t+1}=(1-\eta\lambda)\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}$。                                                     |
| **12. AdamW 将权重衰减解耦有什么好处？**                  | 解耦后，Weight Decay 不会进入 $m_t$、$v_t$，也不会被自适应分母逐参数缩放，因此不同权重可以按照统一比例衰减。这样权重衰减系数 $\lambda$ 的含义更清楚，通常更容易调参，并能获得更稳定的正则化效果。                                                                                            |
| **13. 为什么 AdamW 常用于 Transformer 和大语言模型？**    | AdamW 保留了 Adam 收敛快、逐参数自适应的优点，同时解决了 Adam 中 L2 正则与标准权重衰减不等价的问题，正则化方式更加清晰。因此它广泛用于 Transformer 的预训练和微调。                                                                                                           |
| **14. AdamW 的主要代价是什么？**                      | AdamW 仍然需要保存一阶矩和二阶矩，因此优化器状态显存与 Adam 基本相同。此外，还需要选择合适的权重衰减系数，并决定哪些参数不执行 Weight Decay。                                                                                                                           |
| **15. Transformer 中哪些参数通常不进行 Weight Decay？** | 常见实践是将 bias、LayerNorm 和 RMSNorm 的缩放参数排除在 Weight Decay 之外，因为这些参数主要负责平移或调节特征尺度。需要注意，这属于常见工程实践，不是不可违背的数学定律。                                                                                                      |
| **16. Muon 的核心思想是什么？**                       | Muon 主要针对隐藏层二维权重矩阵。它先对梯度累积动量，再对整个动量矩阵进行近似正交化，削弱不同奇异方向之间过大的尺度差异，使更新不会被少数特别强的方向完全支配。                                                                                                                            |
| **17. Muon 的更新过程是什么？**                       | Muon 首先形成动量矩阵 $M_t=\mu M_{t-1}+G_t$。若 $M_t$ 的奇异值分解为 $M_t=U\Sigma V^\top$，Muon 希望得到近似的 $UV^\top$，然后按照 $W_{t+1}=W_t-\eta\,\operatorname{Orthogonalize}(M_t)$ 更新权重。实际实现通常使用 Newton–Schulz 迭代近似正交化，而不是直接执行完整 SVD。 |
| **18. Muon 为什么通常只处理隐藏层二维矩阵？**                | Muon 的正交化依赖矩阵的行列结构和奇异方向，因此适合线性层、注意力投影等隐藏层二维权重。bias 和归一化缩放参数是一维向量，没有可供正交化的二维矩阵结构；embedding 和输出头的参数结构及功能也较特殊，通常继续交给 AdamW。                                                                                      |
| **19. Muon 与 AdamW 的主要区别是什么？**               | AdamW 主要进行逐元素的自适应缩放，为不同参数设置不同的有效步长；Muon 则把二维更新矩阵作为整体，对矩阵的奇异方向进行处理，使多个主要方向获得更加均衡的更新。                                                                                                                           |
| **20. Muon 的主要优点和代价是什么？**                    | Muon 的优点是能够利用二维权重矩阵的结构，平衡不同奇异方向，在部分训练任务中可能提高优化效率；代价是需要额外的矩阵乘法和正交化计算，分布式通信与实现更复杂，而且适用参数范围有限，不能简单理解为能够无条件替代 AdamW。                                                                                              |
| **21. 实际训练中 Muon 是否会完全替代 AdamW？**            | 通常不会。常见做法是隐藏层二维权重使用 Muon，而 embedding、输出头、bias 和归一化参数继续使用辅助 AdamW。因此工程上通常采用“Muon + AdamW”的组合。                                                                                                                  |
| **22. 如何快速概括五种优化器的区别？**                      | SGD 只看当前梯度；Momentum 在 SGD 上累积历史梯度；Adam 同时利用一阶矩和二阶矩进行逐参数自适应更新；AdamW 在 Adam 基础上解耦权重衰减；Muon 则利用二维矩阵结构，对整个动量矩阵的奇异方向进行近似正交化。                                                                                       |
| **23. 五种优化器的显存开销如何比较？**                      | SGD 的优化器状态最少；Momentum 需要额外保存一份动量；Adam 和 AdamW 通常需要保存一阶矩、二阶矩两份状态，因此显存开销较高；Muon 需要保存动量矩阵，并带来额外矩阵计算，其具体显存和计算成本取决于实现方式。                                                                                           |
| **24. 面对不同任务应如何选择优化器？**                      | 视觉任务或显存受限、愿意充分调参时可以考虑 SGD 或 Momentum；NLP、Transformer 和稀疏梯度任务通常使用 Adam；Transformer 预训练和微调通常优先使用 AdamW；Muon 更适合针对隐藏层二维权重进行探索性或经过验证的训练，通常与辅助 AdamW 配合使用。                                                         |

## 三、梯度、累积、裁剪与 loss 震荡

**1. 梯度消失和梯度爆炸从哪里来？**

深层网络反向传播时，前层梯度需要连续乘上多层 Jacobian：

$$
\frac{\partial L}{\partial h_l}
=
\frac{\partial L}{\partial h_L}
\prod_{k=l+1}^{L}
\frac{\partial h_k}{\partial h_{k-1}}
$$

若这些 Jacobian 的主要奇异值长期小于 $1$，连乘后梯度会指数缩小，形成梯度消失；若长期大于 $1$，连乘后梯度会迅速放大，形成梯度爆炸。

**梯度消失的常见表现：** 前面层的梯度范数接近 $0$，参数几乎不更新，训练损失很早进入平台期。常见诱因包括深层链式结构、Sigmoid/Tanh 饱和区、初始化不当以及过长的 RNN 时间依赖。

**梯度爆炸的常见表现：** 全局梯度范数突然变大，参数更新量暴涨，loss 剧烈跳升，严重时出现 `Inf` 或 `NaN`。常见诱因包括学习率过高、初始化尺度过大、长序列递归、异常 batch、混合精度溢出或优化器状态损坏。

常用缓解手段可以按层次理解：

- **结构层面：** 残差连接让梯度多出接近恒等映射的通路；LSTM/GRU 缓解长序列 RNN 的梯度问题。
- **归一化层面：** LayerNorm、RMSNorm、合理的 Pre-Norm 结构能控制激活和梯度尺度。
- **激活与初始化：** ReLU/GELU/SwiGLU 减少饱和；Xavier 常配合 Tanh，Kaiming 常配合 ReLU 类激活。
- **优化层面：** 降低学习率、使用 warmup、设置合理的梯度裁剪阈值。
- **数值层面：** 混合精度下使用 loss scaling，检查 `NaN/Inf`，对敏感运算保留 FP32。

**2. 梯度裁剪具体做了什么？**

最常见的是全局范数裁剪。把所有参数梯度视为一个长向量 $g$，给定阈值 $\tau$：

$$
g\leftarrow g\cdot\min\left(1,\frac{\tau}{\|g\|_2}\right)
$$

若 $\|g\|_2\leq\tau$，梯度不变；若超过阈值，所有梯度按同一比例缩小。因此，全局范数裁剪会保留整体梯度方向，只限制更新尺度。逐元素 value clipping 则会把每个梯度截断到固定区间，可能改变方向，深度学习训练中通常更常用 norm clipping。

梯度裁剪适合解决：

- 长序列或深层网络中的偶发梯度爆炸；
- 个别异常 batch 造成的极大更新；
- 训练早期或分布切换时的短暂不稳定；
- RNN、Transformer 等模型中需要控制全局更新尺度的场景。

梯度裁剪不能真正解决：

- 梯度消失，因为接近 $0$ 的梯度不会被放大；
- 长期过高的学习率，只会把很多更新持续压在阈值上；
- 错误标签、损坏数据、错误损失归一化和模型实现 bug；
- 已经产生的 `NaN/Inf`，因为对非有限值做缩放通常无济于事；
- 不合理的架构、初始化或优化器状态问题。

若阈值过小，几乎每一步都触发裁剪，训练等价于长期降低有效学习率，还可能掩盖真正的问题。实践中要同时记录**裁剪前梯度范数**和**触发裁剪的比例**。

```python
from torch.nn.utils import clip_grad_norm_

loss.backward()
pre_clip_norm = clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
optimizer.zero_grad(set_to_none=True)

print("裁剪前全局梯度范数：", float(pre_clip_norm))
```

代码解释：

>这段代码完成了一次带**梯度裁剪**的训练更新：`loss.backward()` 先通过反向传播计算所有参数的梯度；`clip_grad_norm_(model.parameters(), max_norm=1.0)` 计算模型的全局梯度范数，如果范数超过 $1.0$，就按相同比例缩小所有梯度，避免梯度爆炸，如果不超过则保持不变，同时返回**裁剪前的梯度范数**并保存到 `pre_clip_norm`；随后 `optimizer.step()` 根据裁剪后的梯度更新参数，`optimizer.zero_grad(set_to_none=True)` 清空本轮梯度，防止梯度累积；最后一行打印裁剪前的全局梯度范数，便于观察阈值是否设置过小，以及训练中有多少步骤触发了裁剪。

**3. 梯度累积是否等价于大 batch？**

假设一个大 batch 被平均分成 $K$ 个等大的 micro-batch，每个 micro-batch 的平均梯度为 $g_k$，则完整大 batch 的平均梯度是：

$$
g_{\mathrm{large}}=\frac{1}{K}\sum_{k=1}^{K}g_k
$$

因此，在参数不更新的前提下，依次对 $K$ 个 micro-batch 反向传播，并把每个 loss 除以 $K$，最后只执行一次 `optimizer.step()`，优化器看到的梯度可以与大 batch 相同。对 Adam/AdamW 也成立，因为它们只在最终 step 时用这一个聚合梯度更新一次 $m_t$ 和 $v_t$。

但“梯度累积等价于大 batch”需要一组条件：

- 累积期间不能提前更新参数，必须完成 $K$ 次 backward 后再 step；
  如果每个 micro-batch 都执行一次 `optimizer.step()`，那么模型参数已经在中途改变，后面的 micro-batch 面对的是不同模型，自然不等价于一次大 batch。
- loss 的缩放方式必须一致，等大 micro-batch 通常把每次平均 loss 除以 $K$；
- optimizer、学习率调度器和权重衰减都只执行一次；
- 梯度裁剪应放在全部梯度累积完成之后，不能每个 micro-batch 分别裁剪；
- BatchNorm 会使用各 micro-batch 的局部统计量，通常不等价于一次真正的大 batch；
- Dropout 等随机算子若随机掩码不同，数值上不一定逐位一致，尽管统计意义可能接近；
- 分布式梯度平均、混合精度 scaler 和随机数状态也会影响严格复现。


| 知识点                              | 通俗理解                                      | 正确做法                                                               | 否则会怎样                                                  |
| -------------------------------- | ----------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------ |
| **参数不能提前更新**                     | 先把多个小组的结果全部统计完，再统一做一次决定                   | 连续执行 $K$ 次 `backward()`，最后只执行一次 `optimizer.step()`                 | 如果每个 micro-batch 都更新参数，后面的数据面对的是已经改变的模型，不再等价于一个大 batch |
| **等大小 micro-batch 的 loss 要正确缩放** | 大 batch 要计算所有 micro-batch 梯度的平均值，而不是简单求和  | 每次使用 `(loss / K).backward()`，使最终梯度为 $\frac{1}{K}\sum_{k=1}^{K}g_k$ | 如果不除以 $K$，累计梯度会放大 $K$ 倍，相当于增大了有效学习率                    |
| **优化器每个累积窗口只更新一次**               | 多个 micro-batch 合起来才算完整的一步训练               | 累积完成后调用一次 `optimizer.step()`，然后再清空梯度                               | 多次调用会把一个大 batch 错误地变成多次参数更新                            |
| **学习率调度器只执行一次**                  | 学习率调度应跟随真正的参数更新次数，而不是 micro-batch 数量      | 每次 `optimizer.step()` 后再执行一次 `scheduler.step()`                    | 如果每个 micro-batch 都调度一次，学习率会下降得过快                       |
| **Weight Decay 只执行一次**           | 权重衰减属于参数更新的一部分，一个大 batch 只应衰减一次           | 让 Weight Decay 随最终的 `optimizer.step()` 执行一次                        | 如果每个 micro-batch 都衰减，权重会被重复缩小                          |
| **梯度裁剪应在累积完成后执行**                | 应先汇总所有人的意见，再对最终结果限幅                       | 完成全部 `backward()` 后，对累计梯度裁剪一次，再执行 `optimizer.step()`               | 每个 micro-batch 分别裁剪后再相加，通常不等于先累积再整体裁剪                  |
| **BatchNorm 通常无法严格等价**           | 真正的大 batch 用所有样本一起计算均值和方差，而梯度累积只能按小组分别计算  | 尽量使用 LayerNorm、RMSNorm，或采用专门的同步统计方法                                | 各 micro-batch 使用不同的局部统计量，前向结果已经发生变化                    |
| **Dropout 可能造成数值差异**             | 每次前向传播随机屏蔽的神经元可能不同                        | 接受统计意义上的近似；严格复现时固定随机种子和随机状态                                        | 即使平均梯度接近，也不一定与真正的大 batch 逐元素完全一致                       |
| **混合精度和分布式计算会带来误差**              | 浮点运算顺序变化可能产生细微数值差异                        | 正确使用 scaler、梯度同步和随机数状态，并保持操作顺序一致                                   | 通常训练效果接近，但难以保证逐位完全相同                                   |
| **语言模型要按有效 token 加权**            | 每个 token 应拥有相同权重，而不是每个 micro-batch 拥有相同权重 | 累计所有有效 token 的 loss 总和，最后除以整个窗口的有效 token 总数                        | 如果短序列和长序列的 micro-batch 被赋予相同权重，结果不等价于对全部 token 统一求平均   |



对于大小相同的 $K$ 个 micro-batch：

$$
g_{\mathrm{large}}
=
\frac{1}{K}\sum_{k=1}^{K}g_k
$$




**4. 梯度累积、混合精度与裁剪的正确顺序**

下面代码假设每个累积窗口正好包含 `accum_steps` 个 micro-batch，且各 micro-batch 的 loss 采用相同归一化方式：

```python
import torch
from torch.nn.utils import clip_grad_norm_

# 累积 4 个 micro-batch 后更新一次参数
accum_steps = 4

# 梯度范数上限
max_grad_norm = 1.0

# 清空上一轮梯度
optimizer.zero_grad(set_to_none=True)

for micro_step, batch in enumerate(train_loader):

    # 使用 BF16 混合精度进行前向计算
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(**batch).loss

        # 防止梯度累积后整体梯度扩大 accum_steps 倍
        scaled_loss = loss / accum_steps

    # 累积当前 micro-batch 的梯度，不更新参数
    scaled_loss.backward()

    # 每累积 accum_steps 次，更新一次参数
    if (micro_step + 1) % accum_steps == 0:

        # 裁剪梯度，返回裁剪前的梯度范数
        grad_norm = clip_grad_norm_(
            model.parameters(),
            max_grad_norm
        )

        # 更新模型参数
        optimizer.step()

        # 更新学习率
        scheduler.step()

        # 清空梯度，开始下一轮累积
        optimizer.zero_grad(set_to_none=True)

        print(
            "原始 loss:", float(loss),
            "裁剪前 grad norm:", float(grad_norm)
        )
```

若使用 FP16 和 `GradScaler`，顺序应为：先 `scaler.scale(loss).backward()`，累积完成后执行 `scaler.unscale_(optimizer)`，再进行梯度裁剪，最后 `scaler.step(optimizer)` 与 `scaler.update()`。原因是裁剪阈值应作用于还原后的真实梯度，而不是被 scale 放大后的梯度。BF16 动态范围更大，很多训练不使用 GradScaler，但仍要依据具体硬件与框架配置判断。

**5. loss 震荡和突然升高怎么排查？**

mini-batch 训练中的小幅上下波动很正常，因为每个 batch 难度不同。真正需要关注的是：震荡幅度持续扩大、单步出现巨大尖峰、尖峰后无法恢复，或直接出现 `NaN/Inf`。

可以按现象分类：

- **偶发尖峰后恢复：** 可能是异常样本、超长序列、极端类别分布、偶发大梯度或随机噪声。
- **周期性尖峰：** 检查数据分桶、不同任务轮换、梯度累积边界、评估/训练模式切换和学习率调度周期。
- **某一步后永久变差：** 检查学习率突变、优化器状态损坏、断点恢复遗漏 scheduler/scaler、参数更新过大或数据分布切换。
- **直接 NaN/Inf：** 检查混合精度溢出、除零、对非正数取对数、无效标签、注意力 mask、Softmax 极值和输入数据非有限值。

建议固定下面的排查顺序：

1. 记录异常发生的全局 step、学习率、batch 标识、序列长度和有效 token 数；
2. 记录总 loss 与各子损失，确认是否某个损失项突然失控；
3. 在裁剪前记录 global grad norm，同时记录 parameter norm 与 update norm；
4. 检查 logits、激活、梯度中是否出现 `NaN/Inf`；
5. 保存异常 batch 并单独重放，判断问题能否稳定复现；
6. 检查最近是否修改了数据、loss 归一化、累积步数、学习率或模型结构；
7. 若从 checkpoint 恢复，确认 optimizer、scheduler、随机数状态和 mixed-precision scaler 都已恢复。

一个很有用的指标是更新参数比例：

$$
\mathrm{UpdateRatio}=\frac{\|\Delta\theta\|_2}{\|\theta\|_2+\epsilon}
$$

若 loss 在某一步之后突然变坏，同时 UpdateRatio 明显升高，通常意味着学习率、梯度尺度或优化器状态导致更新过猛。若 loss 在 forward 时就异常，而参数还没更新，则应优先检查数据和损失计算。

**第三轮自检：** 你应当能说出梯度消失/爆炸的 Jacobian 连乘解释；能写出 global norm clipping 公式；能列出梯度累积与大 batch 等价的至少四个条件；能给出 loss spike 的排查顺序。

## 四、代码验证、面试串讲与复习

**1. 建议完成的两个小实验**

**实验 A：验证 Softmax + 交叉熵的梯度是 $p-y$。** 创建一个带 `requires_grad=True` 的 logits 张量，用 `F.cross_entropy` 反向传播，再手动计算 `softmax(logits) - one_hot(target)`。注意 PyTorch 默认对 batch 求平均，所以手动结果也要除以 batch size。

```python
import torch
import torch.nn.functional as F

logits = torch.tensor(
    [[2.0, 0.5, -1.0], [0.1, 1.2, 0.3]],
    requires_grad=True,
)
target = torch.tensor([0, 2])

loss = F.cross_entropy(logits, target)
loss.backward()

y = F.one_hot(target, num_classes=3).float()
manual_grad = (F.softmax(logits.detach(), dim=-1) - y) / logits.size(0)

print("autograd 梯度：\n", logits.grad)
print("手算 p-y：\n", manual_grad)
print("是否一致：", torch.allclose(logits.grad, manual_grad, atol=1e-6))
```

**实验 B：验证梯度累积与大 batch。** 关闭 Dropout，避免 BatchNorm，使用同一组样本和同一初始参数。方案一一次性对完整 batch 反向并 step；方案二拆成等大的 micro-batch，每次 loss 除以累积步数，全部 backward 后只 step 一次。比较两种方案更新后的参数差异。若条件控制正确，差异应接近浮点误差；随后打开 BatchNorm 或改成每个 micro-batch 都 step，观察两者为何不再相同。

**2. 五个重点问题的面试回答模板**

**Q1：Softmax 与交叉熵为什么通常合并计算？**  
Softmax 把 logits 归一化为类别概率，交叉熵最大化真实类别的对数概率。两者合并后，对 logits 的梯度简化为 $p-y$；实现上还能使用 `logsumexp`，避免先算极小概率再取对数造成下溢。因此训练时一般直接把 logits 交给 CrossEntropyLoss。

**Q2：KL 散度与交叉熵是什么关系？**  
$H(p,q)=H(p)+D_{\mathrm{KL}}(p\|q)$。当目标分布 $p$ 固定时，$H(p)$ 与模型参数无关，所以最小化交叉熵等价于最小化正向 KL。KL 不对称，蒸馏时通常把教师分布放在前面、学生分布放在后面。

**Q3：AdamW 为什么说权重衰减与梯度更新解耦？**  
Adam 中若把 $\lambda\theta$ 直接加到梯度，它会和任务梯度一起进入动量、平方梯度和逐坐标缩放，不再等价于统一的 weight decay。AdamW 直接对参数做 $(1-\eta\lambda)$ 收缩，再执行 Adam 的损失梯度更新，因此衰减不受二阶矩逐坐标缩放。

**Q4：梯度累积是否等价于大 batch？**  
在参数保持不变、loss 缩放一致、只执行一次 optimizer/scheduler step、累积结束后才裁剪梯度，并且没有 BatchNorm 等依赖 micro-batch 统计的操作时，累积平均梯度可以等于大 batch 梯度。变长语言模型还要按总有效 token 数加权，否则简单平均多个 micro-batch loss 不严格等价。

- **batch**：批次，或一批数据
- **loss**：损失值
- **optimizer step**：优化器更新一步，也就是更新一次模型参数
- **scheduler step**：学习率调度器更新一步
- **BatchNorm**：批归一化
- **micro-batch**：微批次，即梯度累积过程中每次实际送入模型的一小批数据

**Q5：梯度裁剪解决什么问题，不能解决什么问题？**  
它限制偶发大梯度造成的更新幅度，主要缓解梯度爆炸和异常 batch 带来的尖峰。它不能修复梯度消失、错误数据、错误 loss、长期过大学习率或已经产生的 NaN。若每一步都被裁剪，说明阈值过小或根因尚未解决。

**3. 其余高频追问速答**

- **为什么 Adam 需要偏差修正？** 因为一阶矩和二阶矩从 $0$ 初始化，训练初期的指数移动平均会系统性偏小，除以 $1-\beta_1^t$ 和 $1-\beta_2^t$ 可校正这一初始化偏差。
- **为什么语言模型是 token-level 交叉熵？** 自回归联合概率可分解成每个 token 在前缀条件下的概率乘积，负对数把乘积变成逐 token 损失之和。
- **为什么不能在 CrossEntropyLoss 前先 Softmax？** 会重复归一化，而且失去融合实现的数值稳定性；框架需要原始 logits。
- **Muon 的一句话定义是什么？** 对隐藏层二维权重的梯度动量做 Newton–Schulz 近似正交化，再用该矩阵方向更新参数；非矩阵参数通常交给 AdamW。
- **loss 震荡一定是训练失败吗？** 不是。mini-batch 噪声会带来正常波动；需要关注幅度是否扩大、平均趋势是否恶化、是否伴随梯度范数或更新比例异常。
