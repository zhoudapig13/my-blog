---
title: "『LLM学习笔记7』后训练：SFT、Instruction Tuning 与 LoRA / QLoRA"
category: "未分类"
tags:
  []
date: "2026-08-21"
summary: ""
pdf: ""
pdfTitle: ""
---

**Day 7：SFT、Instruction Tuning 与 LoRA / QLoRA**

> **今日主线**：前 6 天解决了“LLM 怎么预训练、怎么在多 GPU 上训练”，Day7 开始进入 **Post-training（后训练）**：一个只会做 Next Token Prediction 的 Base Model，怎样通过监督数据学会“听懂指令并按要求回答”，以及怎样用 LoRA / QLoRA 在有限显存下完成微调。
>


## 第一轮：从 Pretraining 到 SFT——Base Model 为什么还不会“听话”

**1. 先把 Pretraining、SFT、Preference Alignment 放到同一条线上**

一个典型 Decoder-only LLM 的训练流程可以粗略看成：

```text
海量通用文本
    ↓
Pretraining
    ↓
Base Model
    ↓
高质量 instruction / dialogue 数据
    ↓
SFT（Supervised Fine-Tuning）
    ↓
SFT / Instruction Model
    ↓
Preference Alignment（DPO / RLHF / GRPO ...）
    ↓
最终 Chat / Reasoning Model
```

Pretraining 解决的是：

> **“语言和世界知识是什么样的？”**

SFT 主要解决：

> **“用户这样问时，我应该以什么方式回答？”**

Preference Alignment 再进一步解决：

> **“多个都能说通的回答里，哪一种更符合人类偏好、任务要求或奖励信号？”**

因此，SFT 不是重新从零教模型语言，而是在 Base Model 已经具有大量语言能力和知识的基础上，用较少但质量更高的监督样本改变模型的**条件输出行为**。

**2. Base Model 为什么不等于 Chat Model？**

预训练样本可能只是：

```text
梯度累积是一种在显存有限时增大全局 batch size 的方法。训练时……
```

Base Model 看到：

```text
用户：请解释什么是梯度累积。
```

它在数学上只被训练成：

> 根据前面的 token，预测最可能的下一个 token。

它并没有天然被规定：

- `用户：` 后面必须由“助手”回答；
- 应该直接回答而不是继续编造新的用户对话；
- 应该遵守 system instruction；
- 应该用问答式、列表式、代码式还是其他格式。

SFT 数据会显式提供这种映射：

```text
System: 你是一名大模型学习助手。
User: 什么是梯度累积？
Assistant: 梯度累积是把多个 micro-batch 的梯度累加后，再进行一次参数更新……
```

模型反复看到类似结构后，就逐渐学到：

```text
instruction / dialogue context
            ↓
      合适的 response
```

**3. SFT 的训练目标有没有变？**

这是 Day7 最重要的概念之一：

> **对于 Decoder-only LLM，SFT 本质上通常仍然是 Causal Language Modeling / Next Token Prediction。**

预训练目标：

$$
\mathcal{L}_{\text{pretrain}}
=
-\sum_{t=1}^{T}
\log p_{\theta}(x_t\mid x_{<t})
$$

SFT 也仍然使用类似的 token-level cross entropy，只是**数据分布**以及**哪些 token 被计入 loss**发生了变化。

如果只对 Assistant 回复计算损失，设监督 mask 为 $m_t\in\{0,1\}$，则可以写成：

$$
\mathcal{L}_{\text{SFT}}
=
-\frac{1}{\sum_t m_t}
\sum_{t=1}^{T}
m_t\log p_{\theta}(x_t\mid x_{<t})
$$

其中：

- $x_t$：第 $t$ 个目标 token；
- $m_t=1$：这个 token 参与 SFT loss；
- $m_t=0$：这个 token 只作为上下文，不计算 loss。

所以 SFT **没有把 GPT 从“预测下一个 token”改造成另一种网络**。它仍然在预测下一个 token，只不过现在被重点监督的是“Assistant 应该如何继续”。

**4. Instruction Tuning 与 SFT 是什么关系？**

日常语境中这两个词经常混用，但可以稍微区分：

- **SFT**：更宽泛，指使用有监督数据对预训练模型继续训练。
- **Instruction Tuning**：SFT 的一种典型形式，数据被组织成“指令 → 回答”或对话格式，目标是增强 instruction-following 能力。

例如分类任务也可以做普通 SFT：

```text
文本 → 标签
```

而 Instruction Tuning 更像：

```text
请判断下面文本的情感：……
→ 正面
```

现代 Chat LLM 的 SFT 通常就是大规模、多任务、多轮对话形式的 Instruction Tuning。

![Pasted image 20260821023526](/my-blog/resources/uploads/obsidian-1787301359044-1.png)

**这一轮面试题**

| 面试问题 | 回答参考 |
| --- | --- |
| Base Model 和 Chat Model 的核心区别是什么？ | Base Model 主要经过大规模 next-token pretraining，学到了语言模式、知识和一定推理能力，但并没有被明确训练成稳定遵循用户指令的对话助手。Chat Model 通常在 Base Model 上继续经历 SFT 和 preference alignment，使模型学习对话模板、instruction-following、输出风格以及人类偏好。因此两者网络架构可以几乎一样，主要差别来自后训练数据与目标。 |
| SFT 的目标函数和预训练有什么本质区别？ | 对 Decoder-only LLM 来说，两者底层通常仍是 causal language modeling，即预测下一个 token。区别主要在数据分布和 loss mask：预训练通常在自然文本的大量 token 上计算 loss；Instruction SFT 常把 system/user 作为条件，只对 assistant response 或 completion token 计算 loss。 |
| 为什么已经预训练好的模型还需要 SFT？ | 预训练优化的是通用文本续写概率，并没有直接规定“看到用户指令后应该如何回答”。SFT 用高质量 instruction-response 数据把已有能力映射到用户希望的交互形式，例如遵循指令、回答而非继续编造对话、遵守输出格式等。 |
| Instruction Tuning 和 SFT 是完全一样的吗？ | 严格说 SFT 更宽泛，只要用有标签监督数据继续训练都可称为 SFT；Instruction Tuning 是其中面向指令遵循的一类 SFT。现代 LLM 场景中两者经常被近似混用。 |
| SFT 会给模型增加大量新知识吗？ | 可能会学到训练数据中的一些新事实，但它更擅长改变行为和任务适配，而不是高效、可靠地向模型注入大规模新知识。大规模知识学习主要仍发生在 pretraining / continued pretraining 阶段。 |

## 第二轮：SFT 数据、Chat Template 与 Loss Mask——模型到底在对哪些 Token 学习

**1. 一条对话在进入模型前是什么样？**

原始数据可能保存成：

```python
messages = [
    {"role": "system", "content": "你是一名大模型学习助手。"},
    {"role": "user", "content": "什么是梯度累积？"},
    {"role": "assistant", "content": "梯度累积是把多个 micro-batch 的梯度累加后再更新参数。"}
]
```

但 Transformer 不认识 Python 字典，也不直接认识 `system/user/assistant` 这些角色。真正输入模型前，需要通过 **Chat Template** 把结构化消息序列化成模型训练时约定的字符串，再 Tokenize。

概念上类似：

```text
<|system|>
你是一名大模型学习助手。
<|user|>
什么是梯度累积？
<|assistant|>
梯度累积是把多个 micro-batch 的梯度累加后再更新参数。
<|eos|>
```

不同模型的特殊 token、角色标记和结束符可能不同，所以：

> **不要随意拿 A 模型的 Chat Template 给 B 模型使用。**

Chat Template 本质上是在告诉模型：

> “这一段 token 属于谁，这一轮在哪里开始、在哪里结束。”

**2. `input_ids`、`attention_mask`、`labels` 再区分一次**

这正好和 Day5 接起来。

假设序列是：

```text
[System tokens] [User tokens] [Assistant tokens] [PAD]
[System tokens] [User tokens]属于prompt tokens
```

三类张量承担的职责不同：

| 张量               | 作用                                       |
| ---------------- | ---------------------------------------- |
| `input_ids`      | 告诉模型当前输入的 token 是什么                      |
| `attention_mask` | 告诉 Attention 哪些位置是有效 token，哪些通常是 padding |
| `labels`         | 告诉 loss 应该监督哪些 token；常用 `-100` 表示忽略      |

例如：

```text
位置：       System      User        Assistant        PAD
input_ids：  [ ... ]     [ ... ]     [ ... ]          [PAD]
attention：     1           1           1               0
labels：       -100        -100       token_id          -100
```

最关键的一句话：

> **Prompt token 完全可以被模型“看到”，但不一定参与 loss。（生成的内容算loss）**

Assistant 在生成回答时当然需要读取 System 和 User，否则它不知道应该回答什么。因此 prompt 必须作为 causal context 参与 forward；只是在计算 cross entropy 时，可以把这些位置的 label 改成 `-100`。

![Pasted image 20260821031424](/my-blog/resources/uploads/obsidian-1787301359044-2.png)

**3. 为什么 labels 还要“右移”？**

Causal LM 并不是用当前位置预测当前位置，而是：

```text
输入：  x1   x2   x3   x4
目标：  x2   x3   x4   x5
```

概念上：

$$
h_t \rightarrow p(x_{t+1}\mid x_{\le t})
$$

实际 Hugging Face 的 Causal LM 实现通常会在 loss 内部完成 logits 与 labels 的 shift，因此你构造 `labels` 时通常直接和 `input_ids` 对齐，不需要手工再右移一次。

容易踩的坑：

> 如果框架内部已经 shift，而你又手工 shift 一次，监督位置就会错一位。

**4. Response-only / Assistant-only Loss 到底是什么？**

单轮 Prompt-Completion：

```text
Prompt:
解释什么是 LoRA。

Completion:
LoRA 通过冻结原模型权重……
```

可以只计算 Completion：

```text
Prompt                  Completion
× × × × × ×             ✓ ✓ ✓ ✓ ✓ ✓
```

多轮对话：

```text
System
User 1
Assistant 1
User 2
Assistant 2
```

常见做法之一是只监督所有 Assistant 段：

```text
System       User 1       Assistant 1       User 2       Assistant 2
  ×            ×              ✓               ×              ✓
```

于是：

$$
\text{labels}_t=
\begin{cases}
x_t,&t\in\text{Assistant tokens}\\
-100,&\text{otherwise}
\end{cases}
$$

但注意：

> **“SFT 必须只对 Assistant 算 loss”并不是数学定律。**

有些训练会对完整序列计算 LM loss，有些只对 completion，有些只对 assistant turn。具体选择取决于数据格式和训练目标。对于标准聊天 Instruction Tuning，assistant-only loss 很常见，也最容易理解。

当前 TRL 中：

```python
SFTConfig(assistant_only_loss=True)
```

可以让兼容 Chat Template 的 conversational dataset 只在 Assistant 消息上计算 loss；Prompt-Completion 数据则可以使用 completion-only loss。

**5. Packing 为什么 Day5 学过，SFT 还要再学一次？**

SFT 数据的长度往往差异很大：

```text
样本 A：120 tokens
样本 B：350 tokens
样本 C：80 tokens
```

如果每个样本都 padding 到 1024：

```text
A：120 有效 + 904 PAD
B：350 有效 + 674 PAD
C： 80 有效 + 944 PAD
```

大量算力花在 padding 上。

Packing 的思路是：

```text
固定长度 1024 token block
┌─────────┬───────────────┬───────┬─────────┐
│ sample A│   sample B    │sample C│ sample D│
└─────────┴───────────────┴───────┴─────────┘
```

把多个短样本尽量塞进一个固定长度序列，从而减少 padding 浪费。

但要注意两个问题：

- **样本边界**：不能因为 pack 在一起，就错误地让不同样本共享语义上下文；具体框架会通过位置、mask 或 packing implementation 正确处理。
- **EOS**：每条样本结束位置必须正确，否则模型可能学不到何时停止回答。

**6. 截断为什么会悄悄毁掉 SFT 数据？**

假设最大长度是 2048，但样本：

```text
System + User = 1900 tokens
Assistant = 600 tokens
```

总长度 2500。

如果粗暴保留前 2048 个 token：

```text
保留了大量 Prompt
却只保留了 Assistant 前 148 tokens
```

那真正有监督价值的回答被切掉大半。

所以 SFT 数据预处理至少要统计：

- token length 分布；
- Prompt / Response 各自长度；
- 被 truncation 的比例；
- assistant supervised token 数；
- 空 response / 异常模板；
- EOS 是否正确。

**7. 一条 SFT 样本真正经过了什么**

```text
messages
   ↓
Chat Template
   ↓
formatted text
   ↓
Tokenizer
   ↓
input_ids
   ↓
构造 attention_mask / labels
   ↓
Causal Transformer
   ↓
logits: [B, L, |V|]
   ↓
只在指定 labels 位置计算 Cross Entropy
   ↓
Backward
```

![Pasted image 20260821031706](/my-blog/resources/uploads/obsidian-1787301359044-3.png)


**这一轮面试题**

| 面试问题                                    | 回答参考                                                                                                                                                                                        |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `attention_mask` 和 `labels=-100` 有什么区别？ | `attention_mask` 主要控制 Attention 中哪些 token 被视为有效上下文，例如屏蔽 padding；`labels=-100` 控制某个位置是否参与 loss。一个 User token 可以 `attention_mask=1`，因为 Assistant 需要读取它，同时 `label=-100`，因为不希望它贡献监督损失。          |
| 为什么 Prompt 不算 loss，模型还能学会根据 Prompt 回答？  | Prompt 仍参与 forward，Assistant token 的 hidden state 会通过 causal attention 条件于前面的 Prompt。Assistant token 上的 loss 反向传播时，梯度会经过整个计算图，因此模型会学习“在这种 Prompt 条件下产生正确 Response”，并不要求 Prompt 自己也必须作为预测目标。 |
| SFT 为什么需要 Chat Template？                | 原始 `messages` 是结构化数据，Transformer 最终只接受 token 序列。Chat Template 把 system/user/assistant 角色、轮次边界和 EOS 等信息编码成模型训练时一致的 token 格式。模板用错会造成明显的 train-inference mismatch。                             |
| Causal LM 的 labels 为什么通常不用自己手工 shift？   | 因为常见 `AutoModelForCausalLM` 的 loss 实现内部会把 logits 和 labels 对齐成“位置 $t$ 预测 $t+1$”。如果外部再 shift 一次，监督会发生错位。具体实现仍应检查模型 forward / loss 代码。                                                         |
| Packing 的主要作用是什么？                       | 减少不同长度 SFT 样本 padding 到统一长度造成的算力浪费，把多个短样本尽量填进固定长度 block，从而提高 token utilization 和吞吐。但需要正确处理样本边界和 EOS，不能简单字符串硬拼。                                                                              |
| 为什么 SFT 要特别关注 truncation？               | 因为长 Prompt 可能把真正用于监督的 Assistant Response 截掉，表面上样本数没变，但有效 supervised tokens 大幅减少。实践中应先统计长度分布，再设置 `max_length` 和截断策略。                                                                         |
| SFT 一定只对 Assistant token 算 loss 吗？      | 不一定。这是常见的 instruction tuning 设计，但不是理论要求。可以 full-sequence loss、completion-only loss 或 assistant-only loss，选择取决于数据格式和训练目标。                                                                    |

## 第三轮：LoRA——为什么两个小矩阵能代替全参数微调

**1. Full Fine-tuning 到底贵在哪里？**

假设 Transformer 中有一个线性层：

$$
y=Wx
$$

其中：

$$
W\in\mathbb{R}^{d_{\text{out}}\times d_{\text{in}}}
$$

Full Fine-tuning 会直接更新整个 $W$：

$$
W\leftarrow W+\Delta W
$$

如果模型有数十亿参数，训练期间不仅要保存模型参数，还涉及：

```text
模型参数
+ 参数梯度
+ optimizer state
+ activation
+ 临时 buffer
```

以 AdamW 为例，优化器通常还需要为每个可训练参数维护一阶矩和二阶矩：

$$
m_t,\qquad v_t
$$

因此“7B 模型权重能装进显存”并不代表“7B 模型就能全参数训练”。

**2. LoRA 的核心不是低秩分解 $W$，而是低秩参数化 $\Delta W$**

这是一个很容易说错的面试点。

LoRA 不需要把原始权重 $W$ 本身分解掉，而是假设：

> 下游任务所需要的**权重更新** $\Delta W$ 可以由较低秩的结构近似表达。

写成：

$$
\Delta W=BA
$$

其中：

$$
A\in\mathbb{R}^{r\times d_{\text{in}}}
$$

$$
B\in\mathbb{R}^{d_{\text{out}}\times r}
$$

并且：

$$
r\ll \min(d_{\text{in}},d_{\text{out}})
$$

于是原本：

$$
y=Wx
$$

变成：

$$
y=Wx+\frac{\alpha}{r}BAx
$$

训练时：

```text
W：冻结
A：训练
B：训练
```

其中 $\alpha/r$ 是常见 LoRA scaling。

![Pasted image 20260821034239](/my-blog/resources/uploads/obsidian-1787301359044-4.png)

**3. 为什么参数量会少这么多？**

原始矩阵参数量：

$$
N_W=d_{\text{out}}d_{\text{in}}
$$

LoRA 参数量：

$$
N_{\text{LoRA}}
=
r(d_{\text{in}}+d_{\text{out}})
$$

例如：

$$
d_{\text{in}}=d_{\text{out}}=4096,\qquad r=16
$$

原矩阵：

$$
4096\times4096=16,777,216
$$

LoRA：

$$
16\times4096+4096\times16=131,072
$$

只相当于这个线性层原始参数量约：

$$
\frac{131072}{16777216}\approx0.78\%
$$

注意：这是**被 LoRA 覆盖的这个矩阵**的比例，不等于整个模型最终 trainable parameter ratio，一切还取决于你给哪些层加 LoRA。

**4. $r$ 到底是什么意思？**

$r$ 是低秩通道的维度：

```text
d_in=4096
   ↓
A
   ↓
r=16
   ↓
B
   ↓
d_out=4096
```

可以把它理解成：

> 原本允许 $\Delta W$ 在一个巨大的参数空间里自由变化；LoRA 强制更新通过一个宽度只有 $r$ 的低维“瓶颈”。

$r$ 越大：

- 可训练参数更多；
- 表达能力通常更强；
- optimizer / gradient 显存更大；
- 但性能并不会保证随 $r$ 单调提高。

LoRA 论文的动机来自对 fine-tuning update 的 **low intrinsic rank / rank deficiency** 的实证观察，但：

> **“所有下游任务的最优更新一定低秩”并不是数学定理。**

**5. `lora_alpha` 是什么？**

常见 LoRA forward：

$$
h
=
Wx+\frac{\alpha}{r}BAx
$$

因此：

$$
s=\frac{\alpha}{r}
$$

控制 LoRA 分支相对原始分支的缩放强度。

不要把：

- $r$：低秩维度 / 容量；
- $\alpha$：LoRA 更新的 scaling；

混成同一个概念。

某些现代 LoRA 变体会使用不同 scaling，例如 Rank-Stabilized LoRA 使用 $\alpha/\sqrt{r}$，但 Day7 面试首先把标准 LoRA 的 $\alpha/r$ 说清楚即可。

**6. 为什么初始化时经常让一边为 0？**

希望刚插入 LoRA 时：

$$
\Delta W=BA=0
$$

这样模型初始输出与原 Base Model 基本一致，再从预训练模型行为附近开始学习。

常见实现中会让一个矩阵随机初始化、另一个初始化为 0，从而保证初始 LoRA 分支是 no-op。

**7. LoRA 一般加在哪里？**

Decoder Block 中有很多线性投影：

```text
Attention:
q_proj
k_proj
v_proj
o_proj

MLP:
gate_proj
up_proj
down_proj
```

LoRA 可以只加在部分模块，也可以覆盖更多线性层。

早期常见配置会重点 target：

```text
q_proj
v_proj
```

现代 LLM 微调也经常覆盖：

```text
q_proj, k_proj, v_proj, o_proj
```

甚至包括 MLP；QLoRA-style 配置常见：

```python
target_modules="all-linear"
```

因此：

> **“LoRA 必须只插 Q、V”是错误说法。**

真正的 `target_modules` 是一个性能、显存、训练成本之间的设计选择。

**8. LoRA 为什么省显存？要按显存四部分回答**

这是最重要的面试题之一。

**模型参数**

Base $W$ 仍然要参与 forward，所以普通 LoRA 并没有让 Base Model 消失：

```text
Frozen Base Weights：仍在显存
LoRA A/B：额外少量参数
```

**Gradient**

冻结的 Base parameters 不需要保存用于参数更新的完整梯度；只需要 LoRA A/B 的参数梯度。

**Optimizer State**

AdamW 的 $m,v$ 只需要为可训练 LoRA 参数维护，而不用给数十亿 Base parameters 都维护。

**Activation**

LoRA 本身不会神奇地消除 Transformer activation。长 sequence、大 batch 时，activation 仍可能成为大头，所以仍然可能需要：

- gradient checkpointing；
- FlashAttention；
- sequence packing；
- 更小 micro-batch；
- Day6 学过的 gradient accumulation。

所以更准确的说法是：

> **LoRA 最大幅度压缩的是“可训练参数相关”的梯度与 optimizer state；Base Model 权重仍需存储，activation 也依然存在。**

这也自然引出第四轮：

> 如果连 Frozen Base Weights 都嫌太大怎么办？——把它量化，这就是 QLoRA 的关键一步。

**9. 推理时为什么可以 Merge LoRA？**

因为：

$$
W'=W+\frac{\alpha}{r}BA
$$

训练结束后，可以提前计算：

$$
W_{\text{merged}}
=
W+\frac{\alpha}{r}BA
$$

之后 forward 直接：

$$
y=W_{\text{merged}}x
$$

不必每次再跑独立的 $A\rightarrow B$ 分支。

因此标准 LoRA 可以做到：

> **训练时参数高效，部署时 merge 后基本不增加额外线性分支延迟。**

当然，如果想在同一个 Base Model 上动态切换多个 adapter，就通常保留 adapter 而不直接永久 merge。

![Pasted image 20260821035038](/my-blog/resources/uploads/obsidian-1787301359044-5.png)

**这一轮面试题**

| 面试问题 | 回答参考 |
| --- | --- |
| LoRA 的核心思想是什么？ | 冻结预训练权重 $W$，不直接学习完整的 $\Delta W$，而是用两个低秩矩阵参数化更新 $\Delta W=BA$。forward 变为 $Wx+\frac{\alpha}{r}BAx$，只训练 $A,B$，从而大幅减少 trainable parameters 及其梯度和 optimizer states。 |
| LoRA 是对原始权重 $W$ 做低秩分解吗？ | 不是。标准 LoRA 保留并冻结原始 $W$，低秩参数化的是任务适配所需的权重增量 $\Delta W$。这是常见面试陷阱。 |
| LoRA 为什么有效？ | LoRA 论文观察到大模型在下游适配时的有效权重更新具有较低的 intrinsic rank / 低维结构，因此不一定需要在完整参数空间中自由更新。低秩分支在大幅减少参数量的同时仍能提供足够的任务适配能力。不过这主要是经验与实证动机，不是所有任务都保证严格低秩。 |
| `lora_r` 越大是不是一定越好？ | 不一定。更大的 $r$ 增加低秩更新的表达能力，也增加参数、显存和计算，但性能可能在某个 rank 后饱和，甚至因数据量、正则化等因素恶化。应把 $r$ 当作容量超参数。 |
| `lora_alpha` 和 `r` 有什么区别？ | $r$ 决定低秩瓶颈维度和参数容量，`lora_alpha` 控制 LoRA 更新分支的 scaling。标准实现常使用 $\alpha/r$ 乘在 $BAx$ 上。 |
| LoRA 为什么能省显存？ | Base weights 仍存在，但被冻结，因此不需要为全部 Base parameters 保存参数梯度和 Adam optimizer states；这些训练状态只为很小的 LoRA 参数保存。Activation 通常不会因为 LoRA 自动大幅减少，因此长序列时仍需要 checkpointing、FlashAttention 等。 |
| LoRA 会让 Base Model 权重本身变小吗？ | 普通 LoRA 不会。Base weights 仍按原精度存储并参与 forward。要进一步降低 Base Weight memory，可以使用 QLoRA 等方法把冻结基座量化。 |
| LoRA 一定只放在 `q_proj` 和 `v_proj` 吗？ | 不一定。Q/V 是经典选择之一，但实际可 target Q/K/V/O、MLP 等线性层；现代 QLoRA-style 微调常覆盖 `all-linear`。具体取决于性能和资源预算。 |
| LoRA 为什么推理时可以不增加额外延迟？ | 因为线性 LoRA 更新可以在部署前 merge：$W_{\text{merged}}=W+\frac{\alpha}{r}BA$。之后直接用合并后的矩阵做一次线性层即可。如果需要动态切换 adapter，则可以不 merge。 |

## 第四轮：QLoRA 与完整 SFT 工程——4-bit 到底发生在哪里

**1. 普通 LoRA 还剩下什么显存问题？**

假设 Base Model 有 7B parameters。

即使完全冻结：

```text
7B Base Weights
```

也仍需要放在设备上参与 forward。

粗略只看权重：

```text
FP32：约 28 GB
FP16/BF16：约 14 GB
4-bit：理论裸权重量级约 3.5 GB
```

实际显存还会有量化 metadata、LoRA、activation、CUDA buffer 等，所以不能把上面的数字直接当最终训练显存。

QLoRA 的核心组合是：

```text
4-bit Frozen Base Model
           +
   Trainable LoRA
```

即：

> **LoRA 减少“需要训练的参数”；Quantization 减少“冻结 Base Weights 的存储”。**

**2. QLoRA 的 4-bit 到底量化了谁？**

最容易犯的错误：

> “QLoRA 把所有训练参数都变成 4-bit。”

更准确的是：

- Base Model 权重：以 4-bit 量化形式存储，冻结；
- LoRA 参数：仍以适合训练的较高精度保存和更新；
- 计算时：4-bit Base Weight 会按需要反量化到计算 dtype，例如 BF16，再进行矩阵计算；
- 梯度：通过量化基座的计算图传播到 LoRA，但不会更新冻结的 4-bit Base Weight。

概念流程：

```text
4-bit Quantized W
      ↓ 临时 dequantize
BF16 / FP16 compute
      ↓
Wx
 +
LoRA branch（trainable）
      ↓
output
      ↓
loss
      ↓
backward
      ↓
只更新 LoRA parameters
```

所以必须区分三个概念：

```text
Storage dtype ≠ Compute dtype ≠ Trainable parameter dtype
```

“权重以 4-bit 存储”并不等于“GPU Tensor Core 全程直接用 4-bit 完成所有反向传播”。

**3. NF4 为什么不是简单 INT4？**

4 bit 只有：

$$
2^4=16
$$

个离散表示状态。

如果简单做均匀量化，相当于把数轴等距离切成若干区间。但预训练权重通常并不是在数轴上均匀分布。

QLoRA 提出的 **NF4（NormalFloat 4）** 针对近似正态分布的权重设计量化点，使有限的 16 个值更适合这种分布。

Day7 不需要背 NF4 的全部编码表，只需要能说清：

> **NF4 不是“更多 bit”，仍是 4-bit；它优化的是 16 个量化值如何分布，以更适合近似正态的权重。**

**4. Double Quantization 是什么？**

分块量化通常不仅保存量化后的低 bit 权重，还要保存每个 block 的 scale / quantization constant。

于是出现：

```text
quantized weights 很小
但 quantization constants 也要占空间
```

Double Quantization 的思路是：

> **连第一次量化所需的量化常数也继续量化。**

即：

```text
W
↓ 第一次量化
4-bit weights + quantization constants
                   ↓ 再量化
            更紧凑的 constants
```

目的仍然是进一步减小平均显存占用。

**5. Paged Optimizer 在解决什么？**

QLoRA 论文中的 paged optimizer 利用统一内存 / paging 思路应对训练中的短时显存峰值，尤其是长序列和 gradient checkpointing 带来的 memory spike。

理解到这一层即可：

> 它不是 QLoRA “低秩数学公式”的一部分，而是避免某些瞬时峰值直接 OOM 的训练工程设计。

![Pasted image 20260821154450](/my-blog/resources/uploads/obsidian-1787301359044-6.png)


**6. LoRA 与 QLoRA 不要混成一个概念**

| 方法 | Base Weight | Trainable Parameters | 主要省哪里 |
| --- | --- | --- | --- |
| Full FT | FP16/BF16 等 | 全模型 | 基本不省 |
| LoRA | 通常 FP16/BF16 Frozen Base | LoRA A/B | gradient + optimizer states |
| QLoRA | 4-bit Quantized Frozen Base | LoRA A/B | 再进一步压缩 Base Weight memory |

一句话：

> **QLoRA = Quantized Frozen Base + LoRA adaptation。**

**7. 把 Day6 的训练知识全部接回来**

SFT / LoRA 训练依然需要 Day6 的：

$$
B_{\text{global}}
=
B_{\text{micro}}
\times N_{\text{GPU}}
\times N_{\text{accum}}
$$

比如：

```text
per_device_train_batch_size = 2 每个GPU放的样本
GPU = 4
gradient_accumulation_steps = 8 梯度积累次数
```

则：

$$
B_{\text{global}}=2\times4\times8=64
$$

依然需要：

```text
forward
→ loss
→ backward / gradient accumulation
→ unscale（混合精度时）
→ grad norm / clipping
→ optimizer.step()
→ scheduler.step()
→ zero_grad()
```

依然需要 checkpoint：

```text
LoRA adapter weights
optimizer states
scheduler state
global step
RNG / data progress（需要严格 resume 时）
```

因此 Day7 不是一条新路线，而是在 Day6 的训练框架里：

> **把“哪些参数训练、哪些参数冻结、权重用什么精度存储”换掉。**

**8. 一份现代 TRL + PEFT 的 SFT / LoRA 代码骨架**

下面不是让你死背 API，而是要看懂每个参数在前面哪一块理论里出现。

```python
import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer

# 1. 数据集应已经是框架支持的 conversational 或 prompt-completion 格式。
#    真实项目中最重要的工作往往是先检查数据质量、长度分布、模板和 EOS。
dataset = load_dataset("your_dataset", split="train")

# 2. LoRA 配置：冻结 Base Model，只给指定 Linear 层增加低秩可训练参数。
lora_config = LoraConfig(
    r=16,                       # 低秩维度：越大容量通常越强，但参数和显存也增加
    lora_alpha=32,              # LoRA scaling，标准形式常体现为 alpha / r
    lora_dropout=0.05,          # 对 LoRA 分支做少量 dropout，可视为一种正则化
    target_modules="all-linear",# QLoRA-style 常覆盖 Transformer 中的 Linear 层
    bias="none",
    task_type="CAUSAL_LM",
)

# 3. SFT 训练配置。
#    下面只是理解结构用的示例值，不应把超参数当固定答案。
training_args = SFTConfig(
    output_dir="./day7_sft_lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,  # Day6：通过多次 micro-batch 累积形成更大有效 batch
    learning_rate=1e-4,             # LoRA 只训练新参数，常允许比 full FT 更高的 LR
    num_train_epochs=2,
    warmup_ratio=0.03,
    max_length=2048,
    packing=True,                   # 减少短样本 padding 浪费
    assistant_only_loss=True,       # 对兼容模板的 conversational dataset 只监督 Assistant
    gradient_checkpointing=True,    # 用计算换 activation memory
    bf16=True,
    logging_steps=10,
    save_steps=200,
)

trainer = SFTTrainer(
    model="your-base-model",
    train_dataset=dataset,
    args=training_args,
    peft_config=lora_config,
)

trainer.train()
```

注意 `assistant_only_loss=True` 依赖 Chat Template 能正确提供 Assistant / generation 区域；不同模型和数据格式要查官方文档，不能看到这个参数就机械打开。

**9. QLoRA 的量化配置骨架**

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training

# Base Model 的 4-bit 存储配置。
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                  # Frozen Base Weight 以 4-bit 形式加载
    bnb_4bit_quant_type="nf4",          # QLoRA 推荐的 NF4 weight quantization
    bnb_4bit_use_double_quant=True,     # 连 quantization constants 也进一步压缩
    bnb_4bit_compute_dtype=torch.bfloat16, # 计算时使用 BF16，而不是“所有计算都是 4-bit”
)

model = AutoModelForCausalLM.from_pretrained(
    "your-base-model",
    quantization_config=bnb_config,
)

# 对 k-bit 模型做 PEFT 训练前的必要准备。
model = prepare_model_for_kbit_training(model)

# 后续再给 model 注入 LoRA，并交给 Trainer。
```

面试不要求你逐行背库函数，但应该能从代码反推：

```text
load_in_4bit
→ 压 Base Weight

nf4
→ 选择 4-bit 的量化表示

compute_dtype=bf16
→ 4-bit 是存储，不代表所有乘法和梯度都以 4-bit 做

LoRA
→ 真正可训练的参数
```

**10. Day7 最容易踩的工程坑**

- **Chat Template 不一致**：训练和推理使用不同模板，模型行为明显变差。
- **EOS 配错**：模型学不会结束，推理一直生成。
- **response 被 truncation**：有样本却几乎没有有效 supervised tokens。
- **错误把所有 token 都算 loss**：不是一定错，但可能偏离原本的 assistant-only 训练目标。
- **`target_modules` 写错**：模型模块名不匹配，LoRA 根本没插到预期层。
- **只看 trainable parameter ratio 判断显存**：忽略 Base Weight 和 Activation。
- **QLoRA 把 4-bit 当计算 dtype**：混淆 storage 与 compute。
- **训练 loss 很低就认为 SFT 成功**：还必须看 held-out instruction following、任务指标、格式遵循、生成质量和过拟合。
- **忘记检查实际可训练参数**：训练前应打印 trainable params，确认冻结 / 解冻是否符合设计。

![Pasted image 20260821162550](/my-blog/resources/uploads/obsidian-1787301359044-7.png)

**这一轮面试题**

| 面试问题 | 回答参考 |
| --- | --- |
| QLoRA 和 LoRA 的核心区别是什么？ | LoRA 冻结 Base Model、只训练低秩 adapter，但 Base Weight 通常仍以 FP16/BF16 等精度存储。QLoRA 在此基础上把 Frozen Base Weight 量化到 4-bit，大幅降低基座权重显存，同时仍通过高精度 LoRA 参数做训练。 |
| QLoRA 中所有东西都是 4-bit 吗？ | 不是。典型 QLoRA 中主要是 Frozen Base Weight 以 4-bit 量化形式存储；计算时会反量化到 BF16/FP16 等 compute dtype，LoRA 参数也使用适合训练的较高精度并接受梯度。必须区分 storage dtype、compute dtype 和 trainable parameter dtype。 |
| NF4 是什么，为什么适合 QLoRA？ | NF4 是 QLoRA 提出的 4-bit NormalFloat 表示，仍然只有 16 个离散状态，但这些量化值的设计更适合近似正态分布的预训练权重，相比简单均匀 4-bit 表示能更有效利用有限的量化级别。 |
| Double Quantization 在量化什么？ | 第一次量化权重时还需要保存每个 block 的 scale / quantization constants。Double Quantization 再进一步量化这些常数，从而继续降低平均内存开销。 |
| Paged Optimizer 的主要作用是什么？ | 它是 QLoRA 的训练工程优化，用 paging / unified memory 思路处理某些训练过程中的短时显存峰值，降低因为瞬时 memory spike 直接 OOM 的风险；它不是 LoRA 低秩数学公式的一部分。 |
| 为什么 QLoRA 的梯度能“穿过”4-bit Base Model？ | Base Weight 虽被冻结，但仍参与 differentiable forward。计算时量化权重会反量化到计算精度参与矩阵运算，梯度可以沿计算图传播到 LoRA 分支；只是 Base Weight `requires_grad=False`，不会用这些梯度更新量化参数。 |
| LoRA/QLoRA 后还需要 Gradient Accumulation 吗？ | 可能仍然需要。LoRA/QLoRA 主要降低参数相关显存，但长序列 activation 和 batch memory 仍可能很大。显存不足时依然可以减小 micro-batch，并用 gradient accumulation 保持较大的 global batch。 |
| 为什么说“trainable parameters 只有 0.5%”不能直接推出“显存只有原来的 0.5%”？ | 因为训练显存还包括 Frozen Base Weights、activation、临时 buffer 等。LoRA 只显著减少可训练参数对应的 gradients 和 optimizer states；普通 LoRA 的 Base Model 本身仍完整存在。 |
| SFT loss 降得很好，为什么生成质量仍可能很差？ | Training loss 只说明模型对训练分布 token 的拟合程度，不能完整代表 instruction following、格式遵循、泛化和生成质量。还可能存在模板错误、数据泄漏、过拟合或训练数据本身质量差，因此必须做独立生成评测与 held-out evaluation。 |
| 如果要从 Base Model 做一次 QLoRA SFT，你会按什么顺序检查？ | 先检查数据与 Chat Template、EOS、长度分布和 loss mask；再加载 4-bit Base、prepare k-bit training、配置 LoRA target modules；确认 trainable params；设置 micro-batch、gradient accumulation、LR/scheduler/checkpoint；训练时监控 loss、grad norm、吞吐和显存；最后用未见指令做生成与任务评测。 |

**Day7 最终复习：把今天压成 12 句话**

1. Pretraining 主要让模型学习语言、知识和 next-token 能力；SFT 主要把这些能力映射成 instruction-following 行为。
2. Decoder-only LLM 的 SFT 本质通常仍是 Causal LM / Next Token Prediction。
3. Chat Template 把 `system/user/assistant` 结构序列化成模型真正看到的 token 格式。
4. `attention_mask` 控制上下文可见性，`labels=-100` 控制某个位置是否进入 loss。
5. Assistant-only loss 中，Prompt 能被看到，但通常不直接贡献 token-level loss。
6. Packing 的核心价值是减少短 SFT 样本的 padding 浪费。
7. LoRA 冻结 $W$，低秩参数化的是更新 $\Delta W=BA$，不是删除或分解掉原始 $W$。
8. $r$ 控制低秩容量，$\alpha$ 控制 LoRA 分支的 scaling。
9. LoRA 主要节省可训练参数对应的 gradient 和 optimizer states；Base Weight 与 activation 仍然存在。
10. QLoRA 在 LoRA 上进一步把 Frozen Base Weight 以 4-bit 存储。
11. QLoRA 中“4-bit storage”不等于“所有计算和梯度都是 4-bit”；要区分 storage、compute、trainable dtype。
12. Day6 的 gradient accumulation、scheduler、checkpoint、FlashAttention 和训练监控，在 SFT / LoRA / QLoRA 中仍然全部有效。
