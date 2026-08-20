---
title: "『LLM学习笔记6』大模型训练工程：显存并行与稳定性"
category: "internship"
tags:
  - "LLM"
date: "2026-08-20"
summary: ""
pdf: ""
pdfTitle: ""
---

## 1. 从训练闭环到显存账本：Chinchilla、梯度累积、Scheduler 与 Checkpoint

**本轮定位**：Day 5 已经打通了 `原始文本 → token → logits → loss → backward → optimizer.step()`，也接触了 Chinchilla、梯度累积、scheduler 和 checkpoint，但后四者还没有真正形成工程直觉。Day 6 从这里继续：**先弄清一次 optimizer update 到底由多少数据构成、训练状态存在哪里、显存又被谁占满，之后再进入多卡并行。**

| 资源 | 今天读什么 | 阅读时盯住什么 |
|---|---|---|
| [Training Compute-Optimal Large Language Models（Chinchilla）](https://arxiv.org/abs/2203.15556) | Abstract、Introduction、Figure 1 与结论 | 固定 compute 下为什么不能只扩大参数；70B/1.4T 例子表达的是什么 |
| [Mixed Precision Training](https://arxiv.org/abs/1710.03740) | Abstract、FP32 master weights、loss scaling | FP16 为什么容易 underflow；混合精度为什么不等于“所有张量都减半” |
| [PyTorch Activation Checkpointing](https://docs.pytorch.org/docs/stable/checkpoint.html) | 开头原理说明即可 | 它保存的不是训练进度，而是用重算换激活显存 |
| [Hugging Face Optimizer Schedules](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules) | warmup、linear、cosine schedule | scheduler 的计数单位是什么；梯度累积后何时调用 `scheduler.step()` |

**先把 Chinchilla 放回正确位置。**

Day 5 已经讲过 Scaling Law：模型最终效果不只由参数量决定，还取决于训练 token 数和总计算量。Chinchilla 讨论的核心问题是：

> 在总训练计算预算固定时，参数量 $N$ 和训练数据量 $D$ 应怎样分配，才能把 loss 降得更低？

其重要结论不是“70B 一定最好”，也不是“任何模型都必须严格使用 20 tokens/parameter”，而是：**很多早期大模型把过多 compute 花在扩大参数上，却没有给模型足够多的训练 token，因而处于 undertrained 状态。**

经典对比是：

```text
Gopher      约 280B 参数，约 300B 训练 token
Chinchilla   约 70B 参数，约 1.4T 训练 token
```

两者训练计算量处于相近量级，但更小、训练数据更多的 Chinchilla 在大量任务上表现更好。由 $1.4T/70B\approx20$ 得到的“约 20 tokens/parameter”是一个著名记忆点，但只能当作原论文设定下的量级直觉。真实项目还会考虑：

```text
数据质量与重复率
模型架构与 tokenizer
是否更重视训练成本还是长期推理成本
目标领域是否需要重复训练高质量数据
训练结束后是否还会进行大量 post-training
```

Chinchilla 对 Day 6 的直接启发是：当训练规模确定后，我们需要把总 token budget 可靠地转化为一次次 optimizer update。这里就进入 **micro batch、梯度累积和 global batch**。

**1）Micro Batch、Gradient Accumulation 与 Global Batch**

假设单张 GPU 一次最多放入 2 条序列，但你希望一次参数更新综合 8 条序列的梯度，可以连续执行 4 个 micro batch：

```text
micro batch 1 → forward → backward → 梯度暂存
micro batch 2 → forward → backward → 梯度继续累加
micro batch 3 → forward → backward → 梯度继续累加
micro batch 4 → forward → backward → 梯度继续累加
                                      ↓
                                optimizer.step()
                                      ↓
                                  清空梯度
```

如果数据并行 GPU 数为 $G$，每张卡一次处理的 micro batch size 为 $B_\mu$，梯度累积步数为 $A$，那么一次 optimizer update 对应的 global batch size (梯度积累)是：

$$
B_{\mathrm{global}}
=
G\times B_\mu\times A
$$

固定序列长度为 $L$ 时，可以粗略写成：

$$
T_{\mathrm{global}}
=
G\times B_\mu\times A\times L
$$

但语言模型训练更严谨的统计应当是**有效 token 数**。如果不同 micro batch 的 padding 比例不同，简单用 `序列数 × 最大长度` 会高估真实监督 token。

这里：

- **Micro Batch Size = 2**：GPU 一次实际处理几条数据。
- **Gradient Accumulation = 4**：连续算 4 次，但中间**不更新模型**，只攒梯度。
- **Global Batch Size = 8**：模型每次真正更新参数之前，总共参考了多少条数据。


梯度累积的标准骨架是：

```python
optimizer.zero_grad()

for micro_step, batch in enumerate(dataloader):
    outputs = model(**batch)

    # A 个 micro batch 的平均梯度，避免梯度整体放大 A 倍。
    loss = outputs.loss / accumulation_steps
    loss.backward()

    if (micro_step + 1) % accumulation_steps == 0:
        # 混合精度时应先 unscale，再做 gradient clipping。
        clip_grad_norm_if_needed()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
```

这里有四个必须说清的点：

```text
1. backward 每个 micro batch 都执行；optimizer.step() 每 A 个 micro batch 才执行一次。
2. scheduler 通常跟 optimizer update 走，而不是每个 micro batch 都走。
3. gradient clipping 应作用于累积完成后的总梯度，而不是分别裁每个 micro batch。
4. PyTorch 默认梯度会累加，因此累积期间不能提前 zero_grad。
```

- `backward()`：每个 micro batch 都执行。PyTorch 默认会累积梯度，因此最终相当于得到 $g=g_1+g_2+g_3+g_4$。注意，`backward()` 只是计算梯度，并不会修改模型参数。
- `optimizer.step()`：累积完 $A$ 个 micro batch 后执行一次，真正根据累积梯度更新模型参数。因此 `backward × 4`，但 `optimizer.step() × 1`。
- `scheduler.step()`：通常跟着 `optimizer.step()` 走，而不是每个 micro batch 都执行。因为 scheduler 一般按照**实际参数更新次数**调整学习率。
- `gradient clipping`：应该作用在累积完成后的总梯度 $g$ 上，而不是分别裁剪每个 micro batch 的梯度，因为真正用于参数更新的是最终累积梯度。
- `zero_grad()`：不能在梯度累积过程中提前调用。PyTorch 的梯度默认是累加的，如果中途 `zero_grad()`，前面已经计算出的梯度会被清空。应在一次 `optimizer.step()` 完成后再清梯度。

| 操作                  | 每个 micro batch | 每次 optimizer update |
| ------------------- | -------------- | ------------------- |
| `forward()`         | ✅              |                     |
| `backward()`        | ✅              |                     |
| `gradient clipping` |                | ✅                   |
| `optimizer.step()`  |                | ✅                   |
| `scheduler.step()`  |                | ✅                   |
| `zero_grad()`       |                | ✅                   |


**梯度累积是否严格等价于一个更大的 batch？**

在以下条件接近满足时，两者的优化语义近似一致：

```text
所有样本相同；累积后只更新一次参数；loss 的归一化正确；
中途不改变学习率；模型没有依赖 batch 统计量的层；
随机性和浮点归约误差可以忽略。
```

Transformer 通常使用 LayerNorm/RMSNorm，不像 BatchNorm 那样依赖整个 batch 的统计量，因此梯度累积更容易逼近大 batch。但仍可能因 dropout 随机数、浮点加法顺序、分布式归约顺序产生细小差异。

还有一个容易被忽略的语言模型细节：许多 CausalLM 返回的是“当前 micro batch 内所有有效 token 的平均 loss”。若不同 micro batch 的有效 token 数相差很大，直接对每个 micro batch 的平均 loss 做等权平均，就不完全等价于对所有 token 统一求平均。更严格的做法是累计：

```text
所有有效 token 的 loss 总和
÷
所有有效 token 的数量
```

这也是为什么大规模训练日志中经常直接记录 `tokens/update`，而不只记录 `sequences/update`。

**2）Learning Rate Scheduler 到底调度什么？**

Optimizer 决定“根据当前梯度怎样更新参数”，scheduler 决定“当前 optimizer update 应使用多大的学习率”。大模型中常见轨迹是：

```text
较小学习率起步
      ↓
Linear Warmup
      ↓
达到 Peak LR
      ↓
Cosine / Linear Decay
      ↓
训练末期使用较小步长
```

一种常见的 warmup + cosine 形式是：

$$
\eta_s=
\begin{cases}
\eta_{\max}\dfrac{s}{W}, & 0\le s<W \\
\eta_{\min}+\dfrac{1}{2}(\eta_{\max}-\eta_{\min})
\left[1+\cos\left(\pi\dfrac{s-W}{T-W}\right)\right], & W\le s\le T
\end{cases}
$$

其中：

- $s$：当前 optimizer step；
- $W$：warmup steps；
- $T$：总 optimizer steps；
- $\eta_{\max}$：峰值学习率；
- $\eta_{\min}$：训练末期学习率。

Warmup 的直觉不是“前几步模型完全不会训练”，而是训练刚开始时参数更新方向、激活尺度和 Adam 的动量统计尚未稳定，突然使用很大的峰值学习率容易产生过强更新。后期 decay 则让模型在接近较优区域时逐渐减小步长。

![Pasted image 20260820160814](/my-blog/resources/uploads/obsidian-1787231837489-1.png)

**最重要的工程坑：梯度累积后，scheduler 的 step 数通常指 optimizer step。**

假设：

```text
总共读取 100,000 个 micro batch
gradient accumulation = 10
```

那么 optimizer update 只有约：

```text
10,000 steps
```

若错误地把 scheduler 的 `num_training_steps` 配成 100,000，学习率曲线会慢 10 倍；若每个 micro batch 都调用一次 `scheduler.step()`，学习率又会提前衰减完。

- `Micro Batch`：微批次 / 小批次；`Micro Batch Size`：单个微批次大小；`Batch`：批次；`Batch Size`：批次大小；`Global Batch`：全局批次；`Global Batch Size`：全局批次大小
- `Gradient`：梯度；`Gradient Accumulation`：梯度累积；`Gradient Accumulation Steps`：梯度累积步数；`Gradient Clipping`：梯度裁剪；`Gradient Norm`：梯度范数
- `Forward` / `Forward Pass`：前向传播；`Backward` / `Backward Pass`：反向传播
- `Optimizer`：优化器；`Optimizer Step`：优化器更新步 / 参数更新步；`optimizer.step()`：执行一次参数更新；`Update`：更新；`Parameter Update`：参数更新
- `zero_grad()`：清空梯度 / 梯度归零
- `Scheduler`：学习率调度器；`scheduler.step()`：学习率调度器更新一步

**3）Checkpoint 有两种完全不同的含义**

第一种是 **training checkpoint**：把训练状态持久化到磁盘，以便故障后继续训练。

一个可恢复训练的 checkpoint 通常不只包含 model weights：

| 状态 | 为什么需要 |
|---|---|
| Model parameters | 恢复模型当前学到的权重 |
| Optimizer states | AdamW 的一阶矩、二阶矩等决定下一步怎样更新 |
| Scheduler state | 恢复当前学习率所处位置，避免 LR 突然跳变 |
| Global optimizer step / token count | 恢复日志、调度、保存和数据进度 |
| AMP GradScaler state | FP16 动态 loss scaling 时维持数值状态 |
| CPU / CUDA RNG states | 尽量恢复 dropout、采样等随机过程 |
| Data sampler / data cursor | 避免恢复后大量重复或跳过训练数据 |
| Model config、tokenizer 与训练配置 | 确保恢复环境与原训练一致 |

只保存 weights 更适合推理或“从这组权重重新开始一种新训练”；若目标是无缝 resume，optimizer、scheduler、step 和数据位置都非常关键。

第二种是 **activation checkpointing / gradient checkpointing**：它不是把文件存到磁盘，而是 forward 时不保存部分中间激活，backward 需要时重新计算这些激活。

```text
普通训练：
forward 保存大量 activations → backward 直接读取

Activation Checkpointing：
forward 只保存少量边界状态 → backward 重跑部分 forward → 再计算梯度
```

因此它的本质是：

```text
减少 activation memory
代价是增加重计算与 step time
```

面试时只说“checkpoint 能断点续训”或只说“checkpoint 用计算换显存”都不完整，必须先说明你指的是哪一种 checkpoint。

**4）训练显存到底被谁占用？**

大模型训练显存可以先拆成六类：

```text
Training Memory
├── Parameters
├── Gradients
├── Optimizer States
├── Activations 激活值 / 中间激活
├── Temporary / Kernel Buffers 临时缓冲区 / 算子缓冲区
└── Communication Buffers 通信缓冲区
```

假设一个 7B 参数模型仅以 BF16 存储权重：

$$
7\times10^9\times2\text{ bytes}
\approx14\text{ GB}
$$

但 14 GB 只代表“低精度参数副本”，不代表训练总显存。以常见 Adam 混合精度实现为例，每个参数还可能对应：

```text
BF16/FP16 参数       2 bytes
低精度或高精度梯度   2～4 bytes
FP32 master weight   4 bytes（取决于实现）
Adam 一阶矩 m         4 bytes
Adam 二阶矩 v         4 bytes
```

所以常见估算可能落在每参数约 12～18 bytes 的范围，7B 模型仅模型状态就可能需要约 84～126 GB，尚未计算激活、临时 buffer 和通信 buffer。具体数字取决于框架是否保留 master weight、梯度精度和 optimizer 实现，不能机械套一个固定常数。

Activation memory 又大致随以下因素增加：

```text
micro batch size ↑
sequence length ↑
hidden size ↑
layer 数 ↑
保存的中间张量 ↑
```

因此：

- 梯度累积通过减小单次 micro batch，主要降低单次 forward/backward 的激活显存；
- activation checkpointing 通过少存激活进一步省显存；
- 二者都不会自动消除完整参数、梯度和 Adam 状态的显存；
- ZeRO/FSDP 才会在多卡之间切分这些模型状态。

**5）FP32、FP16 与 BF16**

| 格式 | 指数位 | 尾数位 | 主要特点 |
|---|---:|---:|---|
| FP32 | 8 | 23 | 范围和精度都高，但计算和存储成本大 |
| FP16 | 5 | 10 | 精度尚可但动态范围窄，容易 overflow/underflow |
| BF16 | 8 | 7 | 动态范围接近 FP32，精度低于 FP16，通常更适合大模型训练 |

FP16 梯度中一些非常小的值可能直接下溢成 0，因此经典 mixed precision 会使用 loss scaling：

```text
原 loss × 较大 scale
      ↓ backward
梯度整体进入 FP16 可表示范围
      ↓ optimizer.step 前
梯度再除以 scale
```

若检测到 Inf/NaN，动态 scaler 会跳过本次更新并调整 scale。BF16 保留了与 FP32 相同数量的指数位，通常不需要依赖 loss scaling 才能覆盖大范围数值，但很多敏感算子、归约和 optimizer state 仍会保留 FP32。**混合精度不是粗暴地把整个训练脚本全部 cast 成半精度，而是让不同算子使用适合的精度。**

![Pasted image 20260818182646](/my-blog/resources/uploads/obsidian-1787231837489-2.png)

**本轮面试题**

| 面试题                                                       | 面试场景回答                                                                                                                                                                                                                            |
| --------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Chinchilla 的核心结论是什么？**                                  | Chinchilla 研究固定训练 compute 下参数量和训练 token 的最优分配。它发现很多早期大模型参数扩得很快，但数据量没有同步增长，因此处于 undertrained 状态。经典例子是约 70B 参数、1.4T token 的 Chinchilla，在相近 compute 下超过约 280B 的 Gopher。核心启发是参数和数据应平衡扩展，而不是只堆参数；约 20 tokens/parameter 只是原论文量级，不是普适定律。 |
| **Gradient Accumulation 是什么？**                            | 它把一个较大的 effective batch 拆成多个能放进显存的 micro batch。每个 micro batch 都 forward/backward，但暂不更新参数，让梯度在 `.grad` 中累积；累计 $A$ 次后才执行一次 clipping、`optimizer.step()`、`scheduler.step()` 和 `zero_grad()`。它主要降低单次激活显存，代价是一次参数更新需要更多次前后向。            |
| **梯度累积为什么常把 loss 除以 accumulation steps？**                 | 因为 PyTorch 的 backward 默认把梯度相加。如果每个 micro batch 的平均 loss 都直接 backward，累计 $A$ 次后梯度通常会放大约 $A$ 倍。对 loss 除以 $A$ 可以得到 micro-batch 平均意义下与大 batch 更接近的梯度。不过变长序列下还要考虑每个 micro batch 的有效 token 数，严格做法是按 token 数加权。                          |
| **梯度累积完全等价于增大 batch size 吗？**                             | 不保证严格等价。参数只在累积结束后更新、归一化正确且模型不依赖 batch 统计时，两者通常近似。差异可能来自 dropout 随机性、浮点归约顺序、变长样本的 token 加权，以及 BatchNorm 等依赖 batch 统计的层。Transformer 多用 LayerNorm/RMSNorm，因此通常比含 BatchNorm 的网络更接近。                                                   |
| **用了梯度累积后 scheduler 应该多久 step 一次？**                       | 通常每次真正的 `optimizer.step()` 后调用一次，而不是每个 micro batch 调一次。Scheduler 的 total steps、warmup steps 也一般按 optimizer update 计算。否则学习率可能提前衰减完，或者整条曲线被错误拉长。恢复 checkpoint 后还要同时恢复 scheduler state 和 global optimizer step。                      |
| **为什么大模型训练需要 warmup？**                                    | 训练初期激活、梯度和 Adam 动量统计还没有稳定，直接使用峰值学习率可能产生过大的参数更新，引发 loss spike 或发散。Warmup 先从较小 LR 逐渐升到峰值，之后再 decay。它不是理论上永远必需，但在深层 Transformer、大 batch 和混合精度训练中是非常常见的稳定化手段。                                                                         |
| **Training checkpoint 至少应保存什么？**                          | 要无缝续训，至少应保存 model、optimizer、scheduler、global step；FP16 训练还要保存 GradScaler。为了尽量复现，还应保存 CPU/CUDA RNG state、数据 sampler 或 data cursor、训练配置和 tokenizer。只保存 model weights 可以继续做新训练或推理，但不能保证优化轨迹连续。                                       |
| **Training checkpoint 和 activation checkpointing 有什么区别？** | Training checkpoint 是把训练状态持久化到磁盘，用于故障恢复；activation checkpointing 是 forward 少存中间激活，backward 时重算，以计算换显存。二者只是名字相似，解决的问题完全不同：前者解决训练进度丢失，后者解决 activation memory 过大。                                                                    |
| **为什么 7B BF16 模型训练显存远大于 14 GB？**                          | 14 GB 只算了 7B 个 BF16 参数。训练还需要梯度、Adam 的一阶/二阶矩、可能存在的 FP32 master weights、激活、临时算子 buffer 和通信 buffer。常见 Adam 混合精度模型状态可能约为每参数 12～18 bytes，因此仅模型状态就可能超过 80 GB，长序列激活还会继续增加。                                                               |
| **FP16 与 BF16 的主要区别是什么？**                                 | FP16 尾数更多、局部精度稍高，但指数位少、动态范围窄，梯度容易 overflow/underflow，因此常配合 loss scaling。BF16 指数位与 FP32 相同，动态范围大得多，更适合大模型训练，但尾数位较少。实践中敏感归约、LayerNorm 或 optimizer state 往往仍使用 FP32。                                                                 |

## 2. 从单卡到多卡：DDP、All-Reduce、ZeRO 与 FSDP

**本轮目标**：理解“多放几张 GPU”并不会自动解决问题。你需要知道每一种并行策略到底切分了什么、复制了什么、何时通信，以及它解决的是吞吐问题还是单卡显存问题。

| 资源 | 今天读什么 | 阅读时盯住什么 |
|---|---|---|
| [PyTorch：What is Distributed Data Parallel](https://docs.pytorch.org/tutorials/beginner/ddp_series_theory.html) | DDP 基本流程与 DistributedSampler | 每个 rank 为什么有完整模型；梯度如何同步 |
| [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) | Abstract、ZeRO-DP 三阶段、内存分析图 | Stage 1/2/3 分别切什么；显存和通信的 trade-off |
| [PyTorch FSDP2 Tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) | How FSDP works | 参数何时 all-gather、梯度怎样 reduce-scatter、何时 reshard |
| [PyTorch Distributed Checkpoint](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html) | How DCP works | 分片权重为什么不能总用单个 `torch.save`；如何并行保存与重新分片 |

**1）Data Parallel：每张卡处理不同数据**

最直观的数据并行是：

```text
GPU 0：完整模型副本 θ₀ + micro batch 0
GPU 1：完整模型副本 θ₁ + micro batch 1
GPU 2：完整模型副本 θ₂ + micro batch 2
GPU 3：完整模型副本 θ₃ + micro batch 3
```

初始时各卡模型参数相同。每张卡独立执行 forward，得到自己的 loss；backward 时各卡得到局部梯度：

$$
g_r=\nabla_\theta\mathcal L_r
$$

随后通过梯度同步得到全局梯度，例如平均形式：

$$
\bar g
=
\frac{1}{G}\sum_{r=1}^{G}g_r
$$

每张卡都用同一个 $\bar g$ 更新参数，于是更新后模型仍保持一致。

PyTorch DistributedDataParallel（DDP）的核心就是：

```text
每个 rank 拥有完整模型副本
每个 rank 读取不重叠的数据分片
backward 期间对梯度 bucket 执行 All-Reduce
所有 rank 得到一致梯度并各自 optimizer.step()
```

**All-Reduce 的直觉**可以拆成两步：先把所有 rank 的梯度求和，再把结果发回所有 rank。真实通信库会使用 ring、tree 等算法，不一定真的先集中到某一张卡。

DDP 的核心通信通常发生在 backward，并可以把某些梯度 bucket 的通信与后续反向计算重叠。严谨地说，forward 也可能包含 buffer broadcast、同步 BatchNorm 或模型中特殊分布式算子的通信，因此不要绝对回答“DDP forward 完全没有通信”。

**DDP 解决什么，不解决什么？**

```text
它擅长：
并行处理更多数据，提高总吞吐；扩大 global batch。

它不擅长：
让一个本来放不进单卡的完整模型突然放进单卡。
```

原因很直接：每张卡仍然保存完整的 parameters、gradients 和 optimizer states。增加 GPU 只增加了模型副本数量，并没有切分单卡模型状态。

**2）DDP 与 Gradient Accumulation 怎样配合？**

若每个 micro batch backward 都立即 All-Reduce，那么累计 $A$ 次就会通信 $A$ 次。DDP 提供 `no_sync()` 思路：前 $A-1$ 个 micro batch 只在本地累积梯度，最后一个 micro batch 才同步。

```text
micro 1：local backward，不 All-Reduce
micro 2：local backward，不 All-Reduce
micro 3：local backward，不 All-Reduce
micro 4：backward + All-Reduce
         ↓
     optimizer.step()
```

这样可以避免无意义的重复通信。要注意，`no_sync()` 通常需要包住 forward 和 backward，而不仅仅包住 `loss.backward()`，因为 DDP 会在 forward 阶段准备梯度同步相关状态。


![Pasted image 20260820181119](/my-blog/resources/uploads/obsidian-1787231837489-3.png)

**DDP 解决“多张卡一起算”；Gradient Accumulation 解决“多算几批再更新”。**


**3）ZeRO：消除数据并行中的模型状态冗余**

在普通 DDP 中，假设有 8 张卡：

```text
同一份 Adam m/v 被复制 8 次
同一份完整 gradient 被复制 8 次
同一份完整 parameter 被复制 8 次
```

这些副本对保证同步训练有用，但从显存角度高度冗余。ZeRO 的思想是：**仍保持数据并行语义，但把原本每卡完整复制的模型状态分片到不同 rank。**

| 方法 | 每张卡完整保留什么 | 被分片的状态 | 主要效果 |
|---|---|---|---|
| DDP | 参数、梯度、optimizer states | 无 | 提高数据吞吐，但单卡状态显存不下降 |
| ZeRO-1 | 参数、梯度 | Optimizer states | 先消除 Adam m/v 等冗余 |
| ZeRO-2 | 参数 | Optimizer states + Gradients | 进一步切分梯度 |
| ZeRO-3 | 当前计算所需参数会临时聚合 | Optimizer states + Gradients + Parameters | 模型参数本身也常驻分片，单卡显存下降最大 |


**ZeRO-1**：每个 rank 只保存一部分 optimizer states。更新时各 rank 负责更新自己那部分参数，再把更新后的参数同步给其它 rank。

**ZeRO-2**：在 Stage 1 基础上，梯度也分片。某个 rank 只长期保存自己负责参数对应的梯度，不需要每张卡都保留完整 gradient tensor。

**ZeRO-3**：参数也分片。某一层即将计算时，各 rank 通过 All-Gather 临时获取该层所需的完整参数；计算结束后再释放或重新分片。Backward 时还需要聚合参数并通过 Reduce-Scatter 把梯度分发给负责的 rank。

所以 ZeRO 的核心 trade-off 是：

```text
模型状态复制减少 → 单卡显存下降
但参数/梯度需要按需聚合与分发 → 通信更频繁
```

不能简单说“ZeRO-3 一定最快”。当模型本来就能放进单卡、网络带宽有限、micro batch 很小时，额外通信可能让吞吐明显下降。

**4）FSDP 与 ZeRO-3 是什么关系？**

PyTorch Fully Sharded Data Parallel（FSDP）和 ZeRO-3 在核心内存思想上非常接近：都把 parameters、gradients、optimizer states 分片，并在计算某个模块时临时 unshard 参数。

概念流程可以写成：

```text
平时：每张卡只持有参数 shard
  ↓
某个 FSDP unit 即将 forward
  ↓ All-Gather
临时得到该 unit 的完整参数
  ↓
执行 forward
  ↓ Reshard / 释放完整参数
  ↓
backward 前再次按需 All-Gather
  ↓
计算梯度并 Reduce-Scatter
  ↓
每张卡只留下自己的 gradient shard
```

FSDP 的包裹粒度很重要：

- 包得太粗：一次 all-gather 的完整参数过大，峰值显存高；
- 包得太细：通信次数和框架开销过多；
- 常见做法：按 Transformer block 自动 wrap，在显存与通信之间平衡。

**不要把 FSDP 说成“PyTorch 给 ZeRO-3 换了个名字”。**它们理念相近，但 API、状态管理、通信调度、offload、prefetch 和工程实现并不完全相同。面试回答可以说：FSDP 的 `FULL_SHARD` 大体对应 ZeRO-3 的“参数、梯度、优化器状态全分片”思路。

![Pasted image 20260820181921](/my-blog/resources/uploads/obsidian-1787231837489-4.png)

分片（sharding）就是：把原本每张 GPU 都完整保存的一份东西拆成几块，让不同 GPU 各自只保存其中一部分。

**5）DDP、ZeRO/FSDP 与 Model Parallel 的边界**

ZeRO-3/FSDP 能让完整模型状态不再常驻单卡，但计算某个 FSDP unit 时，该 unit 的参数仍要临时出现在设备上。如果出现下面情况：

```text
单个 Transformer block 就太大
单次矩阵乘法的输入/输出激活就太大
单层计算时间需要跨 GPU 并行
```

仅靠状态分片仍可能不够，这时需要 Tensor Parallel 或 Pipeline Parallel。可以先记一个判断：

```text
DDP：切数据
ZeRO/FSDP：切数据并行中的模型状态
Tensor Parallel：切一层内部的矩阵计算
Pipeline Parallel：切不同层
Context Parallel：切长序列
```

**6）分片训练的 Checkpoint 为什么更复杂？**

普通单卡模型可以直接：

```text
一个 state_dict → 一个 checkpoint 文件
```

FSDP/ZeRO-3 下，不同 rank 持有不同参数和 optimizer shard。常见 checkpoint 形态包括：

| 形式 | 特点 | 适用场景 |
|---|---|---|
| Full state dict | 聚合成普通完整权重，易于推理与发布，但保存时内存/通信压力大 | 导出最终模型、跨框架使用 |
| Sharded state dict | 各 rank 并行保存自己的 shard，速度快、峰值低 | 大规模训练中间 checkpoint |
| Local state | 强依赖原并行拓扑和具体 rank | 特定框架内部快速恢复 |

现代 distributed checkpoint 工具可以让多 rank 并行写入，并在加载时针对新的 world size 重新分片。但即使参数能够重新分片，数据顺序、通信归约和随机数路径也可能变化，所以“换 GPU 数量恢复”不一定能做到 bitwise identical。

![Pasted image 20260820183002](/my-blog/resources/uploads/obsidian-1787231837489-5.png)

**本轮面试题**

| 面试题                                  | 面试场景回答                                                                                                                                                                          |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **DDP 的基本训练流程是什么？**                  | 每个 rank 保存一份完整模型，并通过 DistributedSampler 读取不同数据。各 rank 独立 forward，backward 时 DDP 对梯度 bucket 做 All-Reduce，使所有 rank 获得一致的平均梯度，随后各自执行相同的 optimizer update。因为初始参数和更新梯度一致，各卡模型始终保持同步。 |
| **DDP 的 forward 完全没有通信吗？**           | 核心梯度同步确实集中在 backward，但不能绝对说 forward 没通信。DDP 可能广播 buffer，SyncBatchNorm 或模型内其它分布式层也会通信。更准确的回答是：标准 DDP 的主要数据并行通信是 backward 中的 gradient All-Reduce，并常与反向计算重叠。                       |
| **All-Reduce 和 All-Gather 有什么区别？**   | All-Reduce 对各 rank 的张量做求和、平均等归约，并把归约结果返回给所有 rank，DDP 用它同步梯度。All-Gather 则把每个 rank 的不同 shard 收集起来，使所有 rank 得到拼接后的完整张量，ZeRO-3/FSDP 常在计算某层前用它临时恢复完整参数。                              |
| **为什么 DDP 不能让放不进单卡的模型直接训练？**         | 因为每张卡都要保存完整 parameters、gradients 和 optimizer states。DDP 切的是 batch，不切模型状态，所以它主要提高吞吐，而不是降低单卡模型状态显存。模型本身放不下时需要 ZeRO/FSDP 或模型并行。                                                    |
| **ZeRO-1、2、3 分别切分什么？**               | ZeRO-1 只分片 optimizer states；ZeRO-2 再分片 gradients；ZeRO-3 进一步分片 parameters。Stage 越高，单卡常驻模型状态越少，但需要更多 All-Gather、Reduce-Scatter 等通信。回答时最好同时说明“切了什么”和“代价是什么”。                       |
| **ZeRO-3 为什么 forward 时还能够计算完整线性层？**  | 参数虽然平时分散在不同 rank，但某个模块计算前会通过 All-Gather 临时收集该模块所需的完整参数。计算完成后完整参数可以再次 reshard 或释放。它不是永远把完整模型放在单卡，而是按模块、按时间短暂聚合。                                                                  |
| **FSDP 与 ZeRO-3 有什么异同？**             | 两者都通过分片参数、梯度和 optimizer state 降低单卡显存，计算模块前按需 all-gather，梯度通常 reduce-scatter。区别在于具体 API、wrap 粒度、状态字典、prefetch/offload 和运行时实现。可以说 FSDP FULL_SHARD 与 ZeRO-3 思路相近，但不能说完全相同。         |
| **什么时候优先选 DDP，什么时候选 FSDP？**          | 模型和单卡 micro batch 能放进单卡时，DDP 通常更简单、通信模式更轻，适合追求吞吐。若模型状态占用过大、单卡无法承载，FSDP/ZeRO 可分片参数、梯度和 Adam states。选择还要看互联带宽、checkpoint 生态和实现复杂度。                                                |
| **DDP 下梯度累积为什么需要 `no_sync()`？**      | 不使用 `no_sync()` 时，每个 micro batch backward 都会触发 All-Reduce，但真正 optimizer step 只在最后发生，前面的同步是多余通信。前 $A-1$ 次在本地累积，最后一次再同步，可以保持更新语义同时减少通信。                                           |
| **为什么分片 checkpoint 不能总当普通权重文件直接加载？** | 因为每个 rank 可能只保存参数和 optimizer state 的一部分，单个 shard 并不是完整模型。需要相同框架按原拓扑加载，或借助 distributed checkpoint 在读取时重新分片；若要发布普通模型，通常还要导出 full state dict。                                      |

## 3. 当模型状态分片仍不够：Tensor、Pipeline、Sequence/Context Parallel 与 FlashAttention

**本轮目标**：理解真正的大模型训练通常不是只选一种并行，而是把多种并行组合成设备网格。与此同时，长序列 Attention 的瓶颈不仅是 FLOPs，还包括 GPU 内存层级之间的数据搬运。

| 资源 | 今天读什么 | 阅读时盯住什么 |
|---|---|---|
| [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) | Model Parallel Transformers、MLP 与 Attention 切分图 | Column Parallel 与 Row Parallel 怎样配对；通信放在哪里 |
| [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473) | Tensor/Pipeline/Data Parallel 组合、pipeline schedule | 为什么并行策略要按网络拓扑组合；pipeline bubble 从哪里来 |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | Abstract、IO-Awareness、算法总览图 | 为什么它是 exact attention；为什么减少 HBM 访问能加速 |
| [PyTorch Context Parallel Tutorial](https://docs.pytorch.org/tutorials/unstable/context_parallel.html) | Introduction 即可 | 长上下文为什么要切 sequence；它与切参数不是一回事 |

**1）为什么 ZeRO/FSDP 之后还需要 Model Parallel？**

ZeRO/FSDP 主要减少“模型状态在每张数据并行卡上的冗余”，但它并没有自动把一个矩阵乘法拆到多张卡上。以下场景仍可能需要模型并行：

```text
某个 Transformer block 临时 all-gather 后仍放不下
单层矩阵本身过大，希望多卡同时计算
模型层数很多，单卡保存这一段层的激活仍过大
上下文太长，单卡 sequence activation 无法承载
```

最常见的四个维度是：

| 并行方式                  | 切分对象               | 主要解决问题               | 主要通信                            |
| --------------------- | ------------------ | -------------------- | ------------------------------- |
| Data Parallel（DP）     | batch / data       | 提高总吞吐                | 梯度 All-Reduce 或状态分片通信           |
| Tensor Parallel（TP）   | 单层矩阵的特征维           | 单层参数与计算过大            | 层内 All-Reduce / All-Gather      |
| Pipeline Parallel（PP） | Transformer layers | 模型深度和层间状态过大          | stage 间发送 activation / gradient |
| Context Parallel（CP）  | sequence / context | 长上下文激活与 Attention 过大 | 跨设备交换 K/V 或部分 Attention 结果      |

总设备数常可概念性写成：

$$
G_{\mathrm{total}}
=
G_{\mathrm{DP}}
\times
G_{\mathrm{TP}}
\times
G_{\mathrm{PP}}
\times
G_{\mathrm{CP}}
$$

例如 64 张 GPU 可以组织为：

```text
DP = 8
TP = 4
PP = 2
CP = 1
总卡数 = 8 × 4 × 2 = 64
```

实际配置要匹配硬件拓扑：TP 通信非常频繁，通常优先放在同一节点的 NVLink/NVSwitch 域内；PP 的通信主要是相邻 stage 的 activation/gradient，可以跨节点；DP 通信粒度较大，常放在更外层。

**2）Tensor Parallel：切的是一层内部的特征维**

考虑线性层：

$$
Y=XW,
\qquad
W\in\mathbb R^{d_{\mathrm{in}}\times d_{\mathrm{out}}}
$$

**Column Parallel** 按输出维切 $W$：

$$
W=[W_1,W_2,\ldots,W_p]
$$

每张卡计算：

$$
Y_i=XW_i
$$

最后：

$$
Y=[Y_1,Y_2,\ldots,Y_p]
$$

直觉是：每张卡负责一部分输出神经元。输入 $X$ 通常在 TP 组内可用，输出特征分片保留在各卡，若后续算子能直接消费分片，就不必立刻 All-Gather。

**Row Parallel** 按输入维切 $W$：

$$
W=
\begin{bmatrix}
W_1\\
W_2\\
\vdots\\
W_p
\end{bmatrix},
\qquad
X=[X_1,X_2,\ldots,X_p]
$$

每张卡先算局部贡献：

$$
Z_i=X_iW_i
$$

完整输出需要求和：

$$
Y=\sum_{i=1}^{p}Z_i
$$

因此通常需要 All-Reduce 或等价的 Reduce-Scatter。

Megatron 风格的 MLP 常把两种切法配对：

```text
输入 X
  ↓ Column Parallel Linear
各卡得到一部分扩展特征
  ↓ GeLU / SwiGLU 等逐元素操作，本地完成
  ↓ Row Parallel Linear
各卡计算部分输出贡献
  ↓ All-Reduce
恢复完整 residual stream
```

Attention 中也有类似结构：Q/K/V projection 常按 head 或输出维做 Column Parallel，Attention heads 在各卡本地计算，最后 output projection 用 Row Parallel 汇总。

> Tensor Parallel 切的是**特征维、head 或矩阵维度**，不是把 token 序列的前半段给 GPU 0、后半段给 GPU 1。后者属于 sequence/context 方向的并行。

**3）Pipeline Parallel：切的是层，并用 micro batch 填流水线**

假设 32 层模型切到 4 张卡：

```text
Stage 0 / GPU 0：Layer 0  ~ 7
Stage 1 / GPU 1：Layer 8  ~ 15
Stage 2 / GPU 2：Layer 16 ~ 23
Stage 3 / GPU 3：Layer 24 ~ 31
```

最朴素地一次只送一个 batch：

```text
GPU 0 forward 时，GPU 1/2/3 等待
GPU 1 forward 时，GPU 0/2/3 部分等待
...
```

会产生大量 pipeline bubble。解决办法是把 global batch 再拆成多个 pipeline micro batch：

```text
时间 1：Stage0 处理 micro0
时间 2：Stage0 处理 micro1；Stage1 处理 micro0
时间 3：Stage0 处理 micro2；Stage1 处理 micro1；Stage2 处理 micro0
...
```

流水线填满后多个 stage 可以并行工作。micro batch 越多，启动和排空阶段的 bubble 占比通常越小，但过多 micro batch 会增加调度复杂度、通信次数，且可能影响激活缓存和 batch 语义。

**1F1B** 是常见调度直觉：warmup 后，每个 stage 交替执行一个 forward 和一个 backward，使流水线保持忙碌，同时避免一次保存所有 micro batch 的激活。

Pipeline Parallel 的负载均衡不能只看层数：某些层、词表 embedding、LM head 或 MoE 层计算更重。如果每个 stage 层数相同但 FLOPs 不同，最快的 stage 仍要等待最慢 stage。

**4）Sequence Parallel 与 Context Parallel 不要混为一谈**

在很多 Megatron 语境中，**Sequence Parallel** 常与 Tensor Parallel 配合：把 LayerNorm、Dropout、residual 等原本在 TP 卡上重复保存的 activation 沿 sequence 维分片，以减少冗余激活；Attention 的完整上下文计算方式不一定被改变。

**Context Parallel** 更直接面向长上下文：把整个序列的 token 分到多张卡，每张卡只保存一部分 query 和 activation，并通过 Ring、All-Gather 或其它通信方式获得所需 K/V 信息，完成全局 Attention。

```text
Sequence Parallel：
常用于减少 TP 区域周边的冗余 activation。

Context Parallel：
真正把长 context 本身分散到多设备，Attention 需要跨设备协作。
```

术语在不同框架中可能略有差别，面试时最好先说明你采用的是哪套定义，而不是只背缩写。

![Pasted image 20260818184624](/my-blog/resources/uploads/obsidian-1787231837489-6.png)

**5）FlashAttention：优化的是 IO，不是改写 Attention 目标**

标准 Attention：

$$
S=\frac{QK^\top}{\sqrt{d_k}},
\qquad
P=\operatorname{softmax}(S),
\qquad
O=PV
$$

朴素实现容易把 $S$ 和 $P$ 这类 $L\times L$ 中间矩阵写入 GPU 高带宽显存 HBM，再反复读回计算。长序列下，不仅中间存储是 $O(L^2)$，大量 HBM 读写也可能成为瓶颈。

GPU 可以粗略理解为两级存储：

```text
HBM：容量大，但离计算单元较远，读写代价较高
SRAM / shared memory：容量小，但位于片上，访问快
```

FlashAttention 使用 tiling：把 Q、K、V 切成能放进片上 SRAM 的小块，在块内完成 score、softmax 统计和与 V 的乘法，只把必要结果写回 HBM。

难点是 softmax 需要知道整行所有 logits 才能归一化。FlashAttention 使用 online softmax 思想，为每个 query block 维护运行中的：

```text
当前最大值 m
当前归一化和 l
当前输出累积值 O
```

读取下一个 K/V block 后更新这些统计量，就能在不保存完整 $L\times L$ 矩阵的情况下得到与标准 softmax 相同的结果。

因此必须记住三句话：

```text
FlashAttention 是 exact attention，不是稀疏近似。
它仍然有 O(L²) 级别的 Attention 计算量。
它通过减少 HBM↔SRAM 数据搬运和中间矩阵存储来提速、省显存。
```

它与 linear attention、sparse attention 的区别是：后两者通常改变了 Attention 的数学结构或可见连接，而 FlashAttention 主要改变计算调度与内存访问方式。

FlashAttention 也不是任何 shape 上都必然等比例加速。收益受 sequence length、head dimension、causal mask、dropout、硬件和 kernel 实现影响；但在长序列训练中，它通常是极关键的基础优化。

![Pasted image 20260818184600](/my-blog/resources/uploads/obsidian-1787231837489-7.png)
**本轮面试题**

| 面试题                                                         | 面试场景回答                                                                                                                                                                                     |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Data、Tensor、Pipeline Parallel 分别切什么？**                    | Data Parallel 切 batch，每卡模型相同；Tensor Parallel 切单层矩阵、head 或特征维，让一层在多卡协同计算；Pipeline Parallel 切不同 Transformer layers，把模型深度分到不同 stage。它们分别解决吞吐、单层规模和整网深度问题，并可组合使用。                              |
| **Column Parallel Linear 是怎么切的？**                           | 对 $Y=XW$，按 $W$ 的输出维切成 $[W_1,\ldots,W_p]$，每卡计算 $Y_i=XW_i$，完整输出是沿特征维 concat。每卡负责一部分输出神经元。若后续算子能直接消费分片输出，就可以避免立刻 All-Gather。                                                                  |
| **Row Parallel Linear 为什么需要 Reduce？**                       | Row Parallel 按输入维切分 $X$ 和 $W$，每卡得到局部贡献 $Z_i=X_iW_i$，而完整输出是 $Y=\sum_iZ_i$。因此需要 All-Reduce 或 Reduce-Scatter 汇总各卡贡献。Megatron 常用 Column Parallel 第一层配 Row Parallel 第二层，把通信放在合适位置。              |
| **Tensor Parallel 为什么通常放在同一节点？**                            | TP 几乎每个 Transformer block 都会发生通信，频率高且延迟敏感。NVLink/NVSwitch 的带宽和时延通常优于跨节点网络，因此常把 TP 组放在同一高速互联域，再把 Pipeline 或 Data Parallel 放到更外层。                                                            |
| **Pipeline bubble 是什么？怎样减小？**                               | 流水线启动和排空时，一些 stage 没有可处理的 micro batch，只能空闲，这段时间就是 bubble。把 batch 拆成更多 micro batch、使用 1F1B 或 interleaved schedule、平衡各 stage 计算量，都能降低 bubble 占比，但会引入更多调度和通信复杂度。                              |
| **Pipeline Parallel 为什么还需要 gradient accumulation？**         | Pipeline 本身依赖多个 micro batch 才能填满各 stage；这些 micro batch 往往共同构成一次 optimizer update，因此天然与梯度累积结合。必须区分 pipeline micro batch 数、每卡 micro batch size 和外层 data-parallel degree，三者共同决定 global batch。 |
| **Sequence Parallel 和 Context Parallel 有什么区别？**             | 常见 Megatron 语境下，Sequence Parallel 主要把 TP 周边的 LayerNorm、Dropout、residual activation 沿序列维分片，减少冗余；Context Parallel 则把长上下文本身分到多卡，并让 Attention 跨设备交换 K/V 或部分结果。不同框架术语可能略有变化，回答时应先说明定义。          |
| **FlashAttention 为什么更快？**                                   | 它把 Q/K/V 分块放入片上 SRAM，在块内完成 score、online softmax 和与 V 的乘法，避免把完整 $L\times L$ score/probability 矩阵反复写入和读取 HBM。很多 GPU workload 的瓶颈是数据搬运而不只是 FLOPs，因此减少 IO 能显著提高速度并降低峰值显存。                      |
| **FlashAttention 是近似 Attention 吗？**                         | 不是。它使用分块和 online softmax 重排计算，但结果在数值误差范围内等价于标准 softmax attention，因此是 exact attention。它没有通过稀疏连接或低秩核函数改变目标。                                                                                  |
| **FlashAttention 把 Attention 的复杂度从 $O(L^2)$ 变成 $O(L)$ 了吗？** | 不能这样说。它显著降低了需要保存的 Attention 中间矩阵和 HBM IO，额外内存可接近线性随序列增长，但 dense attention 的核心 FLOPs 仍是二次量级。它解决的是 IO 和显存瓶颈，不是消除所有二次计算。                                                                      |

## 4. 让大规模训练真正跑稳：Loss Spike、NaN、监控与故障恢复

**本轮目标**：一个训练任务“能启动”不等于“在正确训练”。这一轮要建立监控和排障框架：发现异常、定位第一个坏 step、判断是数据、数值、优化器还是分布式问题，并依靠完整 checkpoint 恢复。

| 资源 | 今天读什么 | 阅读时盯住什么 |
|---|---|---|
| [PyTorch AMP Documentation](https://docs.pytorch.org/docs/stable/amp.html) | autocast、GradScaler 与 unscale | FP16 出现 Inf 后如何跳过 update；clipping 为什么要在 unscale 后 |
| [PyTorch Reproducibility Notes](https://docs.pytorch.org/docs/stable/notes/randomness.html) | 随机种子与非确定性说明 | “加载相同 checkpoint”为什么不必然 bitwise 相同 |
| [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html) | 分片保存与加载概念 | 多 rank 并行 checkpoint 与拓扑变化 |
| ZeRO、Megatron-LM、FlashAttention 回看 | 只回看各自的通信/显存代价 | 发生 OOM、hang、吞吐下降时该怀疑哪一层 |

**1）训练至少要监控哪些信号？**

只看 training loss 是远远不够的。建议把指标分成四组：

| 维度 | 常见指标 | 主要回答的问题 |
|---|---|---|
| 优化质量 | training loss、validation loss、PPL、learning rate、gradient norm、update/weight norm | 模型是否在正确收敛，更新是否过强或过弱 |
| 数值稳定 | activation/logit max、Inf/NaN count、FP16 loss scale、clipping frequency | 是否发生溢出、下溢或异常算子 |
| 吞吐效率 | tokens/s、samples/s、step time、MFU、GPU utilization | GPU 是否真正用于有效计算 |
| 系统健康 | allocated/reserved memory、data wait time、communication time、straggler、checkpoint time | 是否被数据、显存、网络或存储拖慢 |

**Loss 需要按 token 正确统计。**若各 rank、各 micro batch 的有效 token 数不同，应先汇总 loss numerator 和 token denominator，再计算全局平均。简单平均每个 rank 的局部 mean loss，可能让短序列 rank 与长序列 rank 权重相同，从而导致日志失真。

**Gradient norm** 是非常重要的早期预警：

- 长时间处于正常范围后突然暴涨，常与 loss spike、异常 batch 或数值溢出同步；
- 长期接近 0，可能是学习率太小、梯度被错误清空、参数被冻结、mask/label 出错；
- 每一步都触发强 clipping，说明训练可能依赖“不断踩刹车”，需要检查 LR、初始化、数据和精度，而不是把阈值越调越大。

**Update-to-weight ratio** 可以粗略理解为：

$$
\frac{\lVert\Delta\theta\rVert}{\lVert\theta\rVert}
$$

它比只看 gradient norm 更贴近“参数实际被改了多少”，因为 Adam 的自适应缩放和学习率都会改变最终 update。

**2）Loss Spike、Divergence 与 NaN 不完全相同**

```text
Loss Spike：
某一步或短时间 loss 明显升高，之后可能恢复。

Divergence：
loss 持续上升或不再回到正常轨迹，训练已经偏离稳定区域。

NaN / Inf：
数值已经超出表示范围或发生非法运算，常导致训练无法继续。
```

一次孤立 spike 可能来自异常数据；连续 spike 或持续上升更应怀疑学习率、数值稳定和 optimizer state。常见来源可以按层次排查：

```text
数据层：
空样本、异常长样本、错误编码、label 全为 -100、污染 batch、数据 mixture 突变

模型层：
attention mask 错误、除零、log/exp 溢出、初始化问题、特定 fused kernel bug

优化层：
峰值 LR 过高、warmup 太短、梯度爆炸、weight decay 配置错误、恢复后 optimizer state 丢失

精度层：
FP16 overflow/underflow、loss scale 不合适、clipping 前未 unscale

分布式层：
某 rank 数据或 shape 不一致、collective 顺序不一致、通信错误、部分 rank 恢复了不同 checkpoint
```

**3）推荐的 NaN 排查顺序**

不要一看到 NaN 就直接把学习率除以 10。更有效的流程是：

```text
1. 记录第一个出现异常的 global step、rank、数据 shard 和 batch id。
2. 从异常前一个完整 checkpoint 恢复，并尝试重放同一个 batch。
3. 检查 input_ids、attention_mask、labels、有效 token 数是否合理。
4. 检查 loss 进入 NaN 前，logits 和各层 activation 是否仍 finite。
5. backward 后检查哪些参数的 gradient 最先出现 Inf/NaN。
6. FP16 下检查 GradScaler、unscale 和 clipping 顺序。
7. 暂时改用 BF16/FP32、关闭可疑 fused kernel，判断是否为数值/算子问题。
8. 在单卡或更小并行组中复现，隔离 DDP/FSDP/TP/PP 通信因素。
```

检查 finite 可以形成“二分定位”思路：先看每个 Transformer block 的输出，找到第一个坏 block；再在该 block 内检查 Attention、MLP、Norm，避免逐元素盲查整个模型。

**4）Gradient Clipping 应放在哪里？**

Global norm clipping 常写成：

$$
g
\leftarrow
g\cdot
\min\left(1,\frac{\tau}{\lVert g\rVert+\epsilon}\right)
$$

它不是把每个梯度元素硬裁到 $[-\tau,\tau]$，而是当总梯度范数超过阈值时，按统一比例缩小全部梯度，尽量保留方向。

正确顺序通常是：

```text
多个 micro batch 累积完成
      ↓
FP16 GradScaler.unscale_(optimizer)
      ↓
计算并记录 global gradient norm
      ↓
clip_grad_norm_
      ↓
optimizer.step / scaler.step
      ↓
scheduler.step
```

如果在 unscale 前 clipping，看到的是被 loss scale 放大后的梯度，阈值就失去真实意义。如果每个 micro batch 分别 clipping，得到的也不是完整 effective batch 的梯度方向。

Clipping 是安全带，不是修复所有训练问题的万能药。若几乎每一步都严重裁剪，应优先检查峰值学习率、warmup、数据异常、初始化和精度。

**5）Checkpoint 怎样做到可用，而不是“文件存在就算成功”？**

可靠 checkpoint 需要同时考虑**完整性、一致性和可恢复性**。

```text
完整性：
model、optimizer、scheduler、step、scaler、RNG、data cursor 等是否齐全。

一致性：
所有 rank 保存的是同一个 global step 的状态，不能一部分 rank 是 step 1000，另一部分是 1001。

可恢复性：
文件能否真正 load；分片拓扑变化时能否 re-shard；恢复后的 LR、数据位置和 loss 是否连续。
```

推荐的工程原则：

- 先写临时目录或临时文件，全部 rank 成功后再写完成标记，避免把半写入 checkpoint 当成最新版本；
- 保留最近若干个 checkpoint，而不是保存新文件后立刻删除唯一旧版本；
- 定期做真实的 restore test，不能只验证“save 没报错”；
- 保存并记录 `global_optimizer_step` 和累计训练 token，而不只记录 dataloader iteration；
- 恢复后先打印当前 LR、optimizer step、loss scale、数据 shard 和模型 checksum 等关键信息；
- 观察恢复后的第一个有效 step，loss 与 gradient norm 应大体接续原轨迹。

Checkpoint 频率是 trade-off：

```text
保存太频繁 → 存储、网络和训练停顿成本高
保存太稀疏 → 故障后丢失更多训练 compute
```

合理间隔取决于故障率、checkpoint 大小、写入带宽和单次训练成本，而不是固定“每 N steps”适用于所有项目。

**6）OOM、Hang 与吞吐下降怎样区分？**

| 症状 | 优先怀疑 | 常见处理方向 |
|---|---|---|
| CUDA OOM | micro batch/sequence 太大、激活峰值、all-gather 峰值、内存碎片 | 降 micro batch、activation checkpointing、FlashAttention、调整 FSDP wrap/prefetch |
| 所有卡利用率低但程序在走 | data loader、CPU tokenization、checkpoint IO、同步 barrier | 预处理数据、增加 prefetch、异步/并行 IO、分析 step breakdown |
| 部分卡忙、部分卡等 | pipeline stage 不平衡、straggler、变长 batch 不均 | 重分 stage、按 token 均衡 batch、检查慢节点 |
| 程序完全 hang | collective 顺序不一致、某 rank 提前异常、网络/NCCL 问题 | 检查各 rank 日志、collective trace、超时与健康状态 |
| loss 正常但 tokens/s 突降 | sequence 长度分布变化、通信拥塞、频繁 checkpoint、kernel fallback | 分离 compute/data/comm 时间并定位变化点 |

训练工程排障的核心原则是：**先把问题分类为数据、计算、数值、通信或存储，再用最小可复现配置隔离变量。**不要同时修改学习率、batch、精度和并行配置，否则即使恢复正常，也不知道真正原因。

![Pasted image 20260818185634](/my-blog/resources/uploads/obsidian-1787231837489-8.png)

**Day 6 最终串联：面试官问“一个 70B 模型放不进单卡，你如何设计训练？”**

可以按下面的顺序回答，而不是直接堆框架名：

```text
第一步：根据参数精度、梯度、Adam 状态和激活估算显存，明确瓶颈是模型状态、单层参数还是长序列激活。

第二步：使用 BF16、FlashAttention、合理 micro batch 和 activation checkpointing，
通过 gradient accumulation 达到目标 global token batch；scheduler 按 optimizer step 推进。

第三步：模型能放进单卡但要提高吞吐时用 DDP；模型状态放不下时用 ZeRO-3/FSDP 分片。

第四步：若单层仍过大，加入 Tensor Parallel；模型很深时加入 Pipeline Parallel；长上下文再加入 Context Parallel。

第五步：按硬件拓扑安排并行组，把高频 TP 通信放在高速互联域内，并监控 compute、communication、data 和 checkpoint 时间。

第六步：保存可恢复的分片 checkpoint，包括 optimizer、scheduler、step、scaler、RNG 和 data cursor，
并监控 loss、gradient norm、update ratio、tokens/s、显存和各 rank 健康状态。
```

**今天的论文安排**

| 优先级 | 资料 | 阅读要求 |
|---|---|---|
| **必读 A** | ZeRO | 精读三阶段状态切分图；能不看笔记讲清 Stage 1/2/3 与通信代价 |
| **必读 A** | FlashAttention | 精读 IO-aware 动机和算法总览；能解释 exact、tiling、online softmax |
| **重点 B** | Megatron-LM 2019 | 重点读 MLP/Attention 的 Column/Row Parallel 图，不要求推完所有实验 |
| **重点 B** | Megatron-LM 2021 | 重点理解 DP、TP、PP 如何组合，以及 pipeline bubble |
| **补强 B** | Chinchilla | 回看 Abstract 与 Figure 1，把 undertrained、fixed compute、70B/1.4T 讲顺 |
| **工程阅读** | PyTorch DDP、FSDP、Activation Checkpoint、DCP 文档 | 只读原理与状态流，不在今天展开完整实验代码 |

**Day 6 最终自测：下面 14 个问题至少能口头回答 12 个**

1. Chinchilla 为什么说明“参数越大越好”不完整？
2. micro batch、gradient accumulation 和 global batch 的关系是什么？
3. 为什么 scheduler 应按 optimizer step 而不是 micro step 推进？
4. training checkpoint 和 activation checkpointing 有何区别？
5. 7B BF16 权重约 14 GB，为什么训练仍可能需要上百 GB？
6. DDP 在什么阶段同步什么？它为什么不降低单卡完整模型状态？
7. ZeRO-1、2、3 分别切分哪些状态？
8. FSDP 在 forward/backward 前后为什么要 all-gather 与 reshard？
9. Column Parallel 与 Row Parallel 分别怎样切线性层？
10. pipeline bubble 为什么出现？micro batch 怎样缓解？
11. Sequence Parallel 与 Context Parallel 的关注点有何不同？
12. FlashAttention 为什么是 exact attention？它有没有消除 $O(L^2)$ FLOPs？
13. FP16 训练中 clipping 为什么必须放在 unscale 之后？
14. 从 checkpoint 恢复后，怎样判断训练轨迹真的接上了？

**本轮面试题**

| 面试题 | 面试场景回答 |
|---|---|
| **训练中出现一次 loss spike，你会怎么处理？** | 我先判断是孤立 spike 还是持续发散，并定位第一个异常 global step、rank 和 batch。然后从前一个 checkpoint 重放同一数据，检查有效 token、mask 和 labels；再看 activation、logits、unscaled gradient norm 是否 finite。若单卡稳定而多卡异常，再检查 collective 顺序、不同 rank 状态和通信。不会一上来同时改多个超参数。 |
| **出现 NaN 时为什么不能只降低学习率？** | NaN 可能来自错误数据、非法算子、FP16 overflow、错误 mask、optimizer state 损坏或分布式不一致。降低 LR 只对“更新过大”这一类原因有效，还可能暂时掩盖真正 bug。更可靠的方法是定位 first bad step、重放 batch、逐层检查 finite，并通过切换 BF16/FP32、关闭 fused kernel、缩小并行配置隔离原因。 |
| **Gradient clipping 的正确时机是什么？** | 在所有 micro batch 梯度累积完成后执行；FP16 下先用 GradScaler unscale，再计算并记录 global norm，然后 clip，最后 optimizer/scaler step。若在 unscale 前裁剪，阈值对应的是被放大的梯度；若每个 micro batch 分别裁剪，会改变最终累积梯度方向。 |
| **Gradient clipping 能解决梯度爆炸吗？** | 它能限制单次 update 的极端幅度，是有效安全机制，但不应替代根因分析。若长期频繁强裁剪，通常说明 peak LR、warmup、初始化、数据或数值精度存在问题。好的监控应同时记录 clipping 前 gradient norm 和 clipping frequency。 |
| **为什么训练 loss 的多卡平均可能算错？** | 若每个 rank 的有效 token 数不同，直接平均各 rank 的 mean loss 会让每个 rank 等权，而不是让每个 token 等权。正确方法是跨 rank 汇总 loss sum 和有效 token count，再做 `global_loss_sum / global_token_count`。变长样本、padding 和 packing 下尤其重要。 |
| **恢复 checkpoint 后 loss 突然跳高，可能缺了什么？** | 常见是只恢复了 model weights，却没有恢复 optimizer moments、scheduler state、global step 或 GradScaler，导致下一步更新和学习率发生突变；也可能 data cursor 改变，恢复后读到了不同分布的数据。应核对当前 LR、optimizer step、loss scale、数据 shard 和 checkpoint 完整性。 |
| **怎样设计可靠的分布式 checkpoint？** | 所有 rank 应在同一 global step 保存一致状态，采用 sharded/parallel save 降低聚合压力，并写完成标记避免加载半成品。保存 model、optimizer、scheduler、step、scaler、RNG 和 data cursor；保留至少一个旧版本并定期做 restore test。拓扑变化时使用支持 re-shard 的 distributed checkpoint 工具。 |
| **加载同一 checkpoint 为什么不一定 bitwise reproducible？** | GPU kernel、浮点归约顺序和某些并行算子可能非确定；world size 或并行拓扑变化会改变 All-Reduce 顺序和数据分片；若 RNG、sampler cursor 没完整恢复，dropout 和数据顺序也会变化。因此一般追求统计和 loss 轨迹连续，严格逐 bit 一致需要额外确定性设置且可能牺牲性能。 |
| **OOM 时你会按什么顺序优化？** | 先用显存 profile 判断峰值来自 activation、模型状态还是临时 all-gather。激活占主导时减 micro batch、用 gradient accumulation、activation checkpointing 和 FlashAttention；模型状态占主导时用 ZeRO/FSDP；单层仍放不下时用 TP；长上下文再考虑 CP。不能只盲目清 CUDA cache。 |
| **怎样判断训练慢在算力、通信还是数据？** | 我会拆分 step time：data wait、forward/backward compute、collective communication、optimizer 和 checkpoint IO；结合 GPU utilization、MFU、tokens/s、网络带宽和各 rank 时间。GPU 空闲且 data wait 高说明数据瓶颈；计算核持续满载但 MFU低可能 shape/kernel 不理想；collective 占比高则检查并行度、bucket、拓扑和通信重叠。 |
| **一个 70B 模型的并行策略如何选择？** | 先估算显存并确定瓶颈。用 BF16、FlashAttention、activation checkpointing 和梯度累积控制激活与 global batch；状态放不下用 FSDP/ZeRO-3；单层过大用 TP；模型深度和集群规模较大时加 PP；长上下文加 CP。TP 优先放高速互联域，并用完整 checkpoint 与监控保证可恢复和稳定。 |
