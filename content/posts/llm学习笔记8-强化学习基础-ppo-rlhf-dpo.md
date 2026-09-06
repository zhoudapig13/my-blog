---
title: "『LLM学习笔记8』强化学习基础：PPO/RLHF/DPO"
category: "未分类"
tags:
  []
date: "2026-09-06"
summary: ""
pdf: ""
pdfTitle: ""
---

**Day8：LLM 后训练（二）——强化学习基础、PPO、RLHF 与 DPO**

**今日定位：** Day7 的 SFT 解决“模型如何模仿标准回答”，Day8 继续回答“没有唯一标准答案时，如何让模型更倾向于高质量回答”。今天会把强化学习讲得比一般 LLM 教程更细，但始终围绕大模型后训练展开，不额外铺开与 LLM 关系较弱的传统控制任务。

**今日主线：**

```text
SFT 的局限
    ↓
把 LLM 看成 Policy：state → token action → trajectory → reward
    ↓
Policy Gradient：为什么 reward 能改变 token 概率
    ↓
Value / Q / Advantage：如何降低方差并判断“比预期好多少”
    ↓
GAE：如何把后面的 reward 分配给前面的 token
    ↓
PPO：如何避免一次更新把模型推得太远
    ↓
Reward Model + KL + PPO：完整 RLHF
    ↓
DPO：如何把偏好优化改写成简单的成对分类目标
    ↓
GRPO 预览：为什么可以不训练独立 Value Model
```

**学完后的最低标准：** 你需要能够不背稿地解释 Policy Gradient、Value、Advantage、GAE、PPO clipping、RLHF 四类模型、DPO 与 PPO 的区别；看到训练代码中的 `logprobs`、`old_logprobs`、`ref_logprobs`、`values`、`returns`、`advantages` 时，知道它们分别从哪里来、用于什么计算。

## 第一轮：为什么 LLM 后训练会用到强化学习

**1. SFT 到底在训练什么？**

给定 prompt $x$ 和人工示范回答 $y^*=(y_1^*,\ldots,y_T^*)$，SFT 通常最小化 token-level negative log-likelihood：

$$
\mathcal{L}_{\mathrm{SFT}}(\theta)
=-\sum_{t=1}^{T}\log \pi_\theta\left(y_t^*\mid x,y_{<t}^*\right).
$$

它的直接作用是：在每个位置，提高示范答案中下一个 token 的概率。SFT 很擅长教会模型格式、语气、任务流程和基本回答方式，但它没有显式比较“答案 A 比答案 B 好多少”。当同一个问题存在多种合理回答，或者我们在意帮助性、真实性、安全性、简洁性等综合偏好时，单条示范很难完整表达这些相对偏好。

例如，同一个问题有三个回答：

| 回答  | 特征        | SFT / 偏好学习看到的信息                |
| --- | --------- | ------------------------------ |
| A   | 正确、简洁、直接  | 如果它被选作示范，SFT 只会提高 A 的概率        |
| B   | 正确但极其啰嗦   | SFT 数据里若没有 B，就不会直接告诉模型“B 次于 A” |
| C   | 语言流畅但事实错误 | 偏好标注可以明确给出 $A \succ B \succ C$ |

因此，偏好数据通常不要求人类写出唯一的完美答案，而是让标注者比较多个候选，例如：

$$
(x,y_w,y_l),
$$

其中 $y_w$ 是 preferred / chosen response，$y_l$ 是 rejected response。下标 $w$ 可以理解为 winner，$l$ 可以理解为 loser。

![Pasted image 20260821203800](/my-blog/resources/uploads/obsidian-1788716852920-1.png)

**2. 把 LLM 翻译成强化学习语言**

传统强化学习讨论 Agent 与 Environment 的交互。LLM 生成一段回答时，也可以做如下映射：

| 强化学习术语                           | LLM 中的对应物                     | 直观解释                                  |
| -------------------------------- | ----------------------------- | ------------------------------------- |
| State $s_t$                      | Prompt 加已经生成的前缀 $(x,y_{<t})$  | 模型在第 $t$ 步已经看到的全部文本                   |
| Action $a_t$                     | 下一个 token $y_t$               | 当前这一步选择生成什么 token                     |
| Policy $\pi_\theta(a_t\mid s_t)$ | LLM 的 next-token distribution | 给定当前文本前缀，各 token 的生成概率                |
| Transition                       | 把新 token 接到前缀后面               | $s_{t+1}=(x,y_{\le t})$，在纯文本生成中基本是确定的 |
| Trajectory $\tau$                | 一整条生成序列                       | 从第一个 token 到 EOS 的完整回答                |
| Reward $R(x,y)$                  | 对完整回答质量的评分                    | 可来自 Reward Model、人类、规则检查器或可验证答案       |
| Episode                          | 一次从 Prompt 到完整回答的生成           | 遇到 EOS 或最大长度时结束                       |

因此，在标准自回归生成中：

$$
s_t=(x,y_{<t}),\qquad a_t=y_t,
$$

$$
\pi_\theta(a_t\mid s_t)
=\pi_\theta(y_t\mid x,y_{<t}).
$$

完整回答 $y=(y_1,\ldots,y_T)$ 的概率是各步条件概率的乘积：

$$
\pi_\theta(y\mid x)
=\prod_{t=1}^{T}\pi_\theta(y_t\mid x,y_{<t}).
$$

取对数后，乘积变成求和：

$$
\log \pi_\theta(y\mid x)
=\sum_{t=1}^{T}\log \pi_\theta(y_t\mid x,y_{<t}).
$$

这条式子非常重要。后面 Policy Gradient、PPO、DPO 都会不断使用“整段回答的 log-probability 等于各 token log-probability 之和”。

![Pasted image 20260821204542](/my-blog/resources/uploads/obsidian-1788716852920-2.png)

**3. 强化学习真正改变的是什么？**

LLM 强化学习不是把一句奖励文字“塞回模型”，而是调整 policy 的概率分布：高 reward 的回答以后更容易被采样，低 reward 的回答以后更不容易被采样。目标可以写成：

$$
J(\theta)
=\mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot\mid x)}
\left[R(x,y)\right],
$$

我们希望最大化 $J(\theta)$。这里包含一个与 SFT 非常不同的地方：训练答案 $y$ 不是固定标签，而是由当前模型 $\pi_\theta$ 自己采样出来的。因此，模型变了，采样到的数据分布也会跟着变，这就是 on-policy RL 比普通监督学习复杂的重要原因之一。

**4. LLM 的 reward 不一定只在最后出现**

最简单的情况是：完整回答结束后，Reward Model 给一个 sequence-level reward。实际 RLHF 中还可能给每个 token 加 KL penalty，因此可以写成逐步 reward $r_t$，总回报为：

$$
G_t=\sum_{k=t}^{T}\gamma^{k-t}r_k,
$$

其中 $G_t$ 是从第 $t$ 步开始能够获得的 return，$\gamma\in[0,1]$ 是 discount factor。对于有限长度文本，具体是否使用折扣、怎样设置 $\gamma$ 属于实现选择；概念上只需先记住：**Reward 是某一步得到的即时信号，Return 是从现在到结束的累计信号。**

**5. 先消除三个常见误解**

| 误解 | 正确理解 |
|---|---|
| “用了 Reward Model 就叫强化学习” | Reward Model 只负责评分；是否属于 RL，还要看是否从 policy 采样并用 reward 更新 policy |
| “每个 token 都有人工标签” | RLHF 常常只有完整回答的偏好或最终 reward，token 级 credit assignment 需要算法估计 |
| “RL 会给模型添加新知识” | RL 主要重分配已有行为的概率；它可以强化可探索到的能力，但不保证凭空补足事实知识 |

**本轮面试题**

| 面试问题 | 面试场景下的回答 |
|---|---|
| 为什么 SFT 后还需要 RLHF 或 Preference Optimization？ | SFT 最大化示范答案的 token likelihood，擅长教模型模仿正确格式和基本行为，但它没有显式表达多个合理回答之间的相对质量。偏好学习可以使用 chosen / rejected 比较，把帮助性、真实性、安全性、风格等难以写成唯一标签的目标转化为相对监督。RLHF 进一步允许模型从当前 policy 采样回答，再用奖励信号优化其输出分布。 |
| LLM 中的 state、action、policy 和 trajectory 分别是什么？ | 第 $t$ 步的 state 是 prompt 与已生成前缀 $(x,y_{<t})$；action 是下一个 token $y_t$；policy 是 LLM 的 next-token distribution $\pi_\theta(y_t\mid x,y_{<t})$；完整回答是 trajectory。纯文本生成的 transition 通常只是把 token 拼到前缀后面。 |
| SFT 和 RL 的训练数据分布有什么关键区别？ | SFT 通常在固定的人类示范数据上做 teacher forcing；RL 中回答由当前或旧 policy 采样，policy 更新后 rollout 分布也会改变，因此存在 on-policy、分布漂移和旧数据失效等问题。 |
| Reward 与 Return 有什么区别？ | Reward $r_t$ 是某一步的即时反馈，Return $G_t$ 是从当前时刻到 episode 结束的累计 reward。Policy Gradient 真正希望提高的是期望 return，而不只是某一步的局部 reward。 |
| 为什么一整段回答的 log-probability 是 token log-probability 的和？ | 自回归模型用链式法则把序列概率分解为各 token 条件概率的乘积；对乘积取对数后变成求和。这让序列级 reward 可以通过各 token 的 log-probability 共同影响 policy 更新。 |

## 第二轮：Policy Gradient——Reward 为什么能更新离散 token

**1. 最核心的困难：采样操作不能像普通网络层一样直接反传**

假设模型从 categorical distribution 中采样一个 token。采样得到的 token id 是离散的，我们不能沿着“token id → reward”这条路径普通地做链式求导；而且 reward 可能来自人类、规则程序或一个冻结的 Reward Model，本来也不需要对 policy 参数 $\theta$ 求导。

Policy Gradient 的关键不是对 reward 本身求导，而是问：

> 哪些由 policy 产生的轨迹得到了高 reward？那就提高这些轨迹的概率；哪些轨迹得到低 reward？那就降低其概率。

**2. Log-derivative trick 的推导**

先暂时固定一个 prompt，并把所有可能回答记为 $y$：

$$
J(\theta)=\sum_y \pi_\theta(y)R(y).
$$

对参数求梯度：

$$
\nabla_\theta J(\theta)
=\sum_y \nabla_\theta\pi_\theta(y)R(y).
$$

利用恒等式：

$$
\nabla_\theta\pi_\theta(y)
=\pi_\theta(y)\nabla_\theta\log\pi_\theta(y),
$$

得到：

$$
\nabla_\theta J(\theta)
=\sum_y \pi_\theta(y)R(y)\nabla_\theta\log\pi_\theta(y),
$$

也就是：

$$
\boxed{
\nabla_\theta J(\theta)
=\mathbb{E}_{y\sim\pi_\theta}
\left[R(y)\nabla_\theta\log\pi_\theta(y)\right]
}
$$

这就是 REINFORCE / Policy Gradient 最核心的形式。它把“对一个离散采样分布的期望求导”，转成了“对被采样结果的 log-probability 求导”。

![Pasted image 20260821221029](/my-blog/resources/uploads/obsidian-1788716852920-3.png)

**3. 为什么高 reward 会提高概率？**

训练时通常最小化 loss，因此可写成：

$$
\mathcal{L}_{\mathrm{REINFORCE}}
=-R(y)\log\pi_\theta(y).
$$

当 $R(y)>0$ 时，最小化该 loss 会增大 $\log\pi_\theta(y)$，也就是提高这条回答的概率；当有效权重为负时，则降低它的概率。

对自回归 LLM：

$$
\mathcal{L}
=-R(x,y)\sum_{t=1}^{T}
\log\pi_\theta(y_t\mid x,y_{<t}).
$$

因此，一个 sequence-level reward 会同时作用于整条回答中的所有 token。这里马上暴露出一个问题：如果回答前半段很好、后半段犯错，所有 token 却可能拿到同一个最终分数。这个问题称为 **credit assignment**：最终结果应该归功或归咎于哪些 action？

**4. 一个极简数值例子**

模型针对同一 prompt 采样两条回答：

| 回答 | 当前概率 | Reward | 更新方向 |
|---|---:|---:|---|
| $y_A$：正确且清楚 | $0.20$ | $+1.2$ | 提高 $\log\pi_\theta(y_A)$ |
| $y_B$：流畅但错误 | $0.35$ | $-0.4$ | 降低 $\log\pi_\theta(y_B)$ |

Policy Gradient 并不是要求模型立刻把 $y_A$ 变成概率 1，而是给参数一个局部梯度：在类似状态下，让产生 $y_A$ 的 token 组合稍微更可能，让产生 $y_B$ 的 token 组合稍微更不可能。多轮 rollout 与更新后，概率分布才逐渐变化。

**5. 为什么直接乘 Reward 的方差很大？**

假设某个 prompt 本身很容易，模型随便回答也能拿到 $0.8$；另一个 prompt 很难，回答得很好也只有 $0.4$。如果直接看绝对 reward，容易把“题目容易”误当成“action 很好”。同时，采样本身存在随机性，同一 state 下不同回答分数波动很大，梯度方向会非常嘈杂。

解决思路是从 return 中减去一个与 action 无关的 baseline：

$$
\nabla_\theta J(\theta)
=\mathbb{E}
\left[(G_t-b(s_t))\nabla_\theta\log\pi_\theta(a_t\mid s_t)\right].
$$

为什么减 baseline 不会系统性改变梯度方向？因为：

$$
\mathbb{E}_{a\sim\pi_\theta}
\left[b(s)\nabla_\theta\log\pi_\theta(a\mid s)\right]
=b(s)\nabla_\theta\sum_a\pi_\theta(a\mid s)=0.
$$

baseline 不告诉我们哪个 action 更好，它只是把比较基准从“绝对零分”改成“在这个 state 下通常能拿多少分”，从而降低 variance。最常用的 state-dependent baseline 就是 Value Function $V(s)$。

![Pasted image 20260821223205](/my-blog/resources/uploads/obsidian-1788716852920-4.png)

**6. 从 REINFORCE 过渡到 Actor-Critic**

直接用完整 Monte Carlo return 作为权重，通常要等回答结束后才能计算，而且方差较大。Actor-Critic 的基本分工是：

- **Actor：** policy $\pi_\theta$，决定生成哪个 token；
- **Critic：** value function $V_\psi(s_t)$，预测当前前缀未来大概能获得多少 return；
- **更新 Actor：** 使用 $G_t-V_\psi(s_t)$ 或更稳定的 Advantage estimator；
- **更新 Critic：** 让预测值逼近实际 return。

Actor-Critic 不是四个神秘模型，而是一种职责拆分：Actor 决定做什么，Critic 估计这个局面通常有多好。

![Pasted image 20260822011753](/my-blog/resources/uploads/obsidian-1788716852920-5.png)

**本轮面试题**

| 面试问题 | 面试场景下的回答 |
|---|---|
| Reward 不可导，为什么还能更新 LLM？ | Policy Gradient 不需要对 reward 或离散 token 采样本身反传。它利用 log-derivative trick，把期望 reward 的梯度写成 $\mathbb{E}[R\nabla\log\pi]$。Reward 只作为权重：高 reward 提高被采样轨迹的 log-probability，低于基准的轨迹降低其 log-probability。 |
| 为什么公式里使用 $\log\pi$，而不是直接使用 $\pi$？ | 因为 $\nabla\pi=\pi\nabla\log\pi$，这样可以把对所有轨迹概率的求和改写为从当前 policy 采样的期望，得到可用 Monte Carlo 样本估计的梯度；另外自回归序列的 log-probability 能自然分解为 token log-probability 之和。 |
| 什么是 credit assignment？ | 它指最终 reward 应该归因到轨迹中的哪些 action。LLM 常只在回答结束时拿到 sequence reward，但整段回答包含很多 token；如果把同一个 reward 简单赋给所有 token，会产生粗糙、方差较大的更新，因此需要 return、value、advantage、GAE 或 process reward 等机制。 |
| 为什么减去 baseline 不会引入偏差？ | 只要 baseline 依赖 state 而不依赖当前采样 action，$\mathbb{E}[b(s)\nabla\log\pi(a\mid s)]=0$，因此它不会改变 policy gradient 的期望，只会改变样本梯度的方差。 |
| REINFORCE 和 Actor-Critic 的主要区别是什么？ | REINFORCE 通常直接用采样轨迹的 Monte Carlo return 加权 log-probability，结构简单但方差高；Actor-Critic 额外学习 Value Function 作为 baseline，用 Advantage 更新 Actor，并用回归目标训练 Critic，通常更稳定、更具样本效率。 |

## 第三轮：Value、Q、Advantage 与 GAE

**1. Value、Q、Advantage 分别回答什么问题？**

它们都在评价“未来”，但条件不同：

$$
V^\pi(s)
=\mathbb{E}_\pi[G_t\mid s_t=s],
$$

表示处于 state $s$ 后，继续按照 policy $\pi$ 行动，预期能获得多少 return。

$$
Q^\pi(s,a)
=\mathbb{E}_\pi[G_t\mid s_t=s,a_t=a],
$$

表示在 state $s$ 先选择 action $a$，之后继续按照 policy $\pi$ 行动，预期能获得多少 return。

$$
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s),
$$

表示 action $a$ 相比当前 policy 在 state $s$ 下的平均水平好多少。

可以用一句话记：

- $V(s)$：**这个局面平均有多好？**
- $Q(s,a)$：**在这个局面做这个动作后有多好？**
- $A(s,a)$：**这个动作比该局面的平均选择好多少？**

![Pasted image 20260822013829](/my-blog/resources/uploads/obsidian-1788716852920-6.png)

**2. Advantage 为什么比 Reward 更适合指导更新？**

假设模型面对一个很容易的问题，平均都能拿 $0.9$ 分：某回答得 $0.92$，看绝对分数很高，但只比预期好 $0.02$；面对极难问题，平均只能得 $0.2$，某回答得 $0.55$，绝对分不如前者，却比预期好 $0.35$。Policy 更新真正关心的是 action 相对当前能力的提升，因此 Advantage 更合理。

| 情况 | $Q(s,a)$ | $V(s)$ | $A(s,a)$ | 更新直觉 |
|---|---:|---:|---:|---|
| 容易题中的普通好答案 | $0.92$ | $0.90$ | $+0.02$ | 轻微提高概率 |
| 困难题中的突破性答案 | $0.55$ | $0.20$ | $+0.35$ | 明显提高概率 |
| 容易题中的失误答案 | $0.60$ | $0.90$ | $-0.30$ | 降低概率 |

Policy loss 因而常写成：

$$
\mathcal{L}_{\mathrm{actor}}
=-\mathbb{E}_t
\left[\hat A_t\log\pi_\theta(a_t\mid s_t)\right].
$$

$\hat A_t>0$ 时提高 action 概率，$\hat A_t<0$ 时降低 action 概率。帽子表示它不是不可知的真实 Advantage，而是由采样数据估计出来的。

**3. Value Model 与 Reward Model 完全不是一回事**

| 模型 | 输入 | 输出 | 训练目标 | 回答的问题 |
|---|---|---|---|---|
| Reward Model $r_\phi(x,y)$ | Prompt + 完整回答 | 通常是一个 scalar score | 拟合人类偏好排序 | “这条完整回答质量如何？” |
| Value Model $V_\psi(s_t)$ | Prompt + 当前生成前缀 | 当前 state 的 expected return | 回归 rollout 中估计的 return | “从这个前缀继续生成，预计最终能拿多少分？” |

Reward Model 通常在 PPO 前已经用偏好数据训练好并冻结；Value Model 在 PPO 过程中随 policy 一起更新，因为 policy 变了之后，未来 expected return 也会变。

**4. TD Error：当前预测与一步之后预测是否一致**

Value Function 应满足 Bellman 风格关系：

$$
V(s_t)\approx r_t+\gamma V(s_{t+1}).
$$

因此定义 temporal-difference error：

$$
\delta_t
=r_t+\gamma V(s_{t+1})-V(s_t).
$$

如果 $\delta_t>0$，说明实际的一步 reward 加下一状态价值高于此前预期；如果 $\delta_t<0$，说明结果低于预期。$\delta_t$ 本身可以作为一种 one-step Advantage estimate，但它较依赖 Value Model 的准确性。

**5. GAE：把后续的“超出预期”向前传播**

Generalized Advantage Estimation 定义为：

$$
\hat A_t^{\mathrm{GAE}(\gamma,\lambda)}
=\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}.
$$

展开就是：

$$
\hat A_t
=\delta_t+\gamma\lambda\delta_{t+1}
+(\gamma\lambda)^2\delta_{t+2}+\cdots.
$$

$\lambda$ 控制使用多远的未来 TD error：

| $\lambda$ | 近似                                | 特征                                     |
| --------: | --------------------------------- | -------------------------------------- |
|       $0$ | 只用 $\delta_t$ 的 one-step TD       | variance 较低，但更依赖 Value Model，bias 可能较高 |
|    接近 $1$ | 使用更长未来信息，接近 Monte Carlo advantage | bias 通常更低，但受整条轨迹随机性影响，variance 更高      |
|       中间值 | 多步加权折中                            | PPO 中常用来平衡 bias 与 variance，具体取值是超参数    |


![Pasted image 20260822025408](/my-blog/resources/uploads/obsidian-1788716852920-7.png)

**6. 用一条 LLM trajectory 手算 GAE**

假设模型生成四个 token，只有结束时得到最终 reward，令 $\gamma=1$、$\lambda=0.9$，终止状态价值为 $0$：

| $t$ | 生成 token | $r_t$ | $V(s_t)$ | $V(s_{t+1})$ | $\delta_t=r_t+V(s_{t+1})-V(s_t)$ |
|---:|---|---:|---:|---:|---:|
| 1 | `答` | $0$ | $0.30$ | $0.40$ | $0.10$ |
| 2 | `案` | $0$ | $0.40$ | $0.55$ | $0.15$ |
| 3 | `北京` | $0$ | $0.55$ | $0.75$ | $0.20$ |
| 4 | `。` | $1.00$ | $0.75$ | $0$ | $0.25$ |

从后往前计算：

$$
\hat A_4=0.25,
$$

$$
\hat A_3=0.20+0.9\times0.25=0.425,
$$

$$
\hat A_2=0.15+0.9\times0.425=0.5325,
$$

$$
\hat A_1=0.10+0.9\times0.5325=0.57925.
$$

这不是说第一个 token 一定比最后一个 token 更“重要”，而是说明：从第一个 state 往后看，后续多步都持续好于 Value Model 的预期，因此这些正 TD errors 会累积进更早时刻的 Advantage。实际训练中还会做 padding mask、response mask、advantage normalization 等处理。

**7. Return 与 Advantage 如何训练 Critic？**

常见做法先由 GAE 得到 $\hat A_t$，再构造 value target：

$$
\hat G_t=\hat A_t+V_{\mathrm{old}}(s_t).
$$

Value Model 最小化回归损失：

$$
\mathcal{L}_{V}
=\mathbb{E}_t\left[
\left(V_\psi(s_t)-\hat G_t\right)^2
\right].
$$

Actor 使用 Advantage 更新，Critic 使用 Return target 更新。二者相互配合，但要注意：Critic 只提供 baseline，它不直接决定 token。

**本轮面试题**

| 面试问题                              | 面试场景下的回答                                                                                                                                                                         |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| $V(s)$、$Q(s,a)$、$A(s,a)$ 的区别是什么？  | $V(s)$ 是处于 state 后按当前 policy 行动的平均 expected return；$Q(s,a)$ 是先执行指定 action 后的 expected return；$A(s,a)=Q(s,a)-V(s)$ 衡量该 action 相对当前 policy 平均水平好多少。Actor 更新主要需要 Advantage，而不是绝对价值。 |
| Reward Model 与 Value Model 有什么区别？ | Reward Model 对完整 prompt-response 给偏好分数，通常在 RL 阶段冻结；Value Model 对每个生成前缀预测未来 expected return，并在 PPO 过程中训练。前者提供目标信号，后者提供降低方差的 baseline。                                             |
| TD error 的含义是什么？                  | $\delta_t=r_t+\gamma V(s_{t+1})-V(s_t)$ 比较“当前预测”与“经历一步后得到的 reward 加下一状态预测”。正值表示这一步后的结果比原先预期好，负值表示低于预期。                                                                           |
| GAE 解决了什么问题？                      | GAE 用指数加权的多步 TD errors 估计 Advantage，在 one-step TD 的低方差高偏差和 Monte Carlo 的低偏差高方差之间做折中，从而让 PPO 的 policy update 更稳定。                                                                 |
| $\lambda$ 越大越好吗？                  | 不一定。$\lambda$ 越大使用的远期信息越多，通常 bias 降低但 variance 上升；Value Model 不准、trajectory 很随机时，大 $\lambda$ 可能让估计更噪。它是需要结合任务、长度和 reward 结构调节的超参数。                                               |
| 为什么 Value Model 要随 Policy 一起更新？   | $V^\pi(s)$ 定义依赖当前 policy。Policy 改变后，从同一前缀继续生成的行为分布与 expected return 都会变化，旧 Critic 会逐渐失准，因此需要用新 rollout 持续更新。                                                                     |

## 第四轮：PPO——怎样限制 Policy 一次不要改得太猛

**1. Vanilla Policy Gradient 为什么不稳定？**

如果某批回答碰巧获得高 Advantage，普通 Policy Gradient 可能一次大幅提高其中 token 的概率。问题在于：

- 参数共享导致一个 token 的更新会影响大量其他 state；
- rollout 来自更新前的 policy，更新太大后旧数据不再代表新 policy；
- Reward Model 只是近似目标，大步更新更容易找到它的漏洞；
- 对语言模型而言，一个不稳定 step 就可能导致输出分布、长度和格式同时漂移。

PPO 的核心目标可以概括为：**利用旧 policy 采到的数据做多次 minibatch 更新，但限制新 policy 相对采样 policy 的有利变化不要无限扩大。** PPO 论文中的 “Proximal” 就是在强调更新保持在旧 policy 附近。

### 一句话总结

- **Vanilla Policy Gradient：看到一个高分回答，可能立刻大幅提高它的概率，容易矫枉过正。
- **PPO：可以学习这个高分回答，但每次只能改一点，尽量让新模型保持在旧模型附近。

所以，“Proximal”可以理解为：**靠近的、不过度远离的。**

PPO 的核心思想就是：**可以进步，但不要一步迈得太大。**

**2. Old Policy 与 Importance Ratio**

先用 $\pi_{\theta_{\mathrm{old}}}$ 生成 rollout，然后在若干个 PPO epoch 中冻结 old policy。对同一个 state-action pair，定义：

$$
r_t(\theta)
=\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}.
$$

在代码里通常通过 log-probability 计算：

$$
r_t(\theta)
=\exp\left(
\log\pi_\theta(a_t\mid s_t)
-\log\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)
\right).
$$

Ratio 的含义：

| Ratio | 含义 |
|---:|---|
| $r_t=1$ | 新旧 policy 对该 token 的概率相同 |
| $r_t=1.2$ | 新 policy 将该 token 概率提高了约 $20\%$ |
| $r_t=0.7$ | 新 policy 将该 token 概率降到旧 policy 的约 $70\%$ |

未经 clipping 的 surrogate objective 是：

$$
L^{\mathrm{PG}}(\theta)
=\mathbb{E}_t[r_t(\theta)\hat A_t].
$$

如果 $\hat A_t>0$，最大化目标会增大 $r_t$；如果 $\hat A_t<0$，会减小 $r_t$。

**3. PPO Clipped Objective**

PPO 使用：

$$
\boxed{
L^{\mathrm{CLIP}}(\theta)
=\mathbb{E}_t\left[
\min\left(
 r_t(\theta)\hat A_t,
 \operatorname{clip}\left(r_t(\theta),1-\epsilon,1+\epsilon\right)\hat A_t
\right)
\right]
}
$$

训练时通常最大化 $L^{\mathrm{CLIP}}$，或等价地最小化它的负数。$\epsilon$ 决定允许的近邻区间，例如 $[1-\epsilon,1+\epsilon]$。

**4. 为什么同时需要 `clip` 和 `min`？**

不要只背“把 ratio 截断”。PPO 真正做的是取 unclipped objective 与 clipped objective 中更保守的一个：

| Advantage    | Ratio 的情况           | PPO 的行为  | 直觉                |
| ------------ | ------------------- | -------- | ----------------- |
| $\hat A_t>0$ | $r_t\le 1+\epsilon$ | 正常鼓励增大概率 | 这是好 action，可以提高概率 |
| $\hat A_t>0$ | $r_t>1+\epsilon$    | 收益被截平    | 已经提高太多，不再额外奖励继续提高 |
| $\hat A_t<0$ | $r_t\ge 1-\epsilon$ | 正常鼓励降低概率 | 这是差 action，可以降低概率 |
| $\hat A_t<0$ | $r_t<1-\epsilon$    | 收益被截平    | 已经降低太多，不再额外奖励继续降低 |

还要注意两个方向：

- 好 action 的概率若被意外降得很多，PPO 仍会推动它回来；
- 差 action 的概率若反而升得很多，PPO 仍会强烈惩罚。

因此，PPO clipping **不是强制所有 ratio 永远落在区间内**，而是去掉“沿有利方向越走越远”的额外优化激励。这是很常见的面试追问。

![Pasted image 20260826233255](/my-blog/resources/uploads/obsidian-1788716852920-8.png)

**5. PPO Clipping 与 RLHF 的 KL Penalty 不是同一件事**

这是 Day8 最容易混淆的点之一：

| 机制 | 比较谁与谁 | 时间尺度 | 主要目的 |
|---|---|---|---|
| PPO ratio / clipping | 当前 policy $\pi_\theta$ 与本轮 rollout 的 old policy $\pi_{\mathrm{old}}$ | 一轮 rollout 后的若干 optimization epochs | 防止利用同一批数据时单次局部更新过猛 |
| KL penalty | 当前或 rollout policy 与固定 reference policy $\pi_{\mathrm{ref}}$ | 整个 RLHF 训练过程 | 防止长期偏离原 SFT 模型、语言质量退化或 reward hacking |

**Old policy** 是不断更新的训练快照；**Reference model** 通常是训练开始时冻结的 SFT 模型。二者不是同一个概念，即使在训练刚开始时参数可能相同。

![Pasted image 20260901234943](/my-blog/resources/uploads/obsidian-1788716852920-9.png)



**6. PPO 的完整训练目标**

一个常见的最小化形式是：

$$
\mathcal{L}_{\mathrm{total}}
=-L^{\mathrm{CLIP}}
+c_v\mathcal{L}_V
-c_H\mathcal{H}(\pi_\theta),
$$

其中：

- $-L^{\mathrm{CLIP}}$：Actor policy loss；
- $\mathcal{L}_V$：Value Model 回归 loss；
- $\mathcal{H}(\pi_\theta)$：entropy bonus，避免分布过早塌缩；
- $c_v,c_H$：损失权重。

不同实现还可能加入 value clipping、adaptive KL controller、reward / advantage normalization、gradient clipping 等，因此面试时不要把某个代码库的具体总 loss 误认为唯一标准。

**7. 一轮 PPO-RLHF 到底发生什么？**

```text
① 从 prompt 数据集中采样一批 prompt
② 用 Old / Rollout Policy 生成 response
③ 保存每个 response token 的 old_logprob
④ 用 Reference Model 计算同一批 token 的 ref_logprob
⑤ 用 Reward Model 对完整 response 打分
⑥ 把 RM score 与 token-level KL penalty 组合为 rewards
⑦ 用 Value Model 预测每个 prefix 的 value
⑧ 计算 returns、TD errors 与 GAE advantages
⑨ 固定 rollout 数据，做若干次 minibatch PPO 更新
⑩ 更新 Actor 与 Critic；下一轮重新 rollout
```

这里的关键张量可对应为：

| 名称              | 形状示意    | 来源                             | 用途                          |
| --------------- | ------- | ------------------------------ | --------------------------- |
| `response_ids`  | $[B,T]$ | rollout policy 采样              | 实际采取的 token actions         |
| `old_logprobs`  | $[B,T]$ | old policy                     | PPO ratio 的分母               |
| `new_logprobs`  | $[B,T]$ | current policy                 | PPO ratio 的分子，可反传           |
| `ref_logprobs`  | $[B,T]$ | reference model                | 计算 KL penalty               |
| `rewards`       | $[B,T]$ | KL token reward + 终局 RM reward | 构造 return                   |
| `values`        | $[B,T]$ | value model                    | baseline 与 GAE              |
| `advantages`    | $[B,T]$ | rewards 与 values 计算            | policy update 权重            |
| `returns`       | $[B,T]$ | advantage + old value 等方式构造    | value regression target     |
| `response_mask` | $[B,T]$ | 序列边界                           | 排除 prompt、padding 与 EOS 后位置 |

![Pasted image 20260907000035](/my-blog/resources/uploads/obsidian-1788716852920-10.png)

**8. 一个容易忽略的工程事实：Rollout 很贵**

SFT 直接读取固定 token 序列，可对整个序列并行做 teacher forcing；PPO-RLHF 每轮都要让模型自回归生成 response，还要运行 reference、reward、value 等模型并存储逐 token statistics。其复杂度不仅来自公式，也来自在线采样、多个模型、长序列 KV Cache、padding 不均衡和 stale rollout 管理。这就是后面 DPO 等方法强调“训练更像监督学习”的重要背景。

**本轮面试题**

| 面试问题                                                        | 面试场景下的回答                                                                                                                                                                             |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| PPO 的 probability ratio 表示什么？                               | $r_t=\pi_\theta(a_t\mid s_t)/\pi_{\mathrm{old}}(a_t\mid s_t)$ 表示新 policy 相对采样数据的 old policy，把该 action 的概率改变了多少。代码中一般用 `exp(new_logprob-old_logprob)` 计算。                             |
| PPO 为什么要 clipping？                                          | 同一批 rollout 会被用于多次 minibatch update；如果 policy 变化太大，数据就不再近似来自当前 policy，训练容易不稳定。Clipping 对有利方向的过度概率变化截平收益，使更新保持在 old policy 附近。                                                        |
| PPO 是否保证 ratio 一定在 $[1-\epsilon,1+\epsilon]$ 内？             | 不保证。Clipped objective 只是移除超出边界后继续沿有利方向优化的激励；参数共享或其他样本的梯度仍可能让某些 ratio 超出区间。它是软限制，不是硬投影。                                                                                               |
| Old Policy 与 Reference Model 有什么区别？                         | Old Policy 是生成本轮 rollout 的 policy 快照，用于 importance ratio，通常每轮都会刷新；Reference Model 通常是冻结的 SFT 模型，用于 KL 约束，整个 RLHF 过程中保持不变。Clipping 管局部步长，Reference KL 管长期漂移。                          |
| 为什么 PPO 可以对同一批 rollout 做多个 epoch，而普通 on-policy PG 通常不适合反复用？ | PPO 用 old-policy ratio 对分布变化做重要性修正，并通过 clipping 限制过度更新，因此允许在一定范围内重复利用同一批样本；但 epoch 也不能无限增加，否则 policy 与采样分布差异仍会过大。                                                                    |
| PPO 总 loss 通常包含哪些部分？                                        | 至少包含 clipped policy objective 与 value regression loss，很多实现还加入 entropy bonus、value clipping、KL controller 等。Actor 用 Advantage 更新，Critic 拟合 Return；具体符号取决于代码是最大化 objective 还是最小化 loss。 |
| 为什么 PPO-RLHF 成本高？                                           | 它需要在线自回归 rollout，并同时使用 policy、reference、reward、value 等模型；还要保存逐 token logprobs、values、rewards 和 masks，多轮 minibatch 更新。与读取固定数据做 SFT 相比，生成和多模型前向是主要额外成本。                                |

## 第五轮：Reward Model、完整 RLHF、DPO 与 GRPO 预览

**1. Reward Model 怎样从 chosen / rejected 学出一个分数？**

给定 prompt $x$、preferred response $y_w$ 和 rejected response $y_l$，Reward Model 输出两个 scalar：

$$
r_\phi(x,y_w),\qquad r_\phi(x,y_l).
$$

Bradley–Terry 风格偏好概率为：

$$
P_\phi(y_w\succ y_l\mid x)
=\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right),
$$

对应 pairwise ranking loss：

$$
\mathcal{L}_{\mathrm{RM}}
=-\mathbb{E}_{(x,y_w,y_l)}
\left[
\log\sigma\left(r_\phi(x,y_w)-r_\phi(x,y_l)\right)
\right].
$$

如果 chosen 得分高于 rejected，差值为正，loss 变小；反之 loss 变大。模型学到的重点是**分数差与排序**，并不要求“8 分”具有固定、可跨数据集解释的绝对含义。

常见架构是在 pretrained / SFT LM backbone 上增加 scalar value head，读取 EOS 或最后一个有效 token 的 hidden state 输出一个分数；具体 pooling 和归一化方法依实现而异。

![Pasted image 20260907000257](/my-blog/resources/uploads/obsidian-1788716852920-11.png)

**2. InstructGPT 式 RLHF 三阶段**

| 阶段              | 数据                        | 训练对象                 | 结果                            |
| --------------- | ------------------------- | -------------------- | ----------------------------- |
| SFT             | 人类 demonstration          | Policy Model         | 得到能基本遵循指令的初始 policy           |
| Reward Modeling | 同一 prompt 下回答排序           | Reward Model         | 得到人类偏好的可学习 proxy              |
| PPO             | Policy rollout + RM score | Policy 与 Value Model | 最大化偏好 reward，同时限制偏离 reference |

理想化目标可写为：

$$
\max_\theta
\mathbb{E}_{x,\,y\sim\pi_\theta}
\left[r_\phi(x,y)\right]
-\beta
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot\mid x)
\Vert
\pi_{\mathrm{ref}}(\cdot\mid x)
\right).
$$

第一项鼓励高 Reward Model 分数；第二项惩罚 policy 偏离 reference。实际 rollout 中常用采样到的 token log-ratio 构造 KL penalty：

$$
r_t^{\mathrm{KL}}
=-\beta\left[
\log\pi_{\mathrm{rollout}}(a_t\mid s_t)
-
\log\pi_{\mathrm{ref}}(a_t\mid s_t)
\right].
$$

在最后一个有效 response token 再加入 sequence-level RM score。这样 GAE 可以把终局奖励和沿途 KL 成本共同转成逐 token Advantage。

**3. 为什么 KL penalty 必须存在？**

Reward Model 只是用有限偏好数据学到的 proxy。若只最大化 RM score，Policy 可能发现标注分布之外的漏洞，例如异常冗长、固定讨好句式、重复高分关键词、奇怪格式或其他让 RM 误判的模式。此时训练 reward 持续升高，人类真实评价却下降，这通常称为 reward overoptimization / reward hacking。

KL penalty 的直觉是：

> 可以从 SFT policy 出发寻找更受偏好的回答，但每走一步都要付“偏离原语言模型”的成本。

它不能彻底消除 Reward Model 偏差，但能缩小 policy 搜索到极端分布外行为的空间。

![Pasted image 20260907002413](/my-blog/resources/uploads/obsidian-1788716852920-12.png)


**4. RLHF 中四类模型再梳理一次**

| 名称 | 是否更新 | 核心作用 | 最容易混淆之处 |
|---|---|---|---|
| Policy / Actor | 更新 | 生成回答，是最终要部署的模型 | 与 old policy 是同一模型在不同时间的快照 |
| Old / Rollout Policy | 本轮 PPO epoch 内冻结 | 产生 rollout，并提供 old logprobs | 不是长期固定的 reference |
| Reference Model | 通常冻结 | 计算 KL，锚定原 SFT 行为 | 不负责预测 return |
| Reward Model | PPO 阶段通常冻结 | 给完整回答偏好分数 | 不等于 Value Model |
| Value Model / Critic | 更新 | 对每个 prefix 估计 expected return | 不直接给最终回答质量标签 |

虽然表中有五行，但常说“RLHF 四模型”时，通常指 Actor、Reference、Reward、Critic；Old Policy 是 Actor 在一次 rollout 时保存的快照，而不是必须额外永久驻留的一类语义模型。

**5. DPO 为什么可以绕过显式 Reward Model 与 PPO？**

DPO（Direct Preference Optimization）的输入是离线偏好数据：

$$
\mathcal D=\{(x,y_w,y_l)\},
$$

其中 $x$ 是 prompt，$y_w$ 是人类更喜欢的 chosen response，$y_l$ 是 rejected response。训练时保留两个模型：可训练的 $\pi_\theta$ 和冻结的 $\pi_{\mathrm{ref}}$。DPO 的关键不是简单地“提高 chosen 的概率”，而是把 **KL-regularized RLHF 的最优解关系**代入偏好模型，将原本的“训练 Reward Model，再用 PPO 优化”改写成一个可以直接优化 Policy 的二分类式 loss。

### 5.1 从 KL-regularized RLHF 目标开始

对一个给定 prompt $x$，标准目标可写为：

$$
J(\pi)
=\mathbb E_{y\sim\pi(\cdot\mid x)}[r(x,y)]
-\beta D_{\mathrm{KL}}
\left(\pi(\cdot\mid x)\Vert\pi_{\mathrm{ref}}(\cdot\mid x)\right).
$$

展开期望和 KL：

$$
J(\pi)
=\sum_y\pi(y\mid x)r(x,y)
-\beta\sum_y\pi(y\mid x)
\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}.
$$

两项分别表示：

- **Reward 项**：提高高质量回答的生成概率；
- **KL 项**：偏离 Reference Model 需要付出成本，防止模型为了追逐偏好分数而严重漂移。

直观上，这不是“从所有回答中随意寻找最高分”，而是：

> 以 Reference Model 的回答分布为起点，在它附近重新分配概率，把更多概率分给高 Reward 回答。

### 5.2 推导最优 Policy 的闭式形式

Policy 必须满足概率归一化约束：

$$
\sum_y\pi(y\mid x)=1.
$$

引入拉格朗日乘子 $\lambda(x)$：

$$
\mathcal F
=\sum_y\pi(y\mid x)
\left[
r(x,y)-\beta\log\frac{\pi(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
\right]
+\lambda(x)\left(\sum_y\pi(y\mid x)-1\right).
$$

对每个 $\pi(y\mid x)$ 求偏导并令其为零。注意：

$$
\frac{\partial}{\partial\pi}
\left[
\pi\log\frac{\pi}{\pi_{\mathrm{ref}}}
\right]
=\log\frac{\pi}{\pi_{\mathrm{ref}}}+1.
$$

因此：

$$
r(x,y)
-\beta\left(
\log\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}+1
\right)
+\lambda(x)=0.
$$

整理得到：

$$
\log\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
=\frac{r(x,y)}{\beta}+C(x),
$$

其中 $C(x)$ 只依赖 prompt，不依赖具体回答。指数化并利用概率归一化条件，可以得到：

$$
\boxed{
\pi^*(y\mid x)
=\frac{1}{Z(x)}
\pi_{\mathrm{ref}}(y\mid x)
\exp\left(\frac{r(x,y)}{\beta}\right)
}
$$

其中配分函数为：

$$
Z(x)
=\sum_y\pi_{\mathrm{ref}}(y\mid x)
\exp\left(\frac{r(x,y)}{\beta}\right).
$$

这个结果可以理解为：

$$
\text{最优 Policy 概率}
=\text{Reference 概率}
\times\text{Reward 放大系数}
\times\text{归一化系数}.
$$

- Reference 原本就认为合理的回答拥有较高起点；
- Reward 越高，$\exp(r/\beta)$ 越大，该回答的概率提升越明显；
- $Z(x)$ 只负责让所有回答的概率重新归一化为 1。

### 5.3 Policy 本身可以表示一个隐式 Reward

对最优 Policy 公式取对数并整理：

$$
\boxed{
r(x,y)
=\beta\log
\frac{\pi^*(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}
+\beta\log Z(x)
}
$$

因此，回答的 Reward 可以通过“最优 Policy 相对 Reference Model 将其概率提高了多少”来表示。忽略只依赖 $x$ 的常数后，可以定义 Current Policy 的隐式 Reward：

$$
\hat r_\theta(x,y)
=\beta\log
\frac{\pi_\theta(y\mid x)}{\pi_{\mathrm{ref}}(y\mid x)}.
$$

这就是论文标题中 “Your Language Model is Secretly a Reward Model” 的直觉：$\pi_\theta$ 与 $\pi_{\mathrm{ref}}$ 的 log-ratio 已经隐式编码了一个 Reward，而不必额外保存一个输出 scalar 的显式 Reward Model。

需要注意，Reward 本身只在“加上任意仅依赖 prompt 的常数”意义下可识别。也就是说，$r(x,y)$ 与 $r(x,y)+f(x)$ 会产生相同的同-prompt排序；偏好学习真正关心的是回答之间的 Reward 差，而不是某个绝对分数。

### 5.4 从 Bradley–Terry 偏好模型得到 DPO Loss

偏好数据通常使用 Bradley–Terry 模型描述：

$$
P(y_w\succ y_l\mid x)
=\sigma\left(r(x,y_w)-r(x,y_l)\right),
$$

其中 $\sigma(z)=1/(1+e^{-z})$。将上面的隐式 Reward 代入：

$$
\begin{aligned}
r(x,y_w)-r(x,y_l)
={}&\beta\log
\frac{\pi^*(y_w\mid x)}{\pi_{\mathrm{ref}}(y_w\mid x)}
-\beta\log
\frac{\pi^*(y_l\mid x)}{\pi_{\mathrm{ref}}(y_l\mid x)}.
\end{aligned}
$$

两个回答来自同一个 prompt，所以两边原本都有相同的 $\beta\log Z(x)$，在相减时完全抵消。这一步非常关键：**DPO 不需要显式计算巨大回答空间上的 $Z(x)$。**

用可训练的 $\pi_\theta$ 近似未知的最优 Policy $\pi^*$，定义：

$$
\Delta_\theta
=\log\pi_\theta(y_w\mid x)
-\log\pi_\theta(y_l\mid x),
$$

$$
\Delta_{\mathrm{ref}}
=\log\pi_{\mathrm{ref}}(y_w\mid x)
-\log\pi_{\mathrm{ref}}(y_l\mid x).
$$

于是模型认为 chosen 优于 rejected 的概率为：

$$
P_\theta(y_w\succ y_l\mid x)
=\sigma\left[
\beta(\Delta_\theta-\Delta_{\mathrm{ref}})
\right].
$$

对偏好数据做最大似然训练，就得到 DPO Loss：

$$
\boxed{
\mathcal L_{\mathrm{DPO}}
=-\mathbb E_{(x,y_w,y_l)\sim\mathcal D}
\left[
\log\sigma\left(
\beta(\Delta_\theta-\Delta_{\mathrm{ref}})
\right)
\right]
}
$$

### 5.5 Reference-relative margin 的直观含义

将核心差值展开：

$$
\Delta_\theta-\Delta_{\mathrm{ref}}
=\log
\frac{
\pi_\theta(y_w\mid x)/\pi_\theta(y_l\mid x)
}{
\pi_{\mathrm{ref}}(y_w\mid x)/\pi_{\mathrm{ref}}(y_l\mid x)
}.
$$

它比较的是两组“chosen 相对 rejected 的赔率”：

| 情况 | $\Delta_\theta-\Delta_{\mathrm{ref}}$ | 含义 |
|---|---:|---|
| Current 与 Reference 偏好程度相同 | $=0$ | 偏好概率为 $0.5$，说明还没有学到额外偏好 |
| Current 比 Reference 更偏向 chosen | $>0$ | 方向正确，loss 下降 |
| Current 比 Reference 更偏向 rejected | $<0$ | 排序错误，loss 增大 |

因此最值得记住的是：

> DPO 不只是要求 Current Policy 更喜欢 chosen；它要求 Current Policy **相对 Reference Model，额外增加对 chosen 而非 rejected 的偏好**。

例如，Reference 本来就满足 $\Delta_{\mathrm{ref}}=1$。如果训练后的 Current Policy 仍然是 $\Delta_\theta=1$，虽然它也更喜欢 chosen，但两者差值为 0，说明 Current Policy 并没有从这条偏好数据中获得额外改进。

### 5.6 序列 log-probability 怎样计算？

对语言模型来说，$y$ 是完整回答，它的 log-probability 等于所有 response token 条件 log-probability 的和：

$$
\log\pi_\theta(y\mid x)
=\sum_{t=1}^{T}
\log\pi_\theta(y_t\mid x,y_{<t}).
$$

代码中通常对 prompt token 和 padding token 做 mask，只累计 response 部分。因此，一条偏好样本需要计算四个序列分数：

$$
\log\pi_\theta(y_w\mid x),\quad
\log\pi_\theta(y_l\mid x),\quad
\log\pi_{\mathrm{ref}}(y_w\mid x),\quad
\log\pi_{\mathrm{ref}}(y_l\mid x).
$$

理论公式使用序列 log-probability 之和；若某个实现改用长度归一化、加权或其他聚合方式，它对应的训练偏好和长度行为也可能发生变化，不能与标准 DPO 公式混为一谈。

### 5.7 DPO 梯度具体在做什么？

记：

$$
u_\theta=\beta(\Delta_\theta-\Delta_{\mathrm{ref}}).
$$

单条数据的 loss 为 $-\log\sigma(u_\theta)$，其梯度为：

$$
\nabla_\theta\mathcal L_{\mathrm{DPO}}
=-\beta\sigma(-u_\theta)
\left[
\nabla_\theta\log\pi_\theta(y_w\mid x)
-\nabla_\theta\log\pi_\theta(y_l\mid x)
\right].
$$

使用梯度下降更新时，它会：

- 提高 chosen 的 log-probability；
- 降低 rejected 的 log-probability；
- 当模型排序错误、$u_\theta$ 较小时，$\sigma(-u_\theta)$ 较大，更新更强；
- 当 chosen 已经明显胜过 rejected 时，权重逐渐减小，避免所有样本都被无限推开。

Reference Model 只参与计算基准 margin，不接收梯度。

### 5.8 一个数值例子

假设 Reference Model 给出的序列 log-probability 为：

$$
\log\pi_{\mathrm{ref}}(y_w\mid x)=-3,
\qquad
\log\pi_{\mathrm{ref}}(y_l\mid x)=-4,
$$

因此：

$$
\Delta_{\mathrm{ref}}=(-3)-(-4)=1.
$$

如果 Current Policy 为：

$$
\log\pi_\theta(y_w\mid x)=-2.5,
\qquad
\log\pi_\theta(y_l\mid x)=-4.5,
$$

则：

$$
\Delta_\theta=(-2.5)-(-4.5)=2,
$$

$$
\Delta_\theta-\Delta_{\mathrm{ref}}=1>0.
$$

这说明Current Policy不仅喜欢chosen，而且相比Reference又把chosen相对rejected的优势扩大了1个log-probability单位。若取 $\beta=1$：

$$
P_\theta(y_w\succ y_l\mid x)=\sigma(1)\approx0.731,
$$

$$
\mathcal L=-\log(0.731)\approx0.313.
$$

如果Current Policy与Reference完全相同，则差值为0，偏好概率为0.5，单样本loss约为0.693。

### 5.9 $\beta$ 应该怎样理解？

从原始目标看：

$$
\text{Reward}-\beta\cdot\mathrm{KL}.
$$

- $\beta$ 较大：偏离Reference的代价更高；从 $\exp(r/\beta)$ 也能看出，Reward对概率分布的重加权更温和，最优Policy通常更接近Reference；
- $\beta$ 较小：Reward差异对最优分布的影响更强，理论上允许Policy离Reference更远。

但在实际DPO Loss里，$\beta$ 同时位于Sigmoid内部，既改变分类logit尺度，也改变梯度权重，并与学习率、数据难度、batch构成等共同作用。因此不要仅根据Loss中的乘法形式机械地说“$\beta$ 越大，参数更新一定越大”。

### 5.10 实际训练流程与方法边界

一次标准DPO训练step可以概括为：

```text
输入固定的 (prompt, chosen, rejected)
        ↓
Current Policy 计算 chosen / rejected 序列 log-prob
Frozen Reference 计算 chosen / rejected 序列 log-prob
        ↓
计算 Δθ、Δref 和 reference-relative margin
        ↓
计算 DPO Loss，只反向更新 Current Policy
```

因此，标准DPO训练通常不需要：

- 单独训练显式Reward Model；
- 训练Value Model / Critic；
- 在线生成rollout；
- 计算GAE和PPO ratio / clip。

但需要把边界说清楚：

- **DPO不是“没有Reward”**：Reward被隐式写成Policy与Reference的log-ratio；
- **DPO不是普通SFT**：SFT只模仿chosen，DPO同时使用chosen、rejected和Reference基准；
- **DPO不是所有RLHF的无条件等价替代**：上述推导依赖KL正则目标、Bradley–Terry偏好建模等假设；
- **DPO通常是离线方法**：训练效果受偏好数据的质量、噪声和覆盖范围限制，不能像在线RL那样随着Policy变化主动探索新回答。

> 参考论文：Rafailov et al., [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)。

![Pasted image 20260907014004](/my-blog/resources/uploads/obsidian-1788716852920-13.png)

**6. PPO 与 DPO 怎么比较？**

| 维度              | PPO-based RLHF                         | DPO                                  |
| --------------- | -------------------------------------- | ------------------------------------ |
| 训练信号            | 显式 Reward Model、规则 reward 或其他标量 reward | 成对 preference data                   |
| 是否在线 rollout    | 是，训练中持续从 policy 采样                     | 标准 DPO 通常否，使用固定 chosen / rejected 数据 |
| 显式 Reward Model | 通常需要                                   | 不需要                                  |
| Value Model     | 通常需要                                   | 不需要                                  |
| 训练复杂度           | 高，涉及采样、GAE、多模型与稳定性控制                   | 较低，接近 pairwise supervised loss       |
| 探索新回答           | 可以随着 policy 更新产生新轨迹                    | 受固定偏好数据覆盖限制                          |
| 适合的场景           | 有可计算 reward、需要在线优化或推理能力强化              | 有高质量偏好对、希望稳定高效对齐                     |
| 主要风险            | Reward hacking、训练不稳、成本高                | 偏好数据噪声与覆盖不足、离线分布限制                   |

DPO 的训练过程没有 PPO 式在线 RL，但它的推导来自 KL-constrained RLHF。面试中最稳妥的说法是：**DPO 是一种直接偏好优化方法，以监督学习式 loss 隐式优化相应的偏好 / KL 目标，而不是运行显式 Reward Model + Policy Gradient。**

**7. GRPO 只做预览：为什么可以不训练独立 Critic？**

PPO 用 Value Model 估计 baseline；GRPO 的核心想法是：对同一个 prompt 一次采样一组回答，用组内平均 reward 作为相对基准。设同一 prompt 采样 $G$ 个回答，其 reward 为 $r_1,\ldots,r_G$，可构造简化的 group-relative advantage：

$$
\hat A_i
=\frac{r_i-\operatorname{mean}(r_1,\ldots,r_G)}
{\operatorname{std}(r_1,\ldots,r_G)+\varepsilon}.
$$

高于组内平均的回答获得正 Advantage，低于平均的获得负 Advantage。然后仍可使用 PPO-style ratio 与 clipping 更新 policy。它省掉了独立 Value Model，但代价是每个 prompt 要生成多个回答，rollout 计算仍然很重；如果一组回答 reward 几乎相同，也缺少有意义的相对学习信号。

![Pasted image 20260907014353](/my-blog/resources/uploads/obsidian-1788716852920-14.png)

**8. 今天不需要深入的边界**

今天已经需要掌握 GRPO 的动机，但不要求推导所有 GRPO 变体，也不展开 RLOO、REINFORCE++、DAPO、process reward、verifier training 等。后续学习现代 reasoning RL 时，再以今天的 Policy Gradient、Advantage、ratio、KL 为共同底座进行对比，避免把每个算法学成互不相关的新公式。

**本轮面试题**

| 面试问题                              | 面试场景下的回答                                                                                                                                                                                                    |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Reward Model 为什么通常输出一个 scalar？    | 偏好标注提供的是完整回答之间的相对质量，Bradley–Terry 模型只需一个 latent utility score 就能用分数差表示 chosen 胜过 rejected 的概率。这个 scalar 是偏好 proxy，不代表绝对正确率，也不要求跨 prompt 可直接比较。                                                              |
| Reward Model 的 pairwise loss 是什么？ | 常用 $-\log\sigma(r_w-r_l)$。当 chosen 得分高于 rejected 时差值为正，概率接近 1、loss 下降；反之 loss 增大。训练关注回答对的排序差，而不是拟合人工给定的绝对分数。                                                                                                |
| RLHF 为什么需要 KL penalty？            | Reward Model 在有限数据上训练，单独最大化其分数容易把 policy 推向分布外并利用模型漏洞。KL penalty 让 policy 在提升 reward 的同时保持接近 reference SFT 模型，减轻语言能力退化与 reward hacking。                                                                     |
| DPO 为什么不需要显式 Reward Model？        | 在 KL-regularized RLHF 目标下，最优 policy 与 reward 之间存在闭式关系。把这个关系代入 Bradley–Terry 偏好模型，同一 prompt 的 partition function 会在 chosen / rejected 差分中相消，于是可直接用 current 与 reference 的序列 log-probability 构造 pairwise loss。 |
| DPO 是否完全等价于所有 RLHF？               | 不是。DPO 对应特定的 KL-regularized preference objective 与建模假设，训练通常依赖固定偏好数据；它不自动具备在线 rollout、探索任意可验证 reward 或解决所有 reward design 的能力。                                                                                |
| DPO 与 SFT 最本质的差别是什么？              | SFT 只提高 chosen / demonstration 本身的 likelihood；DPO 同时比较 chosen 与 rejected，并减去 reference model 原有的偏好差，优化的是 reference-relative preference margin。                                                              |
| GRPO 相比 PPO 为什么可以不需要 Value Model？ | 它对同一个 prompt 采样一组回答，用组内 reward 的均值和标准差构造相对 Advantage，以组内 baseline 替代学习到的 Critic。这样节省 Value Model 的参数与训练，但需要多个 rollout，且组内 reward 缺乏差异时信号会变弱。                                                                |
| PPO、DPO、GRPO 应该怎样一句话区分？           | PPO 是带 Critic、importance ratio 与 clipping 的 on-policy 优化；DPO 用固定偏好对和 reference-relative log-probability loss 直接优化偏好；GRPO 保留 PPO-style policy update，但用同 prompt 多回答的组内相对 reward 替代独立 Critic。                 |

## 今日总复盘与论文阅读

**1. 一张总表串起所有变量**

| 符号 / 张量 | 含义 | 来自哪里 | 更新谁 |
|---|---|---|---|
| $\pi_\theta(a_t\mid s_t)$ | 当前 Policy 的 token 概率 | Actor forward | Actor |
| $R(x,y)$ | 完整回答的外部评价 | RM、人类、规则或 verifier | 间接更新 Actor |
| $G_t$ | 从第 $t$ 步开始的累计 reward | rewards 累积 | Critic target / Advantage |
| $V_\psi(s_t)$ | 当前前缀的 expected return | Critic forward | Critic |
| $\hat A_t$ | action 相对预期好多少 | GAE 或 group-relative estimate | Actor 权重 |
| $\pi_{\mathrm{old}}$ | 本轮 rollout policy 快照 | Actor 拷贝 / 保存 old logprob | PPO ratio |
| $\pi_{\mathrm{ref}}$ | 冻结的 SFT reference | 训练开始前固定 | KL / DPO reference margin |
| $r_t(\theta)$ | new / old probability ratio | new 与 old logprob | PPO clipping |
| $r_\phi(x,y)$ | Reward Model 分数 | 冻结 RM | RLHF reward |
| $\beta$ | KL / reference 约束相关系数 | 超参数 | 控制 reward 与漂移平衡 |

**2. 今天必须能够口述的完整链路**

```text
LLM 是一个 token-level policy。
它从当前 policy 采样完整回答，回答得到 reward。
Policy Gradient 用 reward 或 Advantage 加权 log-probability 梯度，
从而提高高于预期 action 的概率、降低低于预期 action 的概率。
Value Model 估计每个前缀的 expected return，作为 baseline 降低方差；
GAE 将多步 TD error 组合成稳定的 Advantage estimate。
PPO 使用 new / old probability ratio，并对有利方向的过度变化做 clipping，
避免在同一批 rollout 上更新过猛。
RLHF 再加入 Reward Model 提供偏好信号、Reference Model 提供 KL 锚定。
DPO 则用 chosen / rejected 与 frozen reference 的序列 log-probability，
直接构造 reference-relative preference loss，省掉显式 RM、Critic 与在线 PPO rollout。
GRPO 保留相对 policy optimization，但用同 prompt 的组内 reward baseline 替代 Critic。
```

**3. 今日必读与阅读顺序**

| 优先级 | 论文 | 今天阅读范围 | 阅读时必须回答的问题 |
|---:|---|---|---|
| 1 | Ouyang et al., *Training Language Models to Follow Instructions with Human Feedback*, NeurIPS 2022，arXiv:2203.02155 | Abstract、Introduction、Figure 2、Methods 中 SFT / RM / PPO、主要实验结论 | 三阶段数据分别怎样收集？Policy、RM、Value、Reference 如何配合？ |
| 2 | Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv:1707.06347 | Abstract、Introduction、Section 2、clipped surrogate objective | Ratio 为什么出现？clip 与 min 分别做什么？PPO 为什么能重复利用一批 rollout？ |
| 3 | Rafailov et al., *Direct Preference Optimization*, NeurIPS 2023，arXiv:2305.18290 | Abstract、Introduction、Section 3、Figure 1 | DPO 如何从 KL-regularized RLHF 推出来？为何不需要显式 RM 与 PPO？ |
| 4 | Schulman et al., *High-Dimensional Continuous Control Using Generalized Advantage Estimation*, ICLR 2016，arXiv:1506.02438 | 只读 GAE 定义与 bias–variance 解释 | $\lambda=0$ 与 $\lambda\to1$ 各代表什么？ |
| 5 | Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*, arXiv:2402.03300 | 只看 GRPO 方法图与核心公式，作为预习 | GRPO 用什么替代 Value Model？它省了什么、又增加了什么成本？ |

**精读提醒：** 今天正式精读 InstructGPT 与 DPO；PPO 论文重点读 clipped objective，不要求读完机器人控制实验；GAE 与 DeepSeekMath 只做定向阅读。不要一上来追现代算法缩写，先确保能自己推导 `Policy Gradient → Baseline → Advantage → PPO`。

**4. 今日自测**

1. 不看笔记，写出 $s_t$、$a_t$、$\pi_\theta$ 在 LLM 中的含义。
2. 从 $J(\theta)=\sum_y\pi_\theta(y)R(y)$ 推到 $\mathbb{E}[R\nabla\log\pi]$。
3. 用自己的话解释为什么减去 state baseline 不改变期望梯度。
4. 写出 $V$、$Q$、$A$ 的定义，并给一个正 Advantage 的语言生成例子。
5. 写出 TD error 与 GAE，解释 $\lambda$ 的 bias–variance trade-off。
6. 画出 PPO 中 $A>0$ 与 $A<0$ 的 clipping 情况。
7. 明确区分 `old_logprobs` 与 `ref_logprobs`。
8. 从 Prompt 开始，完整口述一次 PPO-RLHF rollout 到 optimizer step。
9. 写出 Reward Model pairwise loss，并解释为什么只关心分数差。
10. 写出 DPO loss 中 $\Delta_\theta-\Delta_{\mathrm{ref}}$ 的直觉。
11. 一句话解释 GRPO 为什么不需要独立 Value Model。

**5. 最后检查：下面这些话能否判断对错？**

| 说法 | 判断 | 原因 |
|---|---|---|
| “Reward 越高，回答里的每个 token 都一定是正确的。” | 错 | sequence reward 无法天然完成细粒度 credit assignment |
| “Value Model 就是输出分数的 Reward Model。” | 错 | RM 评价完整回答；Value 预测某个前缀的未来 return |
| “PPO clip 会强制所有 ratio 都不越界。” | 错 | 它是 surrogate objective 的软限制，不是参数硬投影 |
| “Old Policy 与 Reference Model 永远是同一个模型。” | 错 | Old 是每轮 rollout 快照，Reference 通常长期冻结 |
| “DPO 只需要 chosen，不需要 rejected。” | 错 | 标准 DPO 使用成对偏好差 |
| “GRPO 完全没有 rollout 成本。” | 错 | 它通常需要同一 prompt 生成多个回答，只是省掉 Critic |
| “KL 越大一定越好。” | 错 | 太强会阻止学习，太弱又可能导致漂移与 reward hacking |
