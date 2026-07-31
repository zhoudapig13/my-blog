"""实验 4：从 logits 计算概率，并执行 Greedy、Top-k、Top-p 和采样。"""

import torch


def top_k_filter(logits, k):
    """只保留 logits 中分数最高的 k 个候选。"""
    # torch.topk(logits, k).values 返回最高的 k 个分数。
    top_k_values = torch.topk(logits, k).values

    # 最高 k 项中的最后一项，就是允许保留的最低分数。
    threshold = top_k_values[-1]

    # 分数大于等于 threshold 的位置保留原值。
    # 其他位置设为负无穷，经过 Softmax 后概率会变成 0。
    negative_infinity = torch.tensor(float("-inf"))
    filtered_logits = torch.where(
        logits >= threshold,
        logits,
        negative_infinity,
    )
    return filtered_logits


def top_p_filter(logits, p):
    """保留累计概率达到 p 所需要的最小候选集合。"""
    # 先按照 logit 从大到小排序，同时保存原位置索引。
    sorted_logits, sorted_indices = torch.sort(
        logits,
        descending=True,
    )

    # 在排序后的 logits 上计算概率。
    sorted_probs = torch.softmax(sorted_logits, dim=-1)

    # cumsum() 计算从第一个候选开始的累计概率。
    cumulative_probs = sorted_probs.cumsum(dim=-1)

    # 减去当前项概率，相当于查看“加入当前项之前”的累计概率。
    # 这样可以保留使累计概率首次达到 p 的那个候选。
    remove_mask = cumulative_probs - sorted_probs >= p

    # 被移除的候选设为负无穷。
    sorted_logits[remove_mask] = float("-inf")

    # 创建与原 logits 相同形状、全部为负无穷的张量。
    filtered_logits = torch.full_like(logits, float("-inf"))

    # 根据排序时保存的索引，把结果放回原来的位置。
    filtered_logits = filtered_logits.scatter(
        dim=0,
        index=sorted_indices,
        src=sorted_logits,
    )
    return filtered_logits


def main():
    # 假设词表中有四个候选 token，模型给出了四个原始分数。
    logits = torch.tensor([4.0, 2.5, 1.5, 0.5])
    print("原始 logits：", logits.tolist())

    # argmax 返回最大值的位置。这里最大值 4.0 的索引为 0。
    greedy_index = torch.argmax(logits).item()
    print("Greedy 索引：", greedy_index)

    # 温度通过 logits / temperature 改变 Softmax 概率的集中程度。
    for temperature in [0.5, 1.0, 2.0]:
        scaled_logits = logits / temperature
        probabilities = torch.softmax(scaled_logits, dim=-1)

        # 为方便阅读，将概率保留 6 位小数。
        rounded_probs = [
            round(probability, 6)
            for probability in probabilities.tolist()
        ]
        print(f"T={temperature}：", rounded_probs)

    # Top-k=2：只保留分数最高的两个候选。
    top_k_logits = top_k_filter(logits, k=2)
    top_k_probs = torch.softmax(top_k_logits, dim=-1)
    print("Top-k=2：", top_k_probs.tolist())

    # Top-p=0.9：保留累计概率达到 0.9 的最小候选集合。
    top_p_logits = top_p_filter(logits, p=0.9)
    top_p_probs = torch.softmax(top_p_logits, dim=-1)
    print("Top-p=0.9：", top_p_probs.tolist())

    # 固定随机种子，使当前环境中的采样结果可以复现。
    torch.manual_seed(42)

    sampling_probs = torch.softmax(logits / 0.7, dim=-1)

    # replacement=True 允许同一个索引被重复抽中。
    sampled_indices = torch.multinomial(
        sampling_probs,
        num_samples=10,
        replacement=True,
    )
    print("10 次采样索引：", sampled_indices.tolist())


if __name__ == "__main__":
    main()


# ============================== 实际运行输出 ==============================
# 原始 logits：[4.0, 2.5, 1.5, 0.5]
# Greedy 索引：0
# T=0.5：[0.945683, 0.047083, 0.006372, 0.000862]
# T=1.0：[0.748832, 0.167087, 0.061468, 0.022613]
# T=2.0：[0.517426, 0.244415, 0.148245, 0.089915]
# Top-k=2：[0.8175744414, 0.1824255139, 0.0, 0.0]
# Top-p=0.9：[0.8175744414, 0.1824255139, 0.0, 0.0]
# 10 次采样索引：[0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
#
# 采样结果由随机数和运行环境共同决定；本例通过 torch.manual_seed(42)
# 固定了当前环境中的结果。
