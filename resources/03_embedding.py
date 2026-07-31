"""实验 3：用一个小矩阵观察 Embedding 的查表过程。"""

import torch

# nn 中包含 Embedding、Linear 等神经网络层。
from torch import nn


def main():
    # 固定随机种子，使 Embedding 的随机初始值每次运行都相同。
    torch.manual_seed(42)

    vocab_size = 10
    embedding_dim = 4

    # 创建形状为 [10, 4] 的 Embedding 权重矩阵。
    # num_embeddings：词表中一共有 10 个 token。
    # embedding_dim：每个 token 使用 4 维向量表示。
    # padding_idx=0：将 ID 0 作为 padding，它的初始向量为全 0。
    embedding = nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
        padding_idx=0,
    )

    # 两行表示两个样本，每行包含三个 token ID。
    # 因此输入形状为 [B, L] = [2, 3]。
    input_ids = torch.tensor(
        [
            [2, 5, 8],
            [4, 1, 0],
        ]
    )

    # 把 input_ids 传给 Embedding。
    # 对每个 ID 查找权重矩阵的对应行。
    output = embedding(input_ids)

    print("weight shape [V,d]：", tuple(embedding.weight.shape))
    print("input shape [B,L]：", tuple(input_ids.shape))
    print("output shape [B,L,d]：", tuple(output.shape))
    print("第一个样本：")
    print(output[0])

    # input_ids[0, 1] 是第一个样本的第二个 token ID，值为 5。
    # item() 把只含一个数的张量转换为普通 Python 整数。
    token_id = input_ids[0, 1].item()

    # 这是 Embedding 层为 ID 5 返回的查表结果。
    lookup_result = output[0, 1]

    # 直接读取 Embedding 权重矩阵中索引为 5 的行。
    weight_row = embedding.weight[token_id]

    print("\ntoken ID：", token_id)
    print("查表输出：", lookup_result)
    print("权重对应行：", weight_row)

    # torch.equal 要求两个张量的形状、数据类型和每一个值都相同。
    print("完全一致：", torch.equal(lookup_result, weight_row))

    # numel() 返回张量的元素总数，即 10 × 4 = 40。
    print("参数量 V*d：", embedding.weight.numel())


if __name__ == "__main__":
    main()

