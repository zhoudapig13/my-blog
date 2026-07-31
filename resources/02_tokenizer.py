"""实验 2：使用真实的 Qwen Tokenizer 处理不同类型的文本。"""

# AutoTokenizer 会根据模型名称自动加载与该模型配套的 tokenizer。
from transformers import AutoTokenizer


# Hugging Face 模型仓库名称。
# 本实验只加载 tokenizer 文件，不会加载模型权重。
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def show_one(tokenizer, text):
    """打印一段文本的 token、token ID、数量和解码结果。"""
    # tokenize() 返回切分后的 token 字符串，主要供人观察。
    tokens = tokenizer.tokenize(text)

    # encode() 返回模型实际接收的整数 ID。
    # add_special_tokens=False 表示暂时不添加特殊 token，便于观察正文。
    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )

    print("-" * 70)
    print("文本：", text)
    print("tokens：", tokens)
    print("token IDs：", token_ids)
    print("token 数：", len(token_ids))

    # decode() 把一组 token ID 转换回可读文本。
    decoded_text = tokenizer.decode(token_ids)
    print("解码：", decoded_text)


def main():
    # 下载或读取缓存中的 tokenizer 配置、词表和切分规则。
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # len(tokenizer) 是完整词表中的 token 数量。
    print("词表大小：", len(tokenizer))

    # eos_token 是序列结束标记；eos_token_id 是它在词表中的编号。
    print(
        "EOS token / ID：",
        repr(tokenizer.eos_token),
        tokenizer.eos_token_id,
    )

    # 使用多种文本观察同一个 tokenizer 的不同切分结果。
    samples = [
        "大语言模型正在改变 NLP, and it is exciting!",
        "Machine learning is interesting.",
        "机器学习很有趣。",
        "ChatGPT-2026",
        "😂🚀",
        "龘靐齉",
    ]

    for text in samples:
        show_one(tokenizer, text)

    # batch 中的两个句子 token 数不同。
    batch_texts = [
        "短句。",
        "这是一个明显更长的句子，用于观察 padding。",
    ]

    # padding=True：把短句补到与最长句相同的长度。
    # return_tensors="pt"：返回 PyTorch 张量，而不是普通列表。
    batch = tokenizer(
        batch_texts,
        padding=True,
        return_tensors="pt",
    )

    # input_ids 保存 token ID；形状是 [batch_size, sequence_length]。
    print("\nBatch input_ids：")
    print(batch["input_ids"])

    # attention_mask 中 1 表示有效 token，0 表示 padding 位置。
    print("Batch attention_mask：")
    print(batch["attention_mask"])

    # tuple() 把 torch.Size([2, 11]) 显示成更直观的 (2, 11)。
    batch_shape = tuple(batch["input_ids"].shape)
    print("Batch shape [B,L]：", batch_shape)


if __name__ == "__main__":
    main()


# ============================== 实际运行输出 ==============================
# 词表大小：151665
# EOS token / ID：'<|im_end|>' 151645
#
# 文本：Machine learning is interesting.
# tokens：['Machine', 'Ġlearning', 'Ġis', 'Ġinteresting', '.']
# token IDs：[21605, 6832, 374, 7040, 13]
# token 数：5
# 解码：Machine learning is interesting.
#
# 文本：机器学习很有趣。
# token IDs：[102182, 100134, 99165, 103027, 1773]
# token 数：5
# 解码：机器学习很有趣。
#
# 文本：ChatGPT-2026
# token IDs：[15672, 38, 2828, 12, 17, 15, 17, 21]
# token 数：8
# 解码：ChatGPT-2026
#
# 文本：😂🚀
# token IDs：[144185, 145836]
# token 数：2
# 解码：😂🚀
#
# 文本：龘靐齉
# token IDs：[82912, 246, 21012, 238, 122125]
# token 数：5
# 解码：龘靐齉
#
# Batch attention_mask：
# tensor([[1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#         [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
# Batch shape [B,L]：(2, 11)
#
# 说明：token 中类似“å¤§”的内容是 Byte-level tokenizer 的内部显示形式；
# decode() 的结果仍能正确还原原始中文。
