"""实验 5：运行真实 Qwen 的 Tokenizer → Embedding → logits → 文本生成。"""

import torch

# AutoTokenizer：加载与模型配套的 tokenizer。
# AutoModelForCausalLM：加载用于自回归文本生成的语言模型。
# set_seed：固定 Sampling 使用的随机种子。
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
TEXT = "大语言模型正在改变自然语言处理。"


def choose_device_and_dtype():
    """选择当前机器能够使用的计算设备和数据类型。"""
    if torch.cuda.is_available():
        # 有 NVIDIA GPU 时使用 CUDA 和 FP16，以节省显存并加速。
        return "cuda", torch.float16

    # CPU 上使用 FP32，兼容性通常更好。
    return "cpu", torch.float32


def print_tokenizer_result(tokenizer, inputs):
    """打印正文的 token、ID 和输入形状。"""
    tokens = tokenizer.tokenize(TEXT)
    input_ids = inputs["input_ids"]

    print("\n[1] Tokenizer")
    print("原始文本：", TEXT)
    print("tokens：", tokens)
    print("input_ids：", input_ids.tolist())
    print("input_ids shape [B,L]：", tuple(input_ids.shape))


def print_embedding_result(model, input_ids):
    """直接调用模型的输入 Embedding 层，观察查表结果。"""
    # 取得模型真实使用的输入 Embedding 层。
    embedding_layer = model.get_input_embeddings()

    # 使用 token ID 查找对应向量。
    embeddings = embedding_layer(input_ids)

    print("\n[2] Embedding")
    print("weight shape [V,d]：", tuple(embedding_layer.weight.shape))
    print("output shape [B,L,d]：", tuple(embeddings.shape))


def print_logits_result(model, tokenizer, inputs):
    """执行一次模型前向传播，并查看最后位置的候选 token。"""
    # **inputs 会把字典展开为 input_ids=...、attention_mask=...。
    outputs = model(**inputs)

    # logits 中保存每个序列位置对完整词表的原始预测分数。
    logits = outputs.logits

    # [:, -1, :] 表示：
    # 保留所有 batch，只取最后一个序列位置，保留全部词表分数。
    last_token_logits = logits[:, -1, :]

    # Softmax 将最后位置的词表分数转换为概率。
    last_token_probs = torch.softmax(last_token_logits, dim=-1)

    # 取概率最高的五个候选及其 token ID。
    top_probs, top_ids = torch.topk(
        last_token_probs,
        k=5,
        dim=-1,
    )

    print("\n[3] LM Head")
    print("logits shape [B,L,V]：", tuple(logits.shape))
    print("概率最高的 5 个下一 token 候选：")

    # top_probs[0] 和 top_ids[0] 表示 batch 中的第一个样本。
    for probability, token_id_tensor in zip(top_probs[0], top_ids[0]):
        token_id = token_id_tensor.item()
        probability_value = probability.item()

        # decode() 把单个候选 ID 转换成可读字符或文本片段。
        token_text = tokenizer.decode([token_id])

        print(
            f"候选={token_text!r}, "
            f"id={token_id}, "
            f"p={probability_value:.6f}"
        )


def build_generation_inputs(tokenizer, device):
    """根据 Qwen 的聊天模板构造生成任务输入。"""
    messages = [
        {
            "role": "user",
            "content": "用两句话解释 Tokenizer 的作用。",
        }
    ]

    # apply_chat_template() 会添加模型要求的角色和特殊 token 格式。
    # tokenize=False 表示这里先得到格式化后的字符串。
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 再将格式化后的 prompt 转换为 PyTorch 张量并移动到模型设备。
    generation_inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(device)

    return generation_inputs


def decode_new_tokens(tokenizer, output_ids, prompt_length):
    """从 generate() 的完整序列中只解码新生成的部分。"""
    # generate() 返回“原 prompt + 新生成 token”。
    # 从 prompt_length 开始切片，就只留下模型新生成的 token。
    new_token_ids = output_ids[0, prompt_length:]

    return tokenizer.decode(
        new_token_ids,
        skip_special_tokens=True,
    )


def generate_text(model, tokenizer, device):
    """分别执行 Greedy 和 Sampling，并打印生成结果。"""
    generation_inputs = build_generation_inputs(tokenizer, device)
    prompt_length = generation_inputs["input_ids"].shape[1]

    # do_sample=False 表示不进行随机采样。
    # 模型每一步都选择当前分数最高的 token。
    greedy_output_ids = model.generate(
        **generation_inputs,
        max_new_tokens=60,
        do_sample=False,
    )

    greedy_text = decode_new_tokens(
        tokenizer,
        greedy_output_ids,
        prompt_length,
    )

    # 固定随机种子，让同一环境中的 Sampling 更容易复现。
    set_seed(42)

    # do_sample=True 表示从处理后的概率分布中随机抽取 token。
    sampled_output_ids = model.generate(
        **generation_inputs,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.05,
    )

    sampled_text = decode_new_tokens(
        tokenizer,
        sampled_output_ids,
        prompt_length,
    )

    print("\n[4] Greedy：")
    print(greedy_text)
    print("\n[5] Sampling：")
    print(sampled_text)


# inference_mode() 关闭本实验不需要的梯度记录，减少内存开销。
@torch.inference_mode()
def main():
    device, dtype = choose_device_and_dtype()
    print("device / dtype：", device, dtype)

    # 加载 tokenizer。首次运行需要从模型仓库下载相关文件。
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 加载模型权重，设置数据类型，然后移动到 CPU 或 GPU。
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=dtype,
    )
    model = model.to(device)

    # eval() 切换为推理模式，关闭 Dropout 等训练行为。
    model.eval()

    # 将实验文本转换为张量，并移动到模型所在设备。
    inputs = tokenizer(
        TEXT,
        return_tensors="pt",
    ).to(device)

    print_tokenizer_result(tokenizer, inputs)
    print_embedding_result(model, inputs["input_ids"])
    print_logits_result(model, tokenizer, inputs)
    generate_text(model, tokenizer, device)


if __name__ == "__main__":
    main()
