"""实验 1：不依赖第三方 tokenizer，从零实现一个简化版 BPE。"""

# Counter 是具有自动计数能力的字典。
# 例如 counts[("e", "s")] += 3 会把 ("e", "s") 的计数增加 3。
from collections import Counter


# 训练语料的简化表示：键是单词，值是单词在语料中出现的次数。
WORD_FREQUENCIES = {
    "low": 5,
    "lower": 2,
    "newest": 6,
    "widest": 3,
}


def initial_corpus(word_frequencies):
    """把每个单词拆成字符，并在末尾添加词尾标记 </w>。"""
    corpus = {}

    # items() 每次返回一个“单词、词频”组合。
    for word, frequency in word_frequencies.items():
        # list("low") 得到 ["l", "o", "w"]。
        characters = list(word)

        # </w> 表示单词在这里结束。
        symbols = characters + ["</w>"]

        # 列表不能作为字典键，因此转换成不可变的元组。
        corpus[tuple(symbols)] = frequency

    return corpus


def count_pairs(corpus):
    """统计语料中每一种相邻 token 对的加权出现次数。"""
    pair_counts = Counter()

    for symbols, frequency in corpus.items():
        # 假设 symbols 是 ("l", "o", "w", "</w>")：
        # symbols[1:] 是 ("o", "w", "</w>")；
        # zip 后得到 ("l", "o")、("o", "w")、("w", "</w>")。
        adjacent_pairs = zip(symbols, symbols[1:])

        for pair in adjacent_pairs:
            # 这里加的是单词词频，而不是简单加 1。
            # low 出现 5 次，所以它的每一个相邻对都贡献 5 次。
            pair_counts[pair] += frequency

    return pair_counts


def merge_pair(corpus, target_pair):
    """把语料中所有 target_pair 合并成一个新 token。"""
    merged_corpus = {}

    for symbols, frequency in corpus.items():
        # result 保存当前单词合并后的 token。
        result = []
        index = 0

        # 使用索引从左到右检查当前单词。
        while index < len(symbols):
            # 必须保证右边还有一个 token，才能取出相邻的一对。
            has_next_token = index + 1 < len(symbols)

            # 取当前位置和下一个位置组成的 token 对。
            current_pair = symbols[index : index + 2]

            if has_next_token and current_pair == target_pair:
                # 例如 ("e", "s") 会通过 join 合并成 "es"。
                new_token = "".join(target_pair)
                result.append(new_token)

                # 一次已经处理了两个 token，所以索引向后移动两位。
                index += 2
            else:
                # 如果当前位置没有匹配目标 token 对，就原样保留。
                result.append(symbols[index])
                index += 1

        # 保留原单词的词频，只改变它的 token 切分方式。
        merged_corpus[tuple(result)] = frequency

    return merged_corpus


def print_corpus(corpus):
    """按照便于阅读的形式打印当前语料。"""
    for symbols, frequency in corpus.items():
        # join 用空格连接 token；:25s 表示至少占 25 个字符宽度。
        readable_symbols = " ".join(symbols)
        print(f"  {readable_symbols:25s} 词频={frequency}")


def main(num_merges=8):
    # 第一步：把原始单词转换成字符级语料。
    corpus = initial_corpus(WORD_FREQUENCIES)

    print("初始语料：")
    print_corpus(corpus)

    # range(1, 9) 会依次产生 1 到 8，一共执行 8 次合并。
    for step in range(1, num_merges + 1):
        # 重新统计当前语料中的所有相邻 token 对。
        pair_counts = count_pairs(corpus)

        # most_common(1) 返回频次最高的一项。
        # [0] 取出该项，再拆成“token 对”和“频次”。
        best_pair, count = pair_counts.most_common(1)[0]

        # 将本轮最高频 token 对在整个语料中全部合并。
        corpus = merge_pair(corpus, best_pair)

        print(f"第 {step} 次：合并 {best_pair}，加权频次={count}")

    print("\n最终切分：")
    print_corpus(corpus)


# 直接运行本文件时 __name__ 等于 "__main__"，因此调用 main()。
# 如果其他文件只是 import 本文件，则不会自动执行实验。
if __name__ == "__main__":
    main()


# ============================== 实际运行输出 ==============================
# 初始语料：
#   l o w </w>                词频=5
#   l o w e r </w>            词频=2
#   n e w e s t </w>          词频=6
#   w i d e s t </w>          词频=3
# 第 1 次：合并 ('e', 's')，加权频次=9
# 第 2 次：合并 ('es', 't')，加权频次=9
# 第 3 次：合并 ('est', '</w>')，加权频次=9
# 第 4 次：合并 ('l', 'o')，加权频次=7
# 第 5 次：合并 ('lo', 'w')，加权频次=7
# 第 6 次：合并 ('n', 'e')，加权频次=6
# 第 7 次：合并 ('ne', 'w')，加权频次=6
# 第 8 次：合并 ('new', 'est</w>')，加权频次=6
#
# 最终切分：
#   low </w>                  词频=5
#   low e r </w>              词频=2
#   newest</w>                词频=6
#   w i d est</w>             词频=3
