"""构建中文字符级 Tokenizer（无需训练，瞬间完成）

对于中文来说，字符级 tokenizer 是最自然高效的选择：
- 每个汉字 = 1 个 token（GPT-2 tokenizer 需要 2-3 个 token）
- 词表约 10000 个字符即可覆盖 99.9% 的中文语料
- 不需要训练，直接从数据中提取字符集
"""
import json
import os
import pickle


def build_char_tokenizer(data_path, save_path, max_lines=100000):
    """从 JSONL 数据中提取字符集，构建字符级 tokenizer"""
    chars = set()
    total_chars = 0

    print(f'扫描数据提取字符集（最多 {max_lines} 行）...')
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            try:
                obj = json.loads(line.strip())
                text = obj['text']
                chars.update(text)
                total_chars += len(text)
            except Exception:
                continue

    # 排序：特殊 token 在前，常用字符在前
    specials = ['<unk>', '<s>', '</s>', '<pad>']
    # 按频率排序（简化版：按 Unicode 码点）
    sorted_chars = sorted(chars)

    # 构建词表映射
    stoi = {}  # string -> id
    itos = {}  # id -> string

    for i, tok in enumerate(specials):
        stoi[tok] = i
        itos[i] = tok

    for i, ch in enumerate(sorted_chars):
        idx = i + len(specials)
        stoi[ch] = idx
        itos[idx] = ch

    vocab_size = len(stoi)
    print(f'词表大小: {vocab_size} (特殊token {len(specials)} + 字符 {len(sorted_chars)})')
    print(f'扫描总字符数: {total_chars:,}')
    print(f'唯一字符数: {len(sorted_chars)}')

    # 保存
    tokenizer_data = {
        'stoi': stoi,
        'itos': itos,
        'vocab_size': vocab_size,
        'unk_token': '<unk>',
        'bos_token': '<s>',
        'eos_token': '</s>',
        'pad_token': '<pad>',
        'unk_id': stoi['<unk>'],
        'bos_id': stoi['<s>'],
        'eos_id': stoi['</s>'],
        'pad_id': stoi['<pad>'],
    }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(tokenizer_data, f)

    print(f'\n已保存: {save_path}')

    # 测试
    test_texts = ['晋太元中，武林人捕鱼为业','人工智能是计算机科学的一个分支','东边日出西边雨，道是无晴却有晴',]

    print('\n--- 编码测试 ---')
    for t in test_texts:
        ids = [stoi.get(ch, stoi['<unk>']) for ch in t]
        decoded = ''.join(itos.get(i, '<unk>') for i in ids)
        print(f'  "{t}" -> {len(ids)} tokens (每字 {len(ids)/len(t):.2f})')
        print(f'    解码: "{decoded}"')

    return tokenizer_data


if __name__ == '__main__':
    data_path = r'..\data\wiki_zh_full.jsonl'
    save_path = r'..\data\tokenizer_zh_char.pkl'
    build_char_tokenizer(data_path, save_path, max_lines=50000)
