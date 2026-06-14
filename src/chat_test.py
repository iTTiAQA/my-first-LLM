"""
============================================================
🤖 GPT 聊天测试脚本（中文字符级模型）
============================================================
用法：
    python chat_test.py

    所有参数已在脚本中预设好，直接运行即可。
    模型默认加载 checkpoints/model_epoch_3.pt
============================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math
import os
import pickle


# ---- 预设参数（与 notebook 一致）----
CKPT_PATH = "./src/video/checkpoints/sft_model_epoch_9.pt"   # 使用 epoch 9
TEMPERATURE = 0.8
TOP_K = 100
MAX_TOKENS = 128

# 预设测试 prompts（可自行修改）
TEST_PROMPTS = [
    "怎么看今天天气",
    "神雕侠侣",
    "国足为什么没进世界杯",
    "讲一下蝰蛇",
]

# ============================================================
# 1. 模型定义（与 build_gpt.ipynb 完全一致）
# ============================================================

@dataclass
class GPTConfig:
    block_size: int = 1024  # 中文需要更长上下文
    batch_size: int = 12
    n_layer: int = 6
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    vocab_size: int = 13005  # 中文字符级词表

    def __post_init__(self):
        self.head_size = self.n_embd // self.n_head


class SingleHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config.n_embd, config.head_size)
        self.value = nn.Linear(config.n_embd, config.head_size)
        self.query = nn.Linear(config.n_embd, config.head_size)
        self.head_size = config.head_size
        self.register_buffer(
            'attention_mask',
            torch.tril(torch.ones(config.block_size, config.block_size))
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        batch_size, seq_len, hidden_size = x.size()
        k = self.key(x)
        v = self.value(x)
        q = self.query(x)
        weight = q @ k.transpose(-2, -1)
        weight = weight.masked_fill(
            self.attention_mask[:seq_len, :seq_len] == 0, float('-inf')
        ) / math.sqrt(self.head_size)
        weight = F.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        return weight @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([SingleHeadAttention(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        output = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(output))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 注意：属性名与 train.py / checkpoint 保持一致
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_final = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        batch, seq_len = idx.size()
        token_emb = self.token_emb(idx)
        pos_emb = self.pos_emb(torch.arange(seq_len, device=idx.device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_final(x)
        logits = self.lm_head(x)

        if targets is None:
            return logits, None
        batch, seq_len, vocab_size = logits.size()
        logits = logits.view(batch * seq_len, vocab_size)
        targets = targets.view(batch * seq_len)
        loss = F.cross_entropy(logits, targets)
        return logits, loss


# ============================================================
# 2. 中文字符级 Tokenizer
# ============================================================

def load_tokenizer():
    """加载中文字符级 tokenizer（与 notebook 一致）"""
    notebook_dir = os.path.dirname(os.path.abspath(__file__))  # src/video/
    project_root = os.path.dirname(os.path.dirname(notebook_dir))  # LLMs-Zero-to-Hero-master/
    tokenizer_path = os.path.join(project_root, "data", "tokenizer_zh_char.pkl")
    with open(tokenizer_path, 'rb') as f:
        tk_data = pickle.load(f)
    return tk_data

# 全局加载
tk_data = load_tokenizer()
stoi = tk_data['stoi']
itos = tk_data['itos']
unk_id = tk_data['unk_id']


def encode(text):
    """中文字符级编码"""
    return [stoi.get(ch, unk_id) for ch in text]


def decode(ids):
    """中文字符级解码"""
    return ''.join(itos.get(i, '<unk>') for i in ids)


# ============================================================
# 3. 生成 / 聊天函数
# ============================================================

def generate(model, prompt, max_tokens=100, temperature=0.6, top_k=50, device="cuda"):
    """用训练好的模型生成文本续写"""
    input_ids = encode(prompt)
    x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    generated = []
    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            x_cond = x if x.size(1) <= GPTConfig.block_size else x[:, -GPTConfig.block_size:]
            logits, _ = model(x_cond)
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float('-inf')

            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, idx_next), dim=1)
            generated.append(idx_next.item())

    return decode(generated)


def interactive_chat(model, device):
    """交互式聊天循环"""
    print("\n" + "=" * 60)
    print("🤖 交互式聊天模式（中文字符级模型）")
    print("   输入 'quit' 或 'exit' 退出")
    print("   参数: temperature=0.6, top_k=50, max_tokens=512")
    print("=" * 60)

    while True:
        try:
            prompt = input("\n💬 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if prompt.lower() in ("quit", "exit", "q"):
            print("👋 再见！")
            break
        if not prompt:
            continue

        print("🤖 模型: ", end="", flush=True)
        reply = generate(model, prompt,
                         max_tokens=MAX_TOKENS,
                         temperature=TEMPERATURE,
                         top_k=TOP_K,
                         device=device)
        print(reply)


# ============================================================
# 4. 主函数
# ============================================================
def main():
    # 设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  设备: {device}")

    # 加载模型
    if not os.path.exists(CKPT_PATH):
        print(f"❌ 找不到 checkpoint: {CKPT_PATH}")
        print(f"   请确认 checkpoints/ 目录下有对应的模型文件")
        return

    print(f"📦 加载模型: {CKPT_PATH}")
    config = GPTConfig()
    model = GPT(config).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    # checkpoint 中 key 可能是 'model_state_dict' 或 'model'
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"   ✅ 加载成功！参数量: {total_params:.2f}M")
    print(f"   Epoch: {ckpt.get('epoch', -1) + 1}")
    if 'val_loss' in ckpt:
        print(f"   验证 Loss: {ckpt['val_loss']:.4f}")

    # Tokenizer
    print(f"   Tokenizer: 中文字符级 (vocab_size={tk_data['vocab_size']})")

    # ---- 先跑预设测试 ----
    print("\n" + "=" * 60)
    print("🧪 预设 Prompt 测试")
    print("=" * 60)

    for prompt in TEST_PROMPTS:
        print(f"\n💬 输入: {prompt}")
        reply = generate(model, prompt,
                         max_tokens=MAX_TOKENS,
                         temperature=TEMPERATURE,
                         top_k=TOP_K,
                         device=device)
        print(f"🤖 输出: {reply}")
        print("-" * 40)

    print("\n✅ 预设测试完成！")
    print(f"💡 参数: temperature={TEMPERATURE}, top_k={TOP_K}, max_tokens={MAX_TOKENS}")

    # ---- 进入交互式聊天 ----
    interactive_chat(model, device)


if __name__ == "__main__":
    main()
