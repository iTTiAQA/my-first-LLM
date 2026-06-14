"""
============================================================
SFT 微调脚本（优化版）
用法: python sft_train.py

优化要点:
  - 学习率 5e-5（原来 2e-5）
  - Epoch 10（原来 3）
  - 10% warmup 避免初期震荡
  - 数据集 70 条，覆盖 AI/ML/DL/Python 等领域
============================================================
"""
import torch, torch.nn as nn, torch.nn.functional as F, time, os, json, pickle, math
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass

torch.manual_seed(1024)


# ==================== Config ====================
@dataclass
class GPTConfig:
    block_size: int = 1024
    batch_size: int = 12
    n_layer: int = 6
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.1
    vocab_size: int = 13005

    def __post_init__(self):
        self.head_size = self.n_embd // self.n_head


# ==================== Model ====================
class SingleHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.key = nn.Linear(config.n_embd, config.head_size)
        self.value = nn.Linear(config.n_embd, config.head_size)
        self.query = nn.Linear(config.n_embd, config.head_size)
        self.head_size = config.head_size
        self.register_buffer('attention_mask',
                             torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, _ = x.size()
        k, v, q = self.key(x), self.value(x), self.query(x)
        w = q @ k.transpose(-2, -1)
        w = w.masked_fill(self.attention_mask[:T, :T] == 0, float('-inf')) / math.sqrt(self.head_size)
        w = self.dropout(F.softmax(w, dim=-1))
        return w @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([SingleHeadAttention(config) for _ in range(config.n_head)])
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd), nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd), nn.Dropout(config.dropout))

    def forward(self, x): return self.net(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x):
        return x + self.ffn(self.ln2(x + self.att(self.ln1(x))))


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_final = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        x = self.token_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        x = self.blocks(x)
        logits = self.lm_head(self.ln_final(x))
        if targets is None:
            return logits, None
        return logits, F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))


# ==================== SFT Forward ====================
def sft_forward(model, input_ids, labels):
    """SFT 专用 forward：只对 labels != -100 计算 loss"""
    batch, seq_len = input_ids.size()
    token_emb = model.token_emb(input_ids)
    pos_emb = model.pos_emb(torch.arange(seq_len, device=input_ids.device))
    x = token_emb + pos_emb
    x = model.blocks(x)
    x = model.ln_final(x)
    logits = model.lm_head(x)

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    shift_labels = shift_labels.view(-1)

    mask = shift_labels != -100
    if mask.sum() == 0:
        return logits, torch.tensor(0.0, device=input_ids.device, requires_grad=True)

    active_logits = shift_logits[mask]
    active_labels = shift_labels[mask]
    loss = F.cross_entropy(active_logits, active_labels)
    return logits, loss


# ==================== SFT Dataset ====================
class SFTDataset(Dataset):
    """SFT 数据集：prompt+response 拼接，只对 response 计算 loss"""

    def __init__(self, path, tokenizer_path, block_size=1024, max_samples=None):
        with open(tokenizer_path, 'rb') as f:
            tk = pickle.load(f)
        self.stoi, self.itos = tk['stoi'], tk['itos']
        self.unk_id, self.eos_id = tk['unk_id'], tk['eos_id']
        self.block_size = block_size

        self.samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples:
                    break
                try:
                    obj = json.loads(line.strip())
                    prompt = obj.get('prompt', '')
                    response = obj.get('response', '')
                    if prompt and response:
                        self.samples.append((prompt, response))
                except:
                    continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt, response = self.samples[idx]
        prompt_ids = [self.stoi.get(c, self.unk_id) for c in prompt]
        response_ids = [self.stoi.get(c, self.unk_id) for c in response]

        input_ids = prompt_ids + response_ids + [self.eos_id]
        labels = [-100] * len(prompt_ids) + response_ids + [self.eos_id]

        if len(input_ids) > self.block_size:
            input_ids = input_ids[:self.block_size]
            labels = labels[:self.block_size]
        else:
            pad_len = self.block_size - len(input_ids)
            input_ids += [self.eos_id] * pad_len
            labels += [-100] * pad_len

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


# ==================== Main ====================
if __name__ == '__main__':
    # ===== 可调参数（优化版）=====
    SFT_DATA_PATH = "data/sft_zh_demo.jsonl"
    TOKENIZER_PATH = "data/tokenizer_zh_char.pkl"
    PRETRAIN_CKPT = "checkpoints/model_epoch_3.pt"
    SFT_LR = 5e-5              # 提高学习率（原来 2e-5）
    SFT_EPOCHS = 10            # 增加训练轮数（原来 3）
    BATCH_SIZE = 4
    WARMUP_RATIO = 0.1         # 10% warmup
    # ============================

    config = GPTConfig()
    model = GPT(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数量: {total_params / 1e6:.2f}M  |  设备: {device}')

    if os.path.exists(PRETRAIN_CKPT):
        ckpt = torch.load(PRETRAIN_CKPT, map_location=device)
        if 'model_state_dict' in ckpt:
            model.load_state_dict(ckpt['model_state_dict'])
        elif 'model' in ckpt:
            model.load_state_dict(ckpt['model'])
        else:
            model.load_state_dict(ckpt)
        print(f'✅ 加载预训练模型: {PRETRAIN_CKPT}')
    else:
        print(f'⚠️ 未找到预训练模型: {PRETRAIN_CKPT}')

    dataset = SFTDataset(SFT_DATA_PATH, TOKENIZER_PATH, block_size=config.block_size)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [0.85, 0.15])
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f'SFT 样本: train={len(train_ds)}, val={len(val_ds)}, total={len(dataset)}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=SFT_LR, weight_decay=0.01)

    total_steps = SFT_EPOCHS * len(train_loader)
    warmup_steps = int(total_steps * WARMUP_RATIO)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    print(f'SFT 超参数: lr={SFT_LR}, epochs={SFT_EPOCHS}, warmup_steps={warmup_steps}')

    def train_epoch():
        model.train()
        total = 0
        for bi, (x, labels) in enumerate(train_loader):
            x, labels = x.to(device), labels.to(device)
            _, loss = sft_forward(model, x, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item()
            if bi % 5 == 0 or len(train_loader) <= 5:
                print(f'  Batch {bi + 1}/{len(train_loader)}  Loss: {loss.item():.4f}  LR: {scheduler.get_last_lr()[0]:.2e}')
        return total / len(train_loader)

    @torch.no_grad()
    def eval_epoch():
        model.eval()
        total = 0
        for x, labels in val_loader:
            x, labels = x.to(device), labels.to(device)
            _, loss = sft_forward(model, x, labels)
            total += loss.item()
        return total / len(val_loader)

    os.makedirs('checkpoints', exist_ok=True)
    print(f'🚀 开始 SFT 微调 (lr={SFT_LR}, epochs={SFT_EPOCHS})')

    for epoch in range(SFT_EPOCHS):
        t0 = time.time()
        train_loss = train_epoch()
        val_loss = eval_epoch()
        elapsed = time.time() - t0

        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss,
            'train_loss': train_loss,
            'is_sft': True,
        }
        torch.save(ckpt, f'checkpoints/sft_model_epoch_{epoch}.pt')

        print(f'Epoch {epoch + 1}/{SFT_EPOCHS} | Time: {elapsed:.0f}s | '
              f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | 💾 Saved\n')

    print('✅ SFT 微调完成！')
