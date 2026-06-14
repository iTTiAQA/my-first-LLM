"""
独立训练脚本 — 从 build_gpt.ipynb 提取
用法: python train.py
服务器路径: /root/llm_project/train.py
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
        self.register_buffer('attention_mask', torch.tril(torch.ones(config.block_size, config.block_size)))
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
            nn.Linear(config.n_embd, 4*config.n_embd), nn.GELU(),
            nn.Linear(4*config.n_embd, config.n_embd), nn.Dropout(config.dropout))
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
        return logits, F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))


# ==================== Dataset ====================
class MyDataset(Dataset):
    def __init__(self, path, block_size=1024, max_lines=None):
        # 注意：服务器上 tokenizer 路径固定为 /root/autodl-tmp/llm_project/data/
        with open('/root/autodl-tmp/llm_project/data/tokenizer_zh_char.pkl', 'rb') as f:
            tk = pickle.load(f)
        self.stoi, self.itos = tk['stoi'], tk['itos']
        self.vocab_size = tk['vocab_size']
        self.unk_id, self.eos_id = tk['unk_id'], tk['eos_id']

        raw = []
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines: break
                try:
                    t = json.loads(line.strip())['text']
                    if len(t) >= 30: raw.append(t)
                except: continue

        encoded = []
        for text in raw:
            encoded += [self.stoi.get(c, self.unk_id) for c in text] + [self.eos_id]

        self.data = []
        for i in range(0, len(encoded)-block_size, block_size):
            chunk = encoded[i:i+block_size+1]
            if len(chunk) < block_size+1:
                chunk += [self.eos_id]*(block_size+1-len(chunk))
            self.data.append(chunk)

    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        c = self.data[i]
        return torch.tensor(c[:-1], dtype=torch.long), torch.tensor(c[1:], dtype=torch.long)


# ==================== Training ====================
if __name__ == '__main__':
    # ----- 可调参数 -----
    MAX_TRAIN_LINES = None    # None = 使用全部 1.15GB 数据（约120万条）
    NUM_EPOCHS = 10           # 训练轮数（数据量大，跑多轮效果好）
    RESUME = True             # True=断点续训, False=从头开始
    # --------------------

    config = GPTConfig()
    model = GPT(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())

    print(f'模型参数量: {total_params/1e6:.2f}M  |  设备: {device}')
    print(f'GPU 名称: {torch.cuda.get_device_name(0)}  |  显存: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
    print(f'词表大小: {config.vocab_size}  |  上下文长度: {config.block_size}')

    # 加载数据
    dataset = MyDataset('/root/autodl-tmp/llm_project/data/wiki_zh_full.jsonl',
                        block_size=config.block_size, max_lines=MAX_TRAIN_LINES)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=0)
    print(f'训练样本: {len(train_ds)}  |  验证样本: {len(val_ds)}  |  每epoch {len(train_loader)} batches')

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

    # ===== 断点续训 =====
    import glob as _glob
    start_epoch = 0
    if RESUME:
        ckpt_files = sorted(_glob.glob('checkpoints/model_epoch_*.pt'))
        if ckpt_files:
            latest_ckpt = ckpt_files[-1]
            print(f'🔍 发现已有 checkpoint: {latest_ckpt}')
            ckpt = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt['epoch'] + 1
            print(f'✅ 从 epoch {start_epoch} 继续训练')
        else:
            print('ℹ️  未找到 checkpoint，从头开始训练')
    else:
        print('ℹ️  RESUME=False，从头开始训练')
        # 清空旧 checkpoint（可选）
        # import shutil
        # if os.path.exists('checkpoints'):
            # shutil.rmtree('checkpoints')
    # =====================

    os.makedirs('checkpoints', exist_ok=True)

    def train_epoch():
        model.train()
        total = 0
        for bi, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            _, loss = model(x, targets=y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total += loss.item()
            if bi % 50 == 0:
                print(f'  Batch {bi}/{len(train_loader)}  Loss: {loss.item():.4f}')
        return total / len(train_loader)

    @torch.no_grad()
    def eval_epoch():
        model.eval()
        total = 0
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, targets=y)
            total += loss.item()
        return total / len(val_loader)

    print('=' * 60)
    print(f'🚀 开始训练 (epoch {start_epoch} → {NUM_EPOCHS-1}, 共 {NUM_EPOCHS-start_epoch} 轮)')
    print('=' * 60)

    train_losses, val_losses = [], []

    for epoch in range(start_epoch, NUM_EPOCHS):
        t0 = time.time()
        train_loss = train_epoch()
        val_loss = eval_epoch()
        elapsed = time.time() - t0

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        ckpt = {
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss,
        }
        torch.save(ckpt, f'checkpoints/model_epoch_{epoch}.pt')

        print(f'\n🔹 Epoch {epoch+1}/{NUM_EPOCHS}  |  '
              f'用时: {elapsed:.0f}s  |  '
              f'Train Loss: {train_loss:.4f}  |  '
              f'Val Loss: {val_loss:.4f}  |  💾 已保存\n')

    print('=' * 60)
    print('✅ 训练完成！')
    print(f'Train Loss: {[f"{l:.4f}" for l in train_losses]}')
    print(f'Val Loss:   {[f"{l:.4f}" for l in val_losses]}')

    # 显示保存的模型
    print(f'\n📂 checkpoints 目录:')
    for f in sorted(os.listdir('checkpoints')):
        size = os.path.getsize(f'checkpoints/{f}') / (1024**2)
        print(f'  {f} ({size:.1f} MB)')
