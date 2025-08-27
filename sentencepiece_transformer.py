import torch
import torch.nn as nn
from torch.nn import functional as F
import sentencepiece as spm
from datasets import load_dataset
import os
import math
import matplotlib.pyplot as plt

# hyperparameters

batch_size = 32
block_size = 384
max_iters = 12000
eval_interval = 500
eval_iters = 100
learning_rate = 6e-4
n_embed = 512
n_head = 8
n_layer = 8
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#################################################################

torch.manual_seed(1337)

dataset = load_dataset("wikitext", "wikitext-103-v1", split="train")
input_file = "wiki_train.txt"

if not os.path.exists(input_file):
    with open(input_file, "w", encoding="utf-8") as f:
        f.write("\n".join(dataset["text"]))


model_prefix = "wiki_bpe"
vocab_size = 5000

if not os.path.exists(f"{model_prefix}.model"):
    spm.SentencePieceTrainer.Train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type='bpe',
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3
    )
    print("Tokenizer trained!")

sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
encode = lambda s: sp.encode(s, out_type=int)
decode = lambda ids: sp.decode(ids)
vocab_size = sp.get_piece_size()
print(f"Vocabulary size: {vocab_size}")

# data encoding
encoded_tensor_file = "wiki_train_tensor.pt"
progress_interval = 10000

if not os.path.exists(encoded_tensor_file):
    print("Encoding dataset line by line with progress...")
    data_tokens = []
    for i, line in enumerate(dataset["text"]):
        data_tokens.extend(encode(line))
        if i % progress_interval == 0:
            print(f"Encoded {i} / {len(dataset['text'])} lines")
    data = torch.tensor(data_tokens, dtype=torch.long)
    torch.save(data, encoded_tensor_file)
    print("Encoding complete and saved!")
else:
    print("Loading pre-encoded dataset...")
    data = torch.load(encoded_tensor_file)
    print("Dataset loaded.")

#################################################################
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data_split = train_data if split == "train" else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# model
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2,-1) * C ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4*n_embed),
            nn.ReLU(),
            nn.Linear(4*n_embed, n_embed),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class TransformerLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)
    
    def forward(self, idx, targets=None):
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens, top_p=0.9):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            mask = cumulative_probs > top_p
            mask[:, 0] = 0
            sorted_probs[mask] = 0.0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

            next_token = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(-1, next_token)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

model = TransformerLanguageModel().to(device)

#################################################################
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1
)

warmup_steps = int(0.05 * max_iters)
def lr_lambda(step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, max_iters - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

train_losses = []
val_losses = []

for iter in range(max_iters):
    xb, yb = get_batch("train")
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    
    if iter % eval_interval == 0:
        losses = estimate_loss()
        train_losses.append(losses['train'])
        val_losses.append(losses['val'])
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

#################################################################
torch.save(model.state_dict(), "transformer_model.pt")
print("Model saved as transformer_model.pt")

plt.figure(figsize=(8,6))
plt.plot(range(0, max_iters, eval_interval), train_losses, label="Train Loss")
plt.plot(range(0, max_iters, eval_interval), val_losses, label="Validation Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curve")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png")
plt.show()
print("Loss curve saved as loss_curve.png")

# generation
prompt = "The Roman Empire was known for"
input_ids = torch.tensor([encode(prompt)], dtype=torch.long).to(device)

model.eval()
with torch.no_grad():
    generated_ids = model.generate(input_ids, max_new_tokens=200, top_p=0.9)

generated_text = decode(generated_ids[0].tolist())
print("\nGenerated text:\n")
print(generated_text)
