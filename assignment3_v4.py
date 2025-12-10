import torch
from data_trf import load_imdb, load_imdb_synth, load_xor, load_toy, load_wp
import torch.nn as nn
import math
import torch.nn.functional as F
import time
import torch.distributions as dist
import matplotlib.pyplot as plt


# Q1

def make_batch(seqs, labels, pad_idx, batch_size, max_len=356):
    """
    Create a padded batch from a list of sequences.
    
    seqs       — list of sequences (each a list of ints)
    labels     — list of labels (ints)
    pad_idx    — index of the '.pad' token
    batch_size — number of sequences to include
    max_len    — maximum sequence length (default: 256)
    
    returns:
        x : LongTensor of shape (batch, max_len)
        y : LongTensor of shape (batch,)
    """

    # slice batch 
    batch_seqs  = seqs[:batch_size]
    batch_lbls  = labels[:batch_size]

    # Truncate sequences to max_len
    truncated_seqs = [seq[:max_len] for seq in batch_seqs]

    # Determine the actual max length after truncation
    actual_max_len = max(len(seq) for seq in truncated_seqs)

    # Pad sequences
    padded = []
    for seq in truncated_seqs:
        pad_amount = actual_max_len - len(seq)
        padded.append(seq + [pad_idx] * pad_amount)

    x = torch.tensor(padded, dtype=torch.long)  # (batch, time)
    y = torch.tensor(batch_lbls, dtype=torch.long)
    return x, y

# Pad all sequences - used for Q3
def collate_fn(batch, max_len =256):
    seqs, labels = zip(*batch)               # list of sequences, list of labels
    return make_batch(seqs, labels, pad_idx, batch_size=len(seqs), max_len=max_len)

'''
(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)
pad_idx = w2i[".pad"]  

x, y = make_batch(x_train, y_train, pad_idx, batch_size=32)
print(x.shape) 
print([i2w[w] for w in x_train[141]])
'''


# Q2

class BaselineClassifier(nn.Module):
    """
    Simple baseline sequence-to-label model:

        input (batch, time) [long tensor with token indices]
        -> Embedding
        -> Global pooling over time (pool_type: one of {"mean", "max", "select"})
        -> Linear layer to class logits

    No softmax is applied here; use nn.CrossEntropyLoss on the logits.
    """

    def __init__(self, vocab_size, num_classes, emb_dim=300, pool_type="mean"):
        super().__init__()

        self.pool_type = pool_type

        # Embedding layer: maps token indices to embedding vectors
        #    vocab_size = number of tokens in the vocabulary (len(i2w))
        #    emb_dim    = 300 as required in the assignment
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        # Linear classifier: maps pooled embedding to class scores
        self.fc = nn.Linear(emb_dim, num_classes)


    def forward(self, x):
        """
        x: LongTensor of shape (batch, time)
           containing token indices.

        returns:
            logits: FloatTensor of shape (batch, num_classes)
        """

        # x -> (batch, time, emb)
        emb = self.embedding(x)

        # Global Pooling
        # emb: (batch, time, emb) -> pooled: (batch, emb)
        if self.pool_type == "mean":
            pooled = emb.mean(dim=1)          # (batch, emb)

        elif self.pool_type == "max":
            pooled, _ = emb.max(dim=1)        # (batch, emb)

        elif self.pool_type == "select":
            pooled = emb[:, 0, :]             # (batch, emb)

        else:
            raise ValueError("Invalid pool type.")

        # Linear layer to class logits: (batch, emb) -> (batch, num_classes)
        logits = self.fc(pooled)

        return logits


# Q3

import torch.nn.functional as F
from torch.utils.data import DataLoader


# Accuracy calculation

def accuracy(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


# Training loop (minimal, 1 epoch)

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0

    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = F.cross_entropy(logits, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

# Validation

def evaluate(model, loader):
    model.eval()
    acc = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x)
            acc += accuracy(logits, batch_y)

    return acc / len(loader)


# Running Q3 experiments for the three pooling modes

"""
def run_q3_training(x_train, y_train, x_val, y_val, vocab_size, num_classes,
                    batch_size=64, num_epochs=1):
    
    train_loader = DataLoader(
        list(zip(x_train, y_train)),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        list(zip(x_val, y_val)),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )


    results = {}

    for pool in ["mean", "max", "select"]:
        print(f"\n====== Training with pooling: {pool} ======")

        model = BaselineClassifier(vocab_size, num_classes, 300, pool)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        for epoch in range(num_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer)
            val_acc = evaluate(model, val_loader)
            print(f"Epoch {epoch+1}: loss={train_loss:.4f}, val_acc={val_acc:.4f}")

        results[pool] = val_acc

    return results
"""

#Q4

class SimpleSelfAttention(nn.Module):
    def forward(self, x):
        # x: (batch, time, emb)
        w_prime = torch.bmm(x, x.transpose(1, 2))  # (batch, time, time)
        w = F.softmax(w_prime, dim=-1)  # (batch, time, time)
        y = torch.bmm(w, x)  # (batch, time, emb)
        return y

class SelfAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, emb_dim=300):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.self_attention = SimpleSelfAttention()
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        emb = self.embedding(x)  # (batch, time, emb)
        attn_out = self.self_attention(emb)  # (batch, time, emb)
        # Select pooling: use only the first token's embedding
        pooled = attn_out[:, 0, :]  # (batch, emb)
        logits = self.fc(pooled)  # (batch, num_classes)
        return logits

#Q6

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, emb_dim, num_heads=6):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        # Linear layers to create keys, queries, and values
        self.tokeys = nn.Linear(emb_dim, emb_dim)
        self.toqueries = nn.Linear(emb_dim, emb_dim)
        self.tovalues = nn.Linear(emb_dim, emb_dim)

        # Final linear layer
        self.unifyheads = nn.Linear(emb_dim, emb_dim)

    def forward(self, x):
        # x: (batch, time, emb)
        batch_size, seq_len, emb_dim = x.size()

        # Compute keys, queries, and values
        keys = self.tokeys(x)    # (batch, time, emb)
        queries = self.toqueries(x)  # (batch, time, emb)
        values = self.tovalues(x)  # (batch, time, emb)

        # Reshape keys, queries, and values for multi-head attention
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, seq_len, self.num_heads, self.head_dim)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Reorder dimensions to (batch, num_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)  # (batch, num_heads, time, head_dim)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention
        # Scaling factor
        scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(x.device)

        # Compute attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / scale  # (batch, num_heads, time, time)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (batch, num_heads, time, time)

        # Multiply attention weights with values
        output = torch.matmul(attn_weights, values)  # (batch, num_heads, time, head_dim)

        # Reorder dimensions back to (batch, time, num_heads, head_dim)
        output = output.transpose(1, 2).contiguous()

        # Reshape to (batch, time, emb_dim)
        output = output.view(batch_size, seq_len, self.emb_dim)

        # Apply final linear layer
        output = self.unifyheads(output)  # (batch, time, emb_dim)

        return output

class MultiHeadSelfAttentionClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, emb_dim=300, num_heads=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.multihead_attention = MultiHeadSelfAttention(emb_dim, num_heads)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        # x: (batch, time)
        emb = self.embedding(x)  # (batch, time, emb)
        attn_out = self.multihead_attention(emb)  # (batch, time, emb)
        pooled, _ = attn_out.max(dim=1)  # (batch, emb)
        logits = self.fc(pooled)  # (batch, num_classes)
        return logits
    
#Q7

class AttentionClassifier(nn.Module):
    """
    Generic classifier that takes an attention module and a pool_type.
    Used to compare simple vs full self-attention under the same conditions.
    """
    def __init__(self, vocab_size, num_classes, emb_dim=300,
                 attention_module=None, pool_type="select"):
        super().__init__()
        self.pool_type = pool_type
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        if attention_module is None:
            # default: simple self-attention
            self.attention = SimpleSelfAttention()
        else:
            self.attention = attention_module

        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        # x: (batch, time)
        emb = self.embedding(x)          # (batch, time, emb)
        attn_out = self.attention(emb)   # (batch, time, emb)

        # GLOBAL POOL
        if self.pool_type == "mean":
            pooled = attn_out.mean(dim=1)
        elif self.pool_type == "max":
            pooled, _ = attn_out.max(dim=1)
        elif self.pool_type == "select":
            pooled = attn_out[:, 0, :]
        else:
            raise ValueError("Unknown pool_type")

        logits = self.fc(pooled)         # (batch, num_classes)
        return logits
    
def make_simple_sa_classifier(vocab_size, num_classes, emb_dim=300):
    # simple self-attention + select pool
    return AttentionClassifier(
        vocab_size, num_classes, emb_dim,
        attention_module=SimpleSelfAttention(),
        pool_type="select"
    )

def make_full_sa_classifier(vocab_size, num_classes,
                            emb_dim=300, num_heads=6):
    # full multi-head self-attention + select pool
    full_attn = MultiHeadSelfAttention(emb_dim, num_heads)
    return AttentionClassifier(
        vocab_size, num_classes, emb_dim,
        attention_module=full_attn,
        pool_type="select"
    )

# Q8 

class PositionalSelfAttentionClassifier(nn.Module):
    """
    Simple self-attention classifier WITH learned positional embeddings.

    Architecture:
      - Token embedding: nn.Embedding(vocab_size, emb_dim)
      - Positional embedding: nn.Embedding(max_len, emb_dim)
      - Simple self-attention (same as Q4)
      - SELECT pooling (first time step)
      - Linear classifier
    """
    def __init__(self, vocab_size, num_classes, emb_dim=300, max_len=512):
        super().__init__()

        self.emb_dim = emb_dim
        self.max_len = max_len

        # Token embeddings (as before)
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        # NEW: positional embeddings, one vector per possible position
        # positions: 0, 1, ..., max_len-1
        self.pos_embedding = nn.Embedding(max_len, emb_dim)

        # Same simple self-attention layer as Q4
        self.self_attention = SimpleSelfAttention()

        # Final classifier
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        """
        x : LongTensor of shape (batch, time)
            Token indices (already padded/truncated).
        """
        batch_size, seq_len = x.size()

        # Token embeddings: (B, T) -> (B, T, E)
        tok_emb = self.embedding(x)

        # Build position indices 0..T-1 for each batch element:
        # positions: (1, T) -> expand -> (B, T)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)

        # Positional embeddings: (B, T) -> (B, T, E)
        pos_emb = self.pos_embedding(positions)

        # Add token and position embeddings
        emb = tok_emb + pos_emb  # (B, T, E)

        # Simple self-attention over the enriched embeddings
        attn_out = self.self_attention(emb)  # (B, T, E)

        # Select pooling: use only the first token's representation
        pooled = attn_out[:, 0, :]  # (B, E)

        # Linear classifier
        logits = self.fc(pooled) # (B, num_classes)

        return logits

# Q9

class TransformerBlock(nn.Module):
    """
    Single Transformer block built around the existing MultiHeadSelfAttention.

    Structure:
      x                       # (batch, time, emb)
        -> Multi-head self-attention
        -> Dropout
        -> Residual + LayerNorm
        -> Feedforward (Linear -> ReLU -> Linear)
        -> Dropout
        -> Residual + LayerNorm
        -> output x           # (batch, time, emb)
    """
    def __init__(self, emb_dim, num_heads=6, ff_mult=4, dropout=0.1):
        super().__init__()

        self.attn = MultiHeadSelfAttention(emb_dim, num_heads)

        # LayerNorm after attention and after feedforward
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)

        # Feedforward network: emb_dim -> 4*emb_dim -> emb_dim
        ff_hidden = ff_mult * emb_dim
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, emb_dim),
        )

        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_ff = nn.Dropout(dropout)

    def forward(self, x):
        """
        x : (batch, time, emb)
        """
        # --- Self-attention sub-layer ---
        attn_out = self.attn(x)                # (B, T, E)
        x = x + self.dropout_attn(attn_out)    # residual connection
        x = self.ln1(x)                        # layer norm

        # --- Feedforward sub-layer ---
        ff_out = self.ff(x)                    # (B, T, E)
        x = x + self.dropout_ff(ff_out)        # residual connection
        x = self.ln2(x)                        # layer norm

        return x

class TransformerClassifier(nn.Module):
    """
    Transformer-based sequence classifier:

      input (batch, time) -> Embedding (+ optional positional embeddings)
          -> 3 TransformerBlocks
          -> select pool (take first token)
          -> Linear classifier -> logits
    """
    def __init__(self,
                 vocab_size,
                 num_classes,
                 emb_dim=300,
                 num_heads=6,
                 num_layers=3,
                 max_len=512,
                 dropout=0.1,
                 use_positional=True):
        super().__init__()

        self.emb_dim = emb_dim
        self.use_positional = use_positional
        self.max_len = max_len

        # Token embeddings
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        # Optional positional embeddings (recommended for IMDb)
        if use_positional:
            self.pos_embedding = nn.Embedding(max_len, emb_dim)
        else:
            self.pos_embedding = None

        # Stack of Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads=num_heads,
                             ff_mult=4, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

        # Final classifier
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        """
        x : (batch, time)
        """
        batch_size, seq_len = x.size()

        # Embeddings 
        tok_emb = self.embedding(x)            # (B, T, E)

        if self.pos_embedding is not None:
            # positions: 0..seq_len-1
            positions = torch.arange(
                seq_len, device=x.device
            ).unsqueeze(0).expand(batch_size, seq_len)
            pos_emb = self.pos_embedding(positions)  # (B, T, E)
            h = tok_emb + pos_emb
        else:
            h = tok_emb

        h = self.dropout(h)

        #  Transformer blocks 
        for block in self.blocks:
            h = block(h)                       # (B, T, E)

        # Global select pooling (first token)
        pooled = h[:, 0, :]                    # (B, E)

        # Classification 
        logits = self.fc(pooled)               # (B, num_classes)
        return logits


#Q10: a function that takes a dataset in the form of an integer tensor, slices out b instances of length l, and returns a batch of size (b, L)
def make_random_batch(data, b, L):
    """
    Slice `b` random instances of length `L` from the dataset.

    Args:
        data: LongTensor of shape (N,), where N is the total length of the dataset.
        b: Number of instances to sample (batch size).
        L: Length of each instance.

    Returns:
        batch: LongTensor of shape (b, L)
    """
    N = data.size(0)
    # Generate random starting indices
    starts = torch.randint(low=0, high=N - L, size=(b,))
    # For each start index, slice out a sequence of length L
    batch = torch.stack([data[start:start+L] for start in starts])
    return batch


#Q10: a function that takes a dataset in the form of an integer tensor, slices out b instances of length l, and returns a batch of size (b, L)
def make_random_batch(data, b, L):
    """
    Slice `b` random instances of length `L` from the dataset.

    Args:
        data: LongTensor of shape (N,), where N is the total length of the dataset.
        b: Number of instances to sample (batch size).
        L: Length of each instance.

    Returns:
        batch: LongTensor of shape (b, L)
    """
    N = data.size(0)
    # Generate random starting indices
    starts = torch.randint(low=0, high=N - L, size=(b,))
    # For each start index, slice out a sequence of length L
    batch = torch.stack([data[start:start+L] for start in starts])
    return batch

# Q11
'''
class CausalSelfAttention(nn.Module):
    def forward(self, x):
        # x: (batch, time, emb)
        batch_size, seq_len, emb_dim = x.size()
        # Compute attention scores
        scores = torch.bmm(x, x.transpose(1, 2))  # (batch, time, time)
        # Causal mask: set attention scores for future tokens to -inf
        mask = torch.triu(torch.ones(seq_len, seq_len) * float('-inf'), diagonal=1).to(x.device)
        scores = scores + mask.unsqueeze(0)  # (batch, time, time)
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)  # (batch, time, time)
        # Apply attention weights to input
        output = torch.bmm(attn_weights, x)  # (batch, time, emb)
        return output

class AutoregressiveModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=300):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.self_attention = CausalSelfAttention()
        self.fc = nn.Linear(emb_dim, vocab_size)  # Project to vocab size for logits

    def forward(self, x):
        # x: (batch, time)
        emb = self.embedding(x)  # (batch, time, emb)
        attn_out = self.self_attention(emb)  # (batch, time, emb)
        logits = self.fc(attn_out)  # (batch, time, vocab_size)
        return logits
'''

# Q11 updated

class CausalSelfAttention(nn.Module):
    def __init__(self, emb_dim, num_heads=6, dropout=0.1):
        super().__init__()
        assert emb_dim % num_heads == 0, "emb_dim must divide num_heads"

        self.num_heads = num_heads
        self.head_dim = emb_dim // num_heads

        # Linear layers for Q, K, V
        self.Wq = nn.Linear(emb_dim, emb_dim)
        self.Wk = nn.Linear(emb_dim, emb_dim)
        self.Wv = nn.Linear(emb_dim, emb_dim)

        # Output projection
        self.Wo = nn.Linear(emb_dim, emb_dim)

        # Dropout on attention weights
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, E = x.shape

        # Project to Q, K, V
        q = self.Wq(x)  # (B, T, E)
        k = self.Wk(x)
        v = self.Wv(x)

        # Reshape into (B, heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Causal mask
        mask = torch.triu(
            torch.ones(T, T, device=x.device) * float('-inf'),
            diagonal=1
        )
        scores = scores + mask  # broadcast to (B, heads, T, T)

        # Attention weights
        att = F.softmax(scores, dim=-1)
        att = self.dropout(att)

        # Weighted sum
        out = torch.matmul(att, v)  # (B, heads, T, head_dim)

        # Combine heads back
        out = out.transpose(1, 2).contiguous().view(B, T, E)

        return self.Wo(out)


class TransformerBlock(nn.Module):
    def __init__(self, emb_dim, num_heads=6, dropout=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(emb_dim)
        self.att = CausalSelfAttention(emb_dim, num_heads, dropout)

        self.ln2 = nn.LayerNorm(emb_dim)
        self.ff = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.ReLU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Attention + residual
        x = x + self.att(self.ln1(x))

        # Feedforward + residual
        x = x + self.ff(self.ln2(x))

        return x

class AutoregressiveModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=300,
                 num_heads=6, num_layers=6, dropout=0.1):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, emb_dim)

        # 6 transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(emb_dim)
        self.fc = nn.Linear(emb_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)  # (B, T, emb)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.fc(x)
        return logits


# Q12

NAT_TO_BIT = 1.4426950408889634   # = log2(e)

def evaluate_autoregressive(model, val_data, batch_size, L, vocab_size, iters=50):
    """
    Compute average log-probability (cross-entropy) on validation set.
    Returns the loss in NATS — caller converts to bits.
    """
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for _ in range(iters):
            batch = make_random_batch(val_data, batch_size, L+1)
            x = batch[:, :-1]       # inputs (t0 ... t_{L-1})
            y = batch[:, 1:]        # targets (t1 ... t_L)

            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                y.reshape(-1)
            )
            total_loss += loss.item()

    return total_loss / iters

# Q13

# ---------- Sampling helpers for Q13 ----------

def sample(lnprobs, temperature=1.0):
    """
    Sample an index from unnormalized logits.
    lnprobs: 1D tensor of size (vocab_size,)
    """
    if temperature == 0.0:
        return lnprobs.argmax()

    p = F.softmax(lnprobs / temperature, dim=0)
    cd = dist.Categorical(p)
    return cd.sample()


def sample_from_model(model, data, i2c, L,
                      seed_length=16, generate=200, temperature=1.0):
    """
    Autoregressive sampling:
    - pick a random seed of length `seed_length` from `data`
    - repeatedly sample next token from the model
    - keep at most the last L tokens as context

    Returns a decoded string.
    """
    model.eval()
    with torch.no_grad():
        data_tensor = torch.tensor(data, dtype=torch.long)
        N = data_tensor.size(0)

        # 1) choose random seed in validation data
        start = torch.randint(low=0, high=N - seed_length, size=(1,)).item()
        seed = data_tensor[start:start + seed_length].unsqueeze(0)  # (1, S)
        seq = seed.clone()  # (1, current_len)

        # 2) autoregressive loop
        for _ in range(generate):
            # context: last L tokens (or less at the beginning)
            if seq.size(1) > L:
                ctx = seq[:, -L:]
            else:
                ctx = seq

            logits = model(ctx)          # (1, ctx_len, vocab_size)
            last_logits = logits[0, -1]  # (vocab_size,)

            idx = sample(last_logits, temperature=temperature).item()
            idx_tensor = torch.tensor([[idx]], dtype=torch.long)
            seq = torch.cat([seq, idx_tensor], dim=1)

        # 3) decode to characters
        chars = [i2c[int(i)] for i in seq[0]]
        return "".join(chars)


# Run Q3 experiment

"""if __name__ == "__main__":
    datasets = {
        "IMDb": load_imdb(final=False),
        "IMDb-synth": load_imdb_synth(),
        "XOR": load_xor()
    }

    # Hyperparameters for the experiment (for Q3 qnd Q4)
    batch_size = 64
    learning_rate = 1e-3
    max_epochs = 10 
    
    print("\n================ RUNNING EXPERIMENTS (Q4) ================\n")

    final_results = {}

    for name, data in datasets.items():
        print(f"\n===== Dataset: {name} =====")

        # Load data
        (x_train, y_train), (x_val, y_val), (i2w, w2i), num_classes = data
        pad_idx = w2i[".pad"]
        vocab_size = len(i2w)

        # Build DataLoaders
        train_loader = DataLoader(
            list(zip(x_train, y_train)),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, max_len=256)
        )

        val_loader = DataLoader(
            list(zip(x_val, y_val)),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

        # Use SelfAttentionClassifier instead of BaselineClassifier
        model = SelfAttentionClassifier(vocab_size, num_classes, emb_dim=300)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        print("Training examples (IMDb-synth):")
        for i in range(3):
            print([i2w[w] for w in x_train[i]], y_train[i])

        print("\nValidation examples (IMDb-synth):")
        for i in range(3):
            print([i2w[w] for w in x_val[i]], y_val[i])


        best_val = 0.0
        for epoch in range(max_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer)
            val_acc = evaluate(model, val_loader)
            best_val = max(best_val, val_acc)
            print(f"Epoch {epoch+1}/{max_epochs} | loss={train_loss:.4f} | val_acc={val_acc:.4f}")

        final_results[name] = best_val
        

        #question 3
        # Train model for each pooling type
        results = {}

        for pool in ["mean", "max", "select"]:
            print(f"\n---- Training pooling='{pool}' ----")

            model = BaselineClassifier(vocab_size, num_classes, 300, pool)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            best_val = 0.0

            for epoch in range(max_epochs):
                train_loss = train_one_epoch(model, train_loader, optimizer)
                val_acc = evaluate(model, val_loader)
                best_val = max(best_val, val_acc)

                print(f"Epoch {epoch+1}/{max_epochs} | loss={train_loss:.4f} | val_acc={val_acc:.4f}")

                # IMDb must reach >= 0.80 accuracy in ≤ 5 epochs
                if name == "IMDb" and val_acc >= 0.80:
                    print(f"✔ IMDb reached 0.8 accuracy with pooling='{pool}'!")
                    print(f"Hyperparameters: batch_size={batch_size}, lr={learning_rate}, epochs={epoch+1}")
                    break

            results[pool] = best_val

        final_results[name] = results

    print("\n================ FINAL RESULTS ================")
    for ds, res in final_results.items():
        print(f"\n{ds}: {res}")

    """


############################################### 
#Q5 tuning
"""
if __name__ == "__main__":
    datasets = {
        "IMDb-synth": load_imdb_synth(),
        "XOR": load_xor()
    }

    # Hyperparameters to tune
    learning_rates = [1e-2, 1e-3, 1e-4]
    batch_sizes = [16, 32, 64]
    max_epochs = 100  # As suggested in the assignment

    print("\n================ RUNNING EXPERIMENTS (Q5) ================\n")
    final_results = {}

    for name, data in datasets.items():
        print(f"\n===== Dataset: {name} =====")
        (x_train, y_train), (x_val, y_val), (i2w, w2i), num_classes = data
        pad_idx = w2i[".pad"]
        vocab_size = len(i2w)

        best_val_acc = 0.0
        best_hyperparams = {}

        # Hyperparameter tuning loop
        for lr in learning_rates:
            for batch_size in batch_sizes:
                print(f"\n--- Learning rate: {lr}, Batch size: {batch_size} ---")

                # Create DataLoaders
                train_loader = DataLoader(
                    list(zip(x_train, y_train)),
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=lambda batch: collate_fn(batch, max_len=256)
                )
                val_loader = DataLoader(
                    list(zip(x_val, y_val)),
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=lambda batch: collate_fn(batch, max_len=256)
                )

                # Initialize model with select pooling
                model = SelfAttentionClassifier(vocab_size, num_classes, emb_dim=300)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                # Training loop
                for epoch in range(max_epochs):
                    model.train()
                    for batch_x, batch_y in train_loader:
                        optimizer.zero_grad()
                        logits = model(batch_x)
                        loss = F.cross_entropy(logits, batch_y)
                        loss.backward()
                        optimizer.step()

                    # Evaluate
                    val_acc = evaluate(model, val_loader)
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_hyperparams = {"learning_rate": lr, "batch_size": batch_size}
                    print(f"Epoch {epoch+1}/{max_epochs} | val_acc={val_acc:.4f}")

        final_results[name] = {"best_val_acc": best_val_acc, "best_hyperparams": best_hyperparams}

    print("\n================ FINAL RESULTS (Q5) ================")
    for ds, res in final_results.items():
        print(f"\n{ds}:")
        print(f"  Best validation accuracy: {res['best_val_acc']:.4f}")
        print(f"  Best hyperparameters: {res['best_hyperparams']}")

"""
###############################################


# run Q7 full self-attention experiments
"""
if __name__ == "__main__":
    datasets = {
        "IMDb": load_imdb(final=False),
        "IMDb-synth": load_imdb_synth(),
        "XOR": load_xor(),
    }

    # We use best Q5 settings
    lr = 1e-2
    batch_size = 64
    max_epochs = 50
    max_seq_len = 256

    print("\n================ RUNNING EXPERIMENTS (Q7) ================\n")

    # Store the best validation accuracy for each dataset/model
    results_q7 = {}

     # Loop over IMDb, IMDb-synth, XOR
    for name, data in datasets.items():
        print(f"\n===== Dataset: {name} =====")
        (x_train, y_train), (x_val, y_val), (i2w, w2i), num_classes = data
        vocab_size = len(i2w)
        pad_idx = w2i[".pad"]

        train_loader = DataLoader(
            list(zip(x_train, y_train)),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, max_len=max_seq_len),
        )
        val_loader = DataLoader(
            list(zip(x_val, y_val)),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, max_len=max_seq_len),
        )
        # ---- ONLY FULL SELF-ATTENTION ----
        print(f"\n--- Model: FULL self-attention (select pool) ---")
        model = make_full_sa_classifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            emb_dim=300,
            num_heads=6,   # same as in Q6
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_acc = 0.0
        for epoch in range(max_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer)
            val_acc = evaluate(model, val_loader)
            best_val_acc = max(best_val_acc, val_acc)
            print(
                f"Epoch {epoch+1}/{max_epochs} | "
                f"loss={train_loss:.4f} | val_acc={val_acc:.4f}"
            )
        # store only full-SA result, keyed by dataset name
        results_q7[name] = best_val_acc

    print("\n================ FULL SELF-ATTENTION RESULTS (Q7) ================")
    for ds_name, acc in results_q7.items():
        print(f"{ds_name}: best val_acc (full SA, select pool) = {acc:.4f}")
"""
############################################### 

# Running questions

if __name__ == "__main__":
    # ==========================================================
    # Choose which question's experiment to run:
    #   3 -> Q3 baseline pooling
    #   4 -> Q4 simple self-attention + max pool
    #   5 -> Q5 tuning simple self-attention + select pool
    #   7 -> Q7 full multi-head self-attention + select pool
    #   8 -> Q8 XOR with/without positional embeddings
    #   9  -> Q9 Transformer IMDb
    #   11 -> Q11 autoregressive sanity check
    #   12 -> Q12 bits-per-char eval
    #   13 -> Q13 long training + sampling + curves
    # ==========================================================
    RUN_QUESTION = 13   # change this number to choose question to run

    # ------------ Q3: Baseline classifier with 3 pooling types ------------
    if RUN_QUESTION == 3:
        datasets = {
            "IMDb": load_imdb(final=False),
            "IMDb-synth": load_imdb_synth(),
            "XOR": load_xor()
        }

        batch_size = 64
        learning_rate = 1e-3
        max_epochs = 5

        print("\n================ RUNNING EXPERIMENTS (Q3) ================\n")

        final_results = {}

        for name, data in datasets.items():
            print(f"\n===== Dataset: {name} =====")
            (x_train, y_train), (x_val, y_val), (i2w, w2i), num_classes = data
            pad_idx = w2i[".pad"]
            vocab_size = len(i2w)

            train_loader = DataLoader(
                list(zip(x_train, y_train)),
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn
            )

            val_loader = DataLoader(
                list(zip(x_val, y_val)),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn
            )

            results = {}
            for pool in ["mean", "max", "select"]:
                print(f"\n---- Training pooling='{pool}' ----")
                model = BaselineClassifier(vocab_size, num_classes, 300, pool)
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

                best_val = 0.0
                for epoch in range(max_epochs):
                    train_loss = train_one_epoch(model, train_loader, optimizer)
                    val_acc = evaluate(model, val_loader)
                    best_val = max(best_val, val_acc)
                    print(f"Epoch {epoch+1}/{max_epochs} | loss={train_loss:.4f} | val_acc={val_acc:.4f}")

                results[pool] = best_val

            final_results[name] = results

        print("\n================ FINAL RESULTS (Q3) ================")
        for ds, res in final_results.items():
            print(f"\n{ds}: {res}")

    # ------------ Q4: Simple self-attention + MAX pool ------------
    elif RUN_QUESTION == 4:
        datasets = {
            "IMDb": load_imdb(final=False),
            "IMDb-synth": load_imdb_synth(),
            "XOR": load_xor()
        }

        batch_size = 64
        learning_rate = 1e-3
        max_epochs = 10
        max_seq_len = 256

        print("\n================ RUNNING EXPERIMENTS (Q4) ================\n")

        final_results = {}

        for name, data in datasets.items():
            print(f"\n===== Dataset: {name} =====")
            (x_train, y_train), (x_val, y_val), (i2w, w2i), num_classes = data
            pad_idx = w2i[".pad"]
            vocab_size = len(i2w)

            train_loader = DataLoader(
                list(zip(x_train, y_train)),
                batch_size=batch_size,
                shuffle=True,
                collate_fn=lambda batch: collate_fn(batch, max_len=max_seq_len)
            )

            val_loader = DataLoader(
                list(zip(x_val, y_val)),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda batch: collate_fn(batch, max_len=max_seq_len)
            )

            model = SelfAttentionMaxPoolClassifier(vocab_size, num_classes, emb_dim=300)
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

            best_val = 0.0
            for epoch in range(max_epochs):
                train_loss = train_one_epoch(model, train_loader, optimizer)
                val_acc = evaluate(model, val_loader)
                best_val = max(best_val, val_acc)
                print(f"Epoch {epoch+1}/{max_epochs} | loss={train_loss:.4f} | val_acc={val_acc:.4f}")

            final_results[name] = best_val

        print("\n================ FINAL RESULTS (Q4) ================")
        for ds, acc in final_results.items():
            print(f"{ds}: best val_acc = {acc:.4f}")

    # ------------ Q5: Tuning simple self-attention + SELECT pool ------------
    elif RUN_QUESTION == 5:
        datasets = {
            "IMDb": load_imdb(final=False),
            "IMDb-synth": load_imdb_synth(),
            "XOR": load_xor()
        }

        learning_rates = [1e-2, 1e-3, 1e-4]
        batch_sizes = [16, 32, 64]
        max_epochs = 100

        print("\n================ RUNNING EXPERIMENTS (Q5) ================\n")
        final_results = {}

        for name, data in datasets.items():
            print(f"\n===== Dataset: {name} =====")
            (x_train, y_train), (x_val, y_val), (i2w, w2i), num_classes = data
            pad_idx = w2i[".pad"]
            vocab_size = len(i2w)

            best_val_acc = 0.0
            best_hyperparams = {}

            for lr in learning_rates:
                for batch_size in batch_sizes:
                    print(f"\n--- Learning rate: {lr}, Batch size: {batch_size} ---")

                    train_loader = DataLoader(
                        list(zip(x_train, y_train)),
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=lambda batch: collate_fn(batch, max_len=256)
                    )
                    val_loader = DataLoader(
                        list(zip(x_val, y_val)),
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=lambda batch: collate_fn(batch, max_len=256)
                    )

                    model = SelfAttentionClassifier(vocab_size, num_classes, emb_dim=300)  # select pool
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                    for epoch in range(max_epochs):
                        model.train()
                        for batch_x, batch_y in train_loader:
                            optimizer.zero_grad()
                            logits = model(batch_x)
                            loss = F.cross_entropy(logits, batch_y)
                            loss.backward()
                            optimizer.step()

                        val_acc = evaluate(model, val_loader)
                        if val_acc > best_val_acc:
                            best_val_acc = val_acc
                            best_hyperparams = {"learning_rate": lr, "batch_size": batch_size}
                        print(f"Epoch {epoch+1}/{max_epochs} | val_acc={val_acc:.4f}")

            final_results[name] = {
                "best_val_acc": best_val_acc,
                "best_hyperparams": best_hyperparams
            }

        print("\n================ FINAL RESULTS (Q5) ================")
        for ds, res in final_results.items():
            print(f"\n{ds}:")
            print(f"  Best validation accuracy: {res['best_val_acc']:.4f}")
            print(f"  Best hyperparameters: {res['best_hyperparams']}")

    # ------------ Q7: Full multi-head SA + SELECT pool ------------
    elif RUN_QUESTION == 7:
        datasets = {
            "IMDb": load_imdb(final=False),
            "IMDb-synth": load_imdb_synth(),
            "XOR": load_xor(),
        }

        lr = 1e-2
        batch_size = 64
        max_epochs = 50
        max_seq_len = 256

        print("\n================ RUNNING EXPERIMENTS (Q7) ================\n")

        results_q7 = {}

        for name, data in datasets.items():
            print(f"\n===== Dataset: {name} =====")
            (x_train, y_train), (x_val, y_val), (i2w, w2i), num_classes = data
            vocab_size = len(i2w)
            pad_idx = w2i[".pad"]

            train_loader = DataLoader(
                list(zip(x_train, y_train)),
                batch_size=batch_size,
                shuffle=True,
                collate_fn=lambda batch: collate_fn(batch, max_len=max_seq_len),
            )
            val_loader = DataLoader(
                list(zip(x_val, y_val)),
                batch_size=batch_size,
                shuffle=False,
                collate_fn=lambda batch: collate_fn(batch, max_len=max_seq_len),
            )

            print(f"\n--- Model: FULL self-attention (select pool) ---")
            model = make_full_sa_classifier(
                vocab_size=vocab_size,
                num_classes=num_classes,
                emb_dim=300,
                num_heads=6,
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

            best_val_acc = 0.0
            for epoch in range(max_epochs):
                train_loss = train_one_epoch(model, train_loader, optimizer)
                val_acc = evaluate(model, val_loader)
                best_val_acc = max(best_val_acc, val_acc)
                print(
                    f"Epoch {epoch+1}/{max_epochs} | "
                    f"loss={train_loss:.4f} | val_acc={val_acc:.4f}"
                )
            results_q7[name] = best_val_acc

        print("\n================ FULL SELF-ATTENTION RESULTS (Q7) ================")
        for ds_name, acc in results_q7.items():
            print(f"{ds_name}: best val_acc (full SA, select pool) = {acc:.4f}")

    # ------------ Q8: XOR with vs without positional embeddings ------------
    elif RUN_QUESTION == 8:
        print("\n================ RUNNING EXPERIMENTS (Q8: XOR + POS EMB) ================\n")

        (x_train, y_train), (x_val, y_val), (i2w, w2i), num_classes = load_xor()
        vocab_size = len(i2w)
        pad_idx = w2i[".pad"]

        batch_size = 64
        max_epochs = 100
        max_seq_len = 2  # XOR sequences have length 2
        learning_rate = 1e-2  # from Q5

        train_loader = DataLoader(
            list(zip(x_train, y_train)),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, max_len=max_seq_len),
        )
        val_loader = DataLoader(
            list(zip(x_val, y_val)),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, max_len=max_seq_len),
        )

        results_q8 = {}

        # 1) Simple SA WITHOUT position embeddings
        print("\n=== Simple self-attention WITHOUT position embeddings ===")
        model_no_pos = SelfAttentionClassifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            emb_dim=300,
        )
        optimizer_no_pos = torch.optim.Adam(model_no_pos.parameters(), lr=learning_rate)

        best_val_no_pos = 0.0
        for epoch in range(max_epochs):
            train_loss = train_one_epoch(model_no_pos, train_loader, optimizer_no_pos)
            val_acc = evaluate(model_no_pos, val_loader)
            best_val_no_pos = max(best_val_no_pos, val_acc)
            print(f"[NO POS] Epoch {epoch+1}/{max_epochs} | loss={train_loss:.4f} | val_acc={val_acc:.4f}")

        results_q8["no_position_embeddings"] = best_val_no_pos

        # 2) Simple SA WITH position embeddings
        print("\n=== Simple self-attention WITH position embeddings ===")
        model_with_pos = PositionalSelfAttentionClassifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            emb_dim=300,
            max_len=max_seq_len,   # only 2 positions needed
        )
        optimizer_with_pos = torch.optim.Adam(model_with_pos.parameters(), lr=learning_rate)

        best_val_with_pos = 0.0
        for epoch in range(max_epochs):
            train_loss = train_one_epoch(model_with_pos, train_loader, optimizer_with_pos)
            val_acc = evaluate(model_with_pos, val_loader)
            best_val_with_pos = max(best_val_with_pos, val_acc)
            print(f"[WITH POS] Epoch {epoch+1}/{max_epochs} | loss={train_loss:.4f} | val_acc={val_acc:.4f}")

        results_q8["with_position_embeddings"] = best_val_with_pos

        print("\n================ Q8 RESULTS (XOR) ================")
        for setting, acc in results_q8.items():
            print(f"{setting}: best val_acc = {acc:.4f}")
    
    # ------------ Q9: 3-block Transformer on IMDb ------------
    elif RUN_QUESTION == 9:
        print("\n================ RUNNING EXPERIMENTS (Q9: TRANSFORMER ON IMDb) ================\n")

        (x_train, y_train), (x_val, y_val), (i2w, w2i), num_classes = load_imdb(final=False)
        print("Checkpoint 1")
        vocab_size = len(i2w)
        pad_idx = w2i[".pad"]

        batch_size = 64
        max_epochs = 10
        max_seq_len = 256    # truncate long IMDb reviews
        learning_rate = 1e-4 # usually needs smaller LR than baseline

        train_loader = DataLoader(
            list(zip(x_train, y_train)),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, max_len=max_seq_len),
        )
        val_loader = DataLoader(
            list(zip(x_val, y_val)),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, max_len=max_seq_len),
        )

        model = TransformerClassifier(
            vocab_size=vocab_size,
            num_classes=num_classes,
            emb_dim=300,
            num_heads=6,
            num_layers=3,
            max_len=max_seq_len,
            dropout=0.1,
            use_positional=True,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        print("Checkpoint 2")
        best_val = 0.0
        for epoch in range(max_epochs):
            train_loss = train_one_epoch(model, train_loader, optimizer)
            val_acc = evaluate(model, val_loader)
            best_val = max(best_val, val_acc)
            print(f"Epoch {epoch+1}/{max_epochs} | loss={train_loss:.4f} | val_acc={val_acc:.4f}")

        print(f"\nBest validation accuracy (Q9 Transformer): {best_val:.4f}")
    
    # ------------ Q11: Autoregressive model sanity check ------------
    elif RUN_QUESTION == 11:
        print("\n================ RUNNING EXPERIMENTS (Q11) ================\n")

        # Load toy dataset
        (train, val), (i2c, c2i) = load_toy(final=False)
        vocab_size = len(i2c)
        train_data = torch.tensor(train, dtype=torch.long).contiguous()
        val_data = torch.tensor(val, dtype=torch.long).contiguous()

        # Hyperparameters
        b = 32  # Batch size
        L = 64  # Sequence length
        emb_dim = 300  # Embedding dimension
        num_epochs = 10  # Number of training epochs
        lr = 1e-3  # Learning rate

        # Initialize model
        model = AutoregressiveModel(vocab_size, emb_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            for _ in range(100):  # Number of batches per epoch
                batch = make_random_batch(train_data, b, L+1)
                x = batch[:, :-1]  # Input: first L tokens
                y = batch[:, 1:]   # Target: last L tokens
                optimizer.zero_grad()
                logits = model(x)  # (batch, L, vocab_size)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / 100
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f}")

        # Test the model on a sample input
        test_input = torch.randint(0, vocab_size, (1, L))
        logits = model(test_input)
        print("\nTest output shape:", logits.shape)

    # ------------ Q12: Bits-per-character evaluation ------------
    elif RUN_QUESTION == 12:
        print("\n================ RUNNING EXPERIMENTS (Q12) ================\n")

        
        start_time = time.time()

        # Load dataset (toy language modeling task)
        (train, val), (i2c, c2i) = load_toy(final=False)
        vocab_size = len(i2c)

        train_data = torch.tensor(train, dtype=torch.long).contiguous()
        val_data   = torch.tensor(val, dtype=torch.long).contiguous()

        # Hyperparameters
        batch_size = 32
        L = 256
        emb_dim = 300
        num_epochs = 5
        lr = 1e-3

        # Model + optimizer
        model = AutoregressiveModel(vocab_size, emb_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # Initial validation log-probability
        init_loss_nats = evaluate_autoregressive(model, val_data, batch_size, L, vocab_size)
        init_loss_bits = init_loss_nats * NAT_TO_BIT
        print(f"Initial validation loss: {init_loss_bits:.4f} bits")

        # ------------------ Training Loop (with time limit) ------------------
        for epoch in range(num_epochs):

            model.train()
            total_loss = 0.0
            epoch_start = time.time()

            for _ in range(1000):

                batch = make_random_batch(train_data, batch_size, L+1)
                x = batch[:, :-1]
                y = batch[:, 1:]

                optimizer.zero_grad()
                logits = model(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, vocab_size),
                    y.reshape(-1)
                )
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            # Convert training loss to bits
            train_bits = (total_loss / 1000) * NAT_TO_BIT

            # Validation (fast, only ~50 batches)
            val_loss_nats = evaluate_autoregressive(
                model, val_data, batch_size, L, vocab_size, iters=50
            )
            val_bits = val_loss_nats * NAT_TO_BIT

            print(f"Epoch {epoch+1}/{num_epochs} | "
                f"train_bits={train_bits:.4f} | "
                f"val_bits={val_bits:.4f} | "
                f"epoch_time={time.time() - epoch_start:.1f}s")

        # --------------------------------------------------------------------
        total_minutes = (time.time() - start_time) / 60
        print(f"\n✓ Finished Q12 in {total_minutes:.2f} minutes.")
    
    # ------------ Q13: Long training, sampling & curves on toy LM ------------
    elif RUN_QUESTION == 13:
        print("\n================ RUNNING EXPERIMENTS (Q13) ================\n")

        start_time = time.time()

        # 1) Load toy dataset
        (train, val), (i2c, c2i) = load_toy(final=False)
        vocab_size = len(i2c)

        train_data = torch.tensor(train, dtype=torch.long).contiguous()
        val_data   = torch.tensor(val,   dtype=torch.long).contiguous()

        # 2) Hyperparameters (as suggested)
        batch_size    = 64
        L             = 256      # context length
        emb_dim       = 300
        num_heads     = 6
        num_layers    = 6
        dropout       = 0.1
        base_lr       = 1e-3
        warmup_steps  = 5_000
        max_steps     = 50_000   # 50k batches
        eval_interval = 10_000   # compute val loss & sample every 10k steps
        val_batches   = 1_000    # can increase to 10_000 if you want

        # 3) Model + optimizer
        model = AutoregressiveModel(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)

        # 4) Logging structures
        train_loss_bits_history = []
        train_steps             = []
        val_bits_history        = []
        val_steps               = []
        grad_norm_history       = []
        grad_steps              = []

        # 5) Initial validation
        init_loss_nats = evaluate_autoregressive(
            model,
            val_data,
            batch_size,
            L,
            vocab_size,
            iters=val_batches
        )
        init_loss_bits = init_loss_nats * NAT_TO_BIT
        print(f"Initial validation loss: {init_loss_bits:.4f} bits")

        # 6) Training loop for 50k batches
        for step in range(1, max_steps + 1):
            model.train()

            # Sample batch of length L+1
            batch = make_random_batch(train_data, batch_size, L + 1)
            x = batch[:, :-1]  # inputs
            y = batch[:, 1:]   # targets

            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                y.reshape(-1)
            )
            loss.backward()

            # Gradient norm (before clipping)
            total_norm_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm_sq += param_norm.item() ** 2
            total_norm = math.sqrt(total_norm_sq)
            grad_norm_history.append(total_norm)
            grad_steps.append(step)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Learning-rate warmup
            if step <= warmup_steps:
                lr_scale = step / warmup_steps
                for g in optimizer.param_groups:
                    g["lr"] = base_lr * lr_scale

            optimizer.step()

            # Store training loss (in bits)
            loss_bits = loss.item() * NAT_TO_BIT
            train_loss_bits_history.append(loss_bits)
            train_steps.append(step)

            # Progress print
            if step % 1_000 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                print(f"[Train] step {step}/{max_steps} "
                      f"| loss_bits={loss_bits:.4f} "
                      f"| grad_norm={total_norm:.3f} "
                      f"| lr={current_lr:.6f}")

            # ---- Evaluation + sampling every eval_interval ----
            if step % eval_interval == 0:
                print(f"\n==== EVAL at step {step} ====")

                val_loss_nats = evaluate_autoregressive(
                    model,
                    val_data,
                    batch_size,
                    L,
                    vocab_size,
                    iters=val_batches
                )
                val_bits = val_loss_nats * NAT_TO_BIT
                val_bits_history.append(val_bits)
                val_steps.append(step)
                print(f"[Validation] loss_bits={val_bits:.4f}")

                # Sample from model
                print("\n--- Sample (temperature=1.0) ---")
                sample_txt = sample_from_model(
                    model,
                    val,
                    i2c,
                    L=L,
                    seed_length=16,
                    generate=200,
                    temperature=1.0
                )
                print(sample_txt)
                print("\n------------------------------\n")

        # 7) Plot curves
        print("Training finished, plotting curves...")

        # Loss curve (training)
        plt.figure()
        plt.plot(train_steps, train_loss_bits_history)
        plt.xlabel("Step")
        plt.ylabel("Train loss (bits/token)")
        plt.title("Q13: Training loss over steps")
        plt.tight_layout()
        plt.savefig("q13_train_loss.png")

        # Validation curve
        if len(val_steps) > 0:
            plt.figure()
            plt.plot(val_steps, val_bits_history, marker="o")
            plt.xlabel("Step")
            plt.ylabel("Validation loss (bits/token)")
            plt.title("Q13: Validation loss over steps")
            plt.tight_layout()
            plt.savefig("q13_val_loss.png")

        # Gradient-norm curve
        plt.figure()
        plt.plot(grad_steps, grad_norm_history)
        plt.xlabel("Step")
        plt.ylabel("Gradient L2 norm")
        plt.title("Q13: Gradient norm over steps")
        plt.tight_layout()
        plt.savefig("q13_grad_norm.png")

        total_minutes = (time.time() - start_time) / 60.0
        print(f"\n✓ Q13 finished. Loss & grad norm plots saved as:")
        print("  - q13_train_loss.png")
        print("  - q13_val_loss.png")
        print("  - q13_grad_norm.png")
        print(f"Total wall time: {total_minutes:.2f} minutes")

    else:
        print("Unknown RUN_QUESTION value")
