import torch
from data_trf import load_imdb, load_imdb_synth, load_xor, load_toy, load_wp
import torch.nn as nn


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
    # ==========================================================
    RUN_QUESTION = 5   # change this number to choose question to run

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


    else:
        print("Unknown RUN_QUESTION value")
