import torch
from data_trf import load_imdb, load_imdb_synth, load_xor
import torch.nn as nn

# Q1

def make_batch(seqs, labels, pad_idx, batch_size):
    """
    Create a padded batch from a list of sequences.
    
    seqs       — list of sequences (each a list of ints)
    labels     — list of labels (ints)
    pad_idx    — index of the '.pad' token
    batch_size — number of sequences to include
    
    returns:
        x : LongTensor of shape (batch, max_len)
        y : LongTensor of shape (batch,)
    """

    # slice batch 
    batch_seqs  = seqs[:batch_size]
    batch_lbls  = labels[:batch_size]

    # determine max length
    max_len = max(len(s) for s in batch_seqs)

    # pad sequences
    padded = []
    for seq in batch_seqs:
        pad_amount = max_len - len(seq)
        padded.append(seq + [pad_idx] * pad_amount)

    # convert to tensors
    x = torch.tensor(padded, dtype=torch.long)   # (batch, time)
    y = torch.tensor(batch_lbls, dtype=torch.long)

    return x, y

# Pad all sequences - used for Q3
def collate_fn(batch):
    seqs, labels = zip(*batch)               # list of sequences, list of labels
    return make_batch(seqs, labels, pad_idx, batch_size=len(seqs))

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


# Run Q3 experiment

if __name__ == "__main__":
    datasets = {
        "IMDb": load_imdb(final=False),
        "IMDb-synth": load_imdb_synth(),
        "XOR": load_xor()
    }

    # Hyperparameters for the experiment
    batch_size = 64
    learning_rate = 1e-3
    max_epochs = 5   
    
    print("\n================ RUNNING EXPERIMENTS (Q3) ================\n")

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
            collate_fn=collate_fn
        )

        val_loader = DataLoader(
            list(zip(x_val, y_val)),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )

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


