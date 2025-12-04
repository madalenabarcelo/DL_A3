import torch
from data_trf import load_imdb
import torch.nn as nn

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


(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)
pad_idx = w2i[".pad"]  

x, y = make_batch(x_train, y_train, pad_idx, batch_size=32)
print(x.shape) 
print([i2w[w] for w in x_train[141]])


# Q2

class BaselineClassifier(nn.Module):
    """
    Simple baseline sequence-to-label model:

        input (batch, time) [long tensor with token indices]
        -> Embedding
        -> Global pooling over time (mean)
        -> Linear layer to class logits

    No softmax is applied here; use nn.CrossEntropyLoss on the logits.
    """

    def __init__(self, vocab_size, num_classes, emb_dim=300):
        super().__init__()

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

        # Global mean pooling over time dimension (dim=1)
        # emb: (batch, time, emb) -> pooled: (batch, emb)
        pooled = emb.mean(dim=1)

        # Linear layer to class logits: (batch, emb) -> (batch, num_classes)
        logits = self.fc(pooled)

        return logits

# Load data
(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)

vocab_size = len(i2w)   # number of tokens in the vocabulary
num_classes = numcls    # = 2 for IMDb

model = BaselineClassifier(vocab_size=vocab_size,
                           num_classes=num_classes,
                           emb_dim=300)

# Example batch
pad_idx = w2i[".pad"]
batch_size = 32
x_batch, y_batch = make_batch(x_train, y_train, pad_idx, batch_size=batch_size)

logits = model(x_batch)          # shape: (batch_size, num_classes)
# loss_fn = torch.nn.CrossEntropyLoss()
# loss = loss_fn(logits, y_batch)