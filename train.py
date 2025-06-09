from model import NanoDeepSeek
from data_loading import get_batch
import torch
import torch.nn.functional as F
import os
import pickle


def train():
    h_dim = 256     # model hidden dimension
    e_dim = 1024    # expert hidden dimension (4*h_dim similar to GPT-2 transformer dim)
    compression_dim = 64    # dimension of the key-value latent compression in MLA
    n_layers = 4    # number of transformer layers (each contain MLA and MoE)
    n_heads = 2     # number of attention heads
    n_shared = 1    # number of shared experts
    n_routed = 10   # number of routed experts
    k = 3   # number of activated routed experts
    epochs = 2
    batch_size = 64
    max_seq_len = 128
    grad_clip = 1.0     # maximum norm of the gradients, clip at this value
    iter_per_epoch = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_dir = 'shakespeare_char'
    train_dir = data_dir + '\\train.bin'
    valid_dir = data_dir + '\\val.bin'

    # get number of tokens/vocabulary size from the dataset meta
    meta_path = os.path.join(data_dir, 'meta.pkl')
    n_tokens = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        n_tokens = meta['vocab_size']
        print(f"found vocab_size = {n_tokens} (inside {meta_path})")

    model = NanoDeepSeek(h_dim, e_dim, compression_dim, n_layers, n_heads, n_tokens, max_seq_len, n_shared, n_routed, k)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=3e-4, total_steps=40, pct_start=0.1)

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = 0.0
        for i in range(iter_per_epoch):
            X, y = get_batch(train_dir, batch_size, max_seq_len, device)
            # Note: autocast with device_type='cpu' does not seem to work
            # Workaround: set device_type='cuda' (even if cuda is not available),
            # then cuda will be disabled on runtime but autocast still works
            with torch.amp.autocast(device_type='cuda'):
                # automatically use float16 and float32 instead of only float32 to improve performance
                logits = model(X)
                loss += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

    # ---- some test generation ----
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    start = 'First Citizen:'
    start_ids = encode(start)
    x_test = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    some_result_seq = model.generate(x_test, max_seq_len)
    print(decode(some_result_seq[0].tolist()))
    pass