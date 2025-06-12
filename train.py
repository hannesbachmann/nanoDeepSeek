from model import NanoDeepSeek, Config
from data_loading import get_batch
import torch
import torch.nn.functional as F
import os
import pickle


def train():
    h_dim = 64     # model hidden dimension
    e_dim = 64    # expert hidden dimension (4*h_dim similar to GPT-2 transformer dim)
    compression_dim = 32    # dimension of the key-value latent compression in MLA
    n_layers = 2    # number of transformer layers (each contain MLA and MoE)
    n_heads = 2     # number of attention heads
    n_shared = 1    # number of shared experts
    n_routed = 1    # 5   # number of routed experts
    k = 2   # number of activated routed experts
    epochs = 5000
    batch_size = 32
    max_seq_len = 32
    grad_clip = 1.0     # maximum norm of the gradients, clip at this value
    iter_per_epoch = 200
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

    model_args = dict(n_layers=n_layers, n_heads=n_heads,
                      h_dim=h_dim, max_seq_len=max_seq_len,
                      n_tokens=None, e_dim=e_dim, compression_dim=128,
                      n_shared=1, n_routed=4, k=1)  # start with model_args from command line
    model_args['n_tokens'] = n_tokens
    ds_config = Config(**model_args)
    model = NanoDeepSeek(ds_config)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=5e-3, total_steps=epochs, pct_start=0.1)

    for epoch in range(epochs):
        model.train()

        loss_acc = 0.0
        for i in range(iter_per_epoch):
            optimizer.zero_grad()
            X, y = get_batch(train_dir, batch_size, max_seq_len, device)
            # Note: autocast with device_type='cpu' does not seem to work
            # Workaround: set device_type='cuda' (even if cuda is not available),
            # then cuda will be disabled on runtime but autocast still works
            with torch.amp.autocast(device_type='cuda'):
                # automatically use float16 and float32 instead of only float32 to improve performance
                logits, _ = model(X)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
                loss_acc += (loss / iter_per_epoch)
            # print(f'Batch: {i+1}/{iter_per_epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            #scheduler.step()
        if epoch % 10 == 0:
            acc_val_loss = 0.0
            for i in range(iter_per_epoch):
                X_val, y_val = get_batch(valid_dir, batch_size, max_seq_len, device)
                with torch.amp.autocast(device_type='cuda'):
                    # automatically use float16 and float32 instead of only float32 to improve performance
                    logits = model(X_val)
                    val_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y_val.view(-1), ignore_index=-1)
                    acc_val_loss += (val_loss / iter_per_epoch)
            print(f'Epoch: {epoch + 1}/{epochs} \t Train Loss: {loss_acc.item()} \t Val Loss: {acc_val_loss.item()}')
        else:
            print(f'Epoch: {epoch+1}/{epochs} \t Loss: {loss_acc.item()}')

    # ---- some test generation ----
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])

    start = 'First Citizen:'
    start_ids = encode(start)
    x_test = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    model.eval()
    # some_result_seq = model.generate(x_test, max_seq_len)
    some_result_seq, beams = model.generate_beam(x_test, 128, beam_width=5)
    print('Beams:')
    for beam in beams:
        print(decode(beam[1][0].tolist()))
        print('-------')
    print('best result:')
    print(decode(some_result_seq))
    # print(decode(some_result_seq[0].tolist()))
    pass