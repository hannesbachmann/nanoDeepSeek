import numpy as np
import torch


def get_batch(file_name, batch_size, seq_len, device):
    data = np.memmap(file_name, mode="r")

    ix = torch.randint(len(data) - seq_len, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + seq_len]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + seq_len]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        # move x and y probably to cpu
        x, y = x.to(device), y.to(device)
    return x, y