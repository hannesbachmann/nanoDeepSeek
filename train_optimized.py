import os
import time
import math
import pickle
from contextlib import nullcontext
import matplotlib.pyplot as plt
import numpy as np
import torch
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

from model import NanoDeepSeek, Config
from tokenization import load_tokenizer
from visualisation import AnimatedPlot, AnimatedHeatmap


def main_train_loop(gui):
    # -----------------------------------------------------------------------------
    # default config values designed to train a gpt2 (124M) on OpenWebText
    # I/O
    out_dir = 'out'
    eval_interval = 100
    log_interval = 10
    eval_iters = 250
    eval_only = False  # if True, script exits right after the first eval
    always_save_checkpoint = True  # if True, always save a checkpoint after each eval
    # wandb logging
    wandb_log = False  # disabled by default
    wandb_project = 'owt'
    wandb_run_name = 'nano_deepseek'  # 'run' + str(time.time())
    # data
    dataset = 'csv_script'
    gradient_accumulation_steps = 1  # used to simulate larger batch sizes
    batch_size = 64  # if gradient_accumulation_steps > 1, this is the micro-batch size
    block_size = 256
    # model
    n_layer = 1
    n_head = 6
    n_routed = 16
    n_shared = 1
    n_active_experts = 4
    n_embd = 768  # 384
    expert_dim = n_embd   # // n_active_experts
    dropout = 0.0  # for pretraining 0 is good, for finetuning try 0.1+
    bias = False  # do we use bias inside LayerNorm and Linear layers?
    # adamw optimizer
    learning_rate = 1e-3  # max learning rate
    max_iters = 5000  # total number of training iterations
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.99
    grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True # whether to decay the learning rate
    warmup_iters = 100 # how many steps to warm up for
    lr_decay_iters = 5000 # should be ~= max_iters per Chinchilla
    min_lr = 1e-4 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    use_rope = True
    # DDP settings
    backend = 'nccl'  # 'nccl', 'gloo', etc.
    # system
    device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile = False  # use PyTorch 2.0 to compile the model to be faster
    # -----------------------------------------------------------------------------
    config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    config = {k: globals()[k] for k in config_keys}  # will be useful for logging
    # -----------------------------------------------------------------------------

    # various inits, derived attributes, I/O setup
    ddp = False  # run only on one GPU for now
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # ----------- batch loader -----------------
    # poor man's data loader
    data_dir = dataset

    tokenizer = load_tokenizer(data_file_name=data_dir + '\\input')

    def get_batch(split):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == 'train':
            data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    # attempt to derive vocab_size from the dataset
    # meta_path = os.path.join(data_dir, 'meta.pkl')
    # meta_vocab_size = None
    # if os.path.exists(meta_path):
    #     with open(meta_path, 'rb') as f:
    #         meta = pickle.load(f)
    meta_vocab_size = tokenizer.get_vocab_size()      # meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size}") # (inside {meta_path})")

    # ----------- model init ------------------------
    model_args = dict(n_layers=n_layer, n_heads=n_head,
                      h_dim=n_embd, max_seq_len=block_size,
                      n_tokens=None, e_dim=expert_dim, compression_dim=128,
                      n_shared=n_shared, n_routed=n_routed, k=n_active_experts,
                      use_rope=use_rope)  # start with model_args from command line
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("! Cant get vocabulary size from meta vocab file")
    model_args['n_tokens'] = meta_vocab_size
    ds_config = Config(**model_args)
    model = NanoDeepSeek(ds_config)

    # crop down the model block size if desired, using model surgery
    if block_size < model.config.max_seq_len:
        model.crop_block_size(block_size)
        model_args['max_seq_len'] = block_size  # so that the checkpoint will have the right value
    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # optimizer
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    checkpoint = None  # free up memory

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        out_loss = {}
        out_ppl = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            ppls = torch.zeros(eval_iters)
            for ki in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[ki] = loss.item()
                ppls[ki] = torch.exp(loss)
            out_loss[split] = losses.mean()
            out_ppl[split] = ppls.mean()
        model.train()
        return out_loss, out_ppl

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * (it + 1) / (warmup_iters + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    # logging
    if wandb_log and master_process:
        import wandb

        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

    # training loop
    X, Y = get_batch('train')  # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model = model
    running_mfu = -1.0
    aux_loss = []

    inverted_dict = {value: key for key, value in tokenizer.get_vocab().items()}
    a000 = [inverted_dict[i] for i in range(model_args['n_tokens'])]
    gui.set_labels(a000, range(n_routed))

    # expert tracking
    tokens_per_expert = torch.zeros(size=(n_routed, model_args['n_tokens']), dtype=torch.float32, device=device)

    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:
            losses, ppls = estimate_loss()
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} | train perplexity: {ppls['train']:.4f}, val perplexity: {ppls['val']:.4f}")
            # if iter_num % (5*eval_interval) == 0:

            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'],
                    "val/loss": losses['val'],
                    "lr": lr,
                    "mfu": running_mfu * 100,  # convert to percentage
                })
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

            tokens_per_expert = torch.zeros(size=(n_routed, model_args['n_tokens']), dtype=torch.float32, device=device)
        if iter_num == 0 and eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps  # scale the loss to account for gradient accumulation

            # live plotting
            exp_counts = model.deepseek_model.transformer_blocks[0].moe.aux_loss  # .cpu().detach().item()
            aux_loss.append(exp_counts)
            # to_plot = exp_counts.cpu().detach().numpy()
            # gui.update_plot(aux_loss)

            exp_ids = model.deepseek_model.transformer_blocks[0].moe.active_experts
            exp_inf = model.deepseek_model.transformer_blocks[0].moe.active_influence.detach()
            if exp_ids is not None:
                # get the active experts for each token
                flat_tokens = X.reshape(-1)  # (64*256,)
                flat_experts = exp_ids.reshape(-1, n_active_experts)  # (64*256, n_active_experts)
                flat_inf = exp_inf.reshape(-1, n_active_experts)    # (64*256, n_active_experts)
                expanded_tokens = flat_tokens.unsqueeze(1).expand(-1, n_active_experts).reshape(-1)  # (64*256*n_active_experts,)
                expanded_experts = flat_experts.reshape(-1)  # (64*256*n_active_experts,)
                expanded_influences = flat_inf.reshape(-1)
                indices = torch.stack([expanded_experts, expanded_tokens])  # (2, 64*256*n_active_experts)
                # tokens_per_expert.index_put_((indices[0], indices[1]), torch.ones_like(indices[0]), accumulate=True)
                tokens_per_expert.index_put_((indices[0], indices[1]), expanded_influences, accumulate=True)

                tpe_np = tokens_per_expert.cpu().detach().numpy()
                gui.update_plot(tpe_np[:, :])

            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train')
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%")
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break

    # ------ some test generation -------
    #with open(meta_path, 'rb') as f:
    #    meta = pickle.load(f)
    #encode = lambda s: [meta['stoi'][c] for c in s]
    #decode = lambda l: ''.join([meta['itos'][i] for i in l])

    start = 'Stan: Eric, what was that sound?'
    start_ids = tokenizer.encode(start).ids
    x_test = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    model.eval()
    some_result_seq = model.generate(x_test, 256)
    # some_result_seq, beams = model.generate_beam(x_test, 512, beam_width=3)
    # print('Beams:')
    # for beam in beams:
    #     print(tokenizer.decode(beam[1][0].tolist()))
    #     print('-------')
    print('best result:')
    # print(tokenizer.decode(some_result_seq))
    print(tokenizer.decode(some_result_seq[0].tolist()))


if __name__ == '__main__':
    root = tk.Tk()
    # gui = AnimatedPlot(root)
    gui = AnimatedHeatmap(root)

    threading.Thread(target=main_train_loop, args=(gui,), daemon=True).start()

    root.mainloop()
