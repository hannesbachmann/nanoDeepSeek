import math
import heapq
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch
from dataclasses import dataclass

@dataclass
class Config:
    h_dim: int = 64
    e_dim: int = 64
    compression_dim:int = 32
    n_layers:int = 2
    n_heads:int = 2
    n_tokens:int = 65
    max_seq_len:int = 32
    n_shared:int = 1
    n_routed:int = 5
    k:int = 2


class NanoDeepSeek(nn.Module):
    def __init__(self, model_config):
        super(NanoDeepSeek, self).__init__()
        self.max_seq_len = model_config.max_seq_len
        self.config = model_config
        self.deepseek_model = nn.ModuleDict(dict(
            token_emb=nn.Embedding(model_config.n_tokens, model_config.h_dim),
            transformer_blocks=nn.ModuleList(
                [TransformerBlock(model_config.h_dim, model_config.e_dim, model_config.n_heads,
                                  model_config.compression_dim, model_config.n_shared,
                                  model_config.n_routed, model_config.k) for _ in
                 range(model_config.n_layers)]),
            norm=nn.LayerNorm(model_config.h_dim),
            proj_head=nn.Linear(model_config.h_dim, model_config.n_tokens, bias=False)
        ))

    def forward(self, x):
        # x: (B, S) -> (B, S, D)
        x = self.deepseek_model.token_emb(x)

        for block in self.deepseek_model.transformer_blocks:
            x, c_kv = block(x)
        x = self.deepseek_model.norm(x)

        return self.deepseek_model.proj_head(x)

    @torch.no_grad()
    def generate(self, x, gen_seq_len):
        top_k = 5
        for t in range(gen_seq_len):
            # crop input sequence to match max seq_len
            if x.size(1) > self.max_seq_len:
                x_input = x[:, -self.max_seq_len:]
            else:
                x_input = x
            logits = self.forward(x_input)
            last_logits = logits[:, -1, :]
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
                last_logits[last_logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(last_logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            x = torch.cat((x, idx_next), dim=1)
        return x

    @torch.no_grad()
    def generate_beam(self, x, max_new_tokens, beam_width=3):
        seed_ids = x
        # Each beam is a tuple: (log_prob, token_ids)
        beams = [(0.0, seed_ids)]
        while any(beam[1].size(1) < max_new_tokens for beam in beams):
            new_beams = []
            for log_prob, seq in beams:
                if seq.size(1) >= max_new_tokens:
                    # stop for this beam
                    new_beams.append((log_prob, seq))
                    continue

                seq_input = seq
                if seq_input.size(1) > self.max_seq_len:
                    seq_input = seq_input[:, -self.max_seq_len:]

                logits = self.forward(seq_input)
                next_token_logits = logits[:, -1, :]  # Last token's logits (1, vocab_size)

                probs = torch.softmax(next_token_logits, dim=-1)

                top_probs, top_indices = torch.topk(probs, beam_width, dim=-1)

                top_probs = top_probs.squeeze(0)
                top_indices = top_indices.squeeze(0)

                for prob, token_id in zip(top_probs, top_indices):
                    new_seq = torch.cat([seq, token_id.view(1, -1)], dim=-1)
                    new_log_prob = log_prob + math.log(prob.item())
                    new_beams.append((new_log_prob, new_seq))

            # Keep top `beam_width` sequences
            beams = heapq.nlargest(beam_width, new_beams, key=lambda x: x[0])

        # Choose the best final sequence
        best_seq = max(beams, key=lambda x: x[0])[1][0]
        return best_seq.tolist(), beams


class ExpertBlock(nn.Module):
    def __init__(self, h_dim, e_dim):
        super(ExpertBlock, self).__init__()
        # very small expert (as in deepseek MoE)
        self.h_dim = h_dim
        self.e_dim = e_dim
        self.up = nn.Linear(self.h_dim, self.e_dim, bias=False)
        self.down = nn.Linear(self.e_dim, self.h_dim, bias=False)

    def forward(self, x):
        up_c = self.up(x)
        out_c = self.down(F.gelu(up_c))
        return out_c


class TransformerBlock(nn.Module):
    def __init__(self, h_dim, e_dim, n_heads, compression_dim, n_shared=1, n_routed=10, k=3):
        super(TransformerBlock, self).__init__()
        # MLA and deepseek MoE
        self.h_dim = h_dim
        self.e_dim = e_dim
        self.norm1 = nn.LayerNorm(h_dim)
        self.attn = MLA(h_dim, n_heads, compression_dim)
        self.norm2 = nn.LayerNorm(h_dim)
        self.moe = MoE(h_dim, e_dim, n_shared, n_routed, k)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, prev_kv=None):
        # can use previous key, value during inference
        # MLA with layer norm and residual connection
        attn, c_kv = checkpoint(self.attn, self.norm1(x), prev_kv, use_reentrant=True)
        attn = attn + x
        # MoE with layer norm and residual connection
        moe = checkpoint(self.moe, self.norm2(attn), use_reentrant=True)
        moe = moe + x

        out = self.dropout(moe)

        return out, c_kv


class MoE(nn.Module):
    def __init__(self, h_dim, e_dim, n_shared, n_routed, k):
        """finegrained expert segmentation and shared expert isolation (as in deepseek Moe)

        :param h_dim: model hidden dimension
        :param e_dim: expert hidden dimension
        :param n_shared: number of shared experts
        :param n_routed: number of routed experts
        :param k: number of activated routed experts    (top-k activation)
        """
        super(MoE, self).__init__()
        self.h_dim = h_dim
        # self.e_dim = e_dim
        self.e_dim = h_dim * 4  # GPT2 transformer MLP inspired up-down-projection
        self.n_shared = n_shared
        self.n_routed = n_routed
        self.k = k
        # use only a few (1-2) shared experts and a lot of routed experts
        self.shared_experts = nn.ModuleList([ExpertBlock(self.h_dim, self.e_dim) for _ in range(n_shared)])
        self.routed_experts = nn.ModuleList([ExpertBlock(self.h_dim, self.e_dim) for _ in range(n_routed)])

        # routing network
        self.router = nn.Linear(self.h_dim, n_routed, bias=False)

    def forward(self, x):
        # compute output for shared experts (always active)
        shared_out = sum(expert(x) for expert in self.shared_experts)
        # routing to get the top-k active routed experts with their contribution
        router_out = self.router(x)
        all_prob = F.softmax(router_out, dim=-1)
        top_k_prob, top_k_idx = torch.topk(all_prob, k=self.k)

        routed_out = torch.zeros_like(x)
        for k in range(self.k):
            expert_mask = top_k_idx[..., k]
            expert_contrib = torch.zeros_like(x)

            for expert_idx in range(self.n_routed):
                mask = (expert_mask == expert_idx)
                if mask.any():
                    expert_out = self.routed_experts[expert_idx](x[mask])
                    expert_contrib[mask] = expert_out * top_k_prob[..., k][mask].unsqueeze(-1)

            routed_out += expert_contrib
        return shared_out + routed_out + x


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_rope, device='cpu'):
        """creating relative positional embedding using RoPE

        :param d_rope: dimension of the rotary embedding (maybe chose d_head // 2 as in deepseek-v2)
        """
        super().__init__()
        assert d_rope % 2 == 0, "Dimension must be even for rotary embeddings"
        self.d_rope = d_rope
        half_dim = self.d_rope // 2
        # calculate the angles
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, 2, device=device).float() / half_dim))
        self.register_buffer("inv_freq", inv_freq)
        self.scale = 40

    def forward(self, k, q):
        # q: (B, S, d_rope*n_heads)
        batch_size, seq_len, n_heads, d_rope = q.shape
        # get positions or times
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq) / self.scale
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, d_rope // 2)
        # double the length (repeat freq matrix)
        rotation = torch.cat((freqs, freqs), dim=-1)  # (seq_len, d_rope)
        cos = torch.cos(rotation).view(1, seq_len, 1, -1)  # [1, seq, 1, dim]
        sin = torch.sin(rotation).view(1, seq_len, 1, -1)
        # apply rotations on keys and queries
        k_rotary = apply_rotary(k, cos, sin)
        q_rotary = apply_rotary(q, cos, sin)
        return k_rotary, q_rotary


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(x, cos, sin):
    """
    Apply rotary embeddings to the first half of x.
    """
    # Split x into two parts: one for rotary embeddings and the other untouched
    x_rot, x_base = x.split(cos.shape[-1], dim=-1)
    # Apply rotary embeddings to the rotary part
    x_rot = (x_rot * cos) + (rotate_half(x_rot) * sin)
    # Concatenate the rotary-applied and base parts
    return torch.cat([x_rot, x_base], dim=-1)


class MLA(nn.Module):
    def __init__(self, h_dim, n_heads, compression_dim):
        super(MLA, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # model size related parameters
        self.n_heads = n_heads
        self.compression_dim = compression_dim
        self.d_head = h_dim // self.n_heads
        self.d_rope = self.d_head // 2  # as chosen in deepseek-v2
        self.up_proj_dim = (self.d_head - self.d_rope) * self.n_heads  # for keys and queries

        # define all down- and up-projections
        self.W_dkv = nn.Linear(h_dim, self.compression_dim, bias=False)
        self.W_dq = nn.Linear(h_dim, self.compression_dim, bias=False)
        self.W_uk = nn.Linear(self.compression_dim, self.up_proj_dim, bias=False)
        self.W_uv = nn.Linear(self.compression_dim, self.n_heads * self.d_head, bias=False)
        self.W_uq = nn.Linear(self.compression_dim, self.up_proj_dim, bias=False)

        # projections to produce decoupled keys and queries
        self.W_kr = nn.Linear(h_dim, self.n_heads * self.d_rope, bias=False)
        self.W_qr = nn.Linear(self.compression_dim, self.n_heads * self.d_rope, bias=False)

        # RoPE
        self.RoPE = RotaryPositionalEmbedding(self.d_rope, self.device)

        # final projection to match attention output dimension
        self.out_proj = nn.Linear(self.n_heads * self.d_head, h_dim, bias=False)

    def forward(self, x, prev_kv=None):
        # create all down-projection latents
        # x: (B, S, D)
        batch_size, seq_len, h_dim = x.shape

        # compute down-projections
        c_kv = self.W_dkv(x)
        c_q = self.W_dq(x)

        # compute up_projections
        q_c = self.W_uq(c_q).view(batch_size, seq_len, self.n_heads, self.d_head - self.d_rope)
        k_c = self.W_uk(c_kv).view(batch_size, seq_len, self.n_heads, self.d_head - self.d_rope)
        v = self.W_uv(c_kv).view(batch_size, seq_len, self.n_heads, self.d_head)

        # compute decoupled keys and queries
        k_r = self.W_kr(x).view(batch_size, seq_len, self.n_heads, self.d_rope)
        q_r = self.W_qr(c_q).view(batch_size, seq_len, self.n_heads, self.d_rope)

        # compute rotary positional embeddings (RoPE)
        k_rope, q_rope = self.RoPE(k_r, q_r)

        # combine to get keys and queries
        k = torch.cat([k_c, k_rope], dim=-1)
        q = torch.cat([q_c, q_rope], dim=-1)

        # use fast flash attention
        out1 = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                                is_causal=True)

        # # calculate attention
        # attn_scores = torch.einsum("bqhd, bkhd->bhqk", k, q) / math.sqrt(self.d_head)
        # attn_weights = F.softmax(attn_scores, dim=-1)
        # out = torch.einsum("bhqk,bkhd->bqhd", attn_scores, v)

        # reassemble the heads and project to the dimension of input x
        output = self.out_proj(out1.contiguous().view(batch_size, seq_len, -1))

        return output, (c_kv, k_rope)
