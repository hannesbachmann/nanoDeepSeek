import math
from torch import nn
import torch.nn.functional as F
import torch


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
    def __init__(self, h_dim, e_dim):
        super(TransformerBlock, self).__init__()
        # MLA and deepseek MoE
        self.h_dim = h_dim
        self.e_dim = e_dim


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
        self.e_dim = e_dim
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
        shared = sum(expert(x) for expert in self.shared_experts)
        # routing to get the top-k active routed experts
        router_out = self.router(x)
        all_prob = F.softmax(router_out, dim=-1)
        top_k_prob, top_k_idx = torch.topk(all_prob, k=self.k)



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
        batch_size, seq_len, _ = k.shape
        # get positions or times
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq) / self.scale
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)   # (seq_len, d_rope // 2)
        # double the length (repeat freq matrix)
        rotation = torch.cat((freqs, freqs), dim=-1)    # (seq_len, d_rope)
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
        self.d_rope = self.d_head // 2      # as chosen in deepseek-v2
        self.up_proj_dim = (self.d_heads - self.d_rope) * self.n_heads  # for keys and queries

        # define all down- and up-projections
        self.W_dkv = nn.Linear(h_dim, self.compression_dim, bias=False)
        self.W_dq = nn.Linear(h_dim, self.compression_dim, bias=False)
        self.W_uk = nn.Linear(self.compression_dim, self.up_proj_dim, bias=False)
        self.W_uv = nn.Linear(self.compression_dim, self.n_heads * self.d_head, bias=False)
        self.W_uq = nn.Linear(self.compression_dim, self.up_proj_dim, bias=False)

        # projections to produce decoupled keys and queries
        self.W_kr = nn.Linear(self.h_dim, self.n_heads * self.d_rope, bias=False)
        self.W_qr = nn.Linear(self.compression_dim, self.n_heads * self.d_rope, bias=False)

        # RoPE
        self.RoPE = RotaryPositionalEmbedding(self.d_rope, self.device)

        # final projection to match attention output dimension
        self.out_proj = nn.Linear(self.n_heads * self.d_head, h_dim, bias=False)

    def forward(self, x):
        # create all down-projection latents
        batch_size, seq_len, h_dim = x.shape

        # compute down-projections
        c_kv = self.W_dkv(x)
        c_q = self.W_dq(x)

        # compute up_projections
        q_c = self.W_uq(c_q)
        k_c = self.W_uk(c_kv)
        v = self.W_uv(c_kv)

        # compute decoupled keys and queries
        k_r = self.W_kr(x)
        q_r = self.W_qr(q_c)

        # compute rotary positional embeddings (RoPE)
        k_rope, q_rope = self.RoPE(k_r, q_r)

        # combine to get keys and queries
        k = torch.cat([k_c, k_rope], dim=-1)
        q = torch.cat([q_c, q_rope], dim=-1)

        # calculate attention
        attn_scores = torch.einsum("bqhd, bkhd->bhqk", k, q) / math.sqrt(self.d_head)
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = torch.einsum("bhqk,bkhd->bqhd", attn_scores, v)

        # reassemble the heads and project to the dimension of input x
        output = self.out_proj(out.contiguous().view(batch_size, seq_len, -1))

        return output, (c_kv, k_rope)
