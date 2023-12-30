import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MHCausalSelfAttention(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 nb_heads: int) -> None:
        super().__init__()
        assert embed_dim % nb_heads == 0, "q,k,v dim: embed_dim/nb_heads should be an integer"
        self.nb_heads = nb_heads
        self.qkv_proj = nn.Conv1d(embed_dim, 3*embed_dim, kernel_size=1, bias=False)    # TODO: bias in literature ?
        self.conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, bias=False)          # TODO: bias in literature ?
        # TODO: dropout in literature ?

    def scaled_dot_product_attention(self,
            Q: torch.Tensor,
            K: torch.Tensor,
            V: torch.Tensor,
            attn_mask: torch.Tensor
        ) -> torch.Tensor:
        # Can replace this by F.scaled_dot_product_attention in practice
        score = torch.einsum("...ld,...sd->...ls", Q, K)
        score[~attn_mask] = -torch.inf
        A = torch.softmax(score/np.sqrt(Q.shape[-1]), dim=-1)
        Y = torch.einsum("...ls,...sd->...ld", A, V)
        return Y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, L, C = x.shape
        qkv = self.qkv_proj(x.transpose(-1, -2))                                        # N x L x C -> N x C x L -> N x 3C x L
        qkv = qkv.view(N, 3*self.nb_heads, C//self.nb_heads, L)                         # N x 3nb_heads x C//nb_heads x L
        Q, K, V = (qkv.transpose(-1, -2)).chunk(3, dim=1)                               # N x nb_heads x L x C//nb_heads each
        attn_mask = torch.tril(torch.ones(N, self.nb_heads, L, L, dtype=torch.bool))    # causal attention mask
        Y = self.scaled_dot_product_attention(Q, K, V, attn_mask=attn_mask)             # N x nb_heads x L x C//nb_heads
        y = self.conv(Y.transpose(-1, -2).reshape(N, C, L)).transpose(-1, -2)           # N x nb_heads x C//nb_heads x L -> N x C x L -> N x L x C
        return y

class TransformerBlock(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 mlp_hidden_dim: int,
                 nb_heads: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mha_causal = MHCausalSelfAttention(embed_dim, nb_heads)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, mlp_hidden_dim),
                                 nn.GELU(),
                                 nn.Linear(mlp_hidden_dim, embed_dim)
                                 )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mha_causal(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x