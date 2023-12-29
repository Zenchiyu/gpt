import torch
import torch.nn as nn


class MHSelfAttention2d(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 nb_heads: int) -> None:
        super().__init__()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B x L x E
        q, k, v = # TODO
        return x

class TransformerBlock(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 mlp_hidden_dim: int,
                 nb_heads: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mha_causal = MHSelfAttention2d(embed_dim, nb_heads)
        self.mlp = nn.Sequential(nn.Linear(embed_dim, mlp_hidden_dim),
                                 nn.GELU(),
                                 nn.Linear(mlp_hidden_dim, embed_dim)
                                 )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mha_causal(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x