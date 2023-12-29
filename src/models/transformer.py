import torch
import torch.nn as nn

from blocks import TransformerBlock


class Transformer(nn.Module):
    def __init__(self,
            vocab_size: int,
            max_seq_len: int,
            embed_dim: int,
            mlp_hidden_dim: int,
            nb_layers: int,
            nb_heads=int
    ) -> None:
        super().__init__()
        self.wte = nn.Embedding(vocab_size, embed_dim)  # token embedding
        self.wpe = None  # TODO, use max_seq_len here?
        self.layers = nn.Sequential(
            nn.Dropout(p=0.1),  # TODO: find dropout used in literature
            *[TransformerBlock(nb_heads=nb_heads, mlp_hidden_dim=mlp_hidden_dim) for _ in range(nb_layers)],
            nn.LayerNorm(embed_dim),  # TODO: to verify. input is batch_size x max_seq_len x embed_dim
            nn.Linear(embed_dim, vocab_size)  # logits: batch_size x max_seq_len x vocab_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch of seq. of token ids B x L
        x = self.dropout(self.wte(x) + self.wpe(x))
        return self.layers(x)