import torch
import torch.nn as nn

from torch.distributions.categorical import Categorical
from tqdm import tqdm
from .blocks import TransformerBlock


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
        self.max_seq_len = max_seq_len
        self.wte = nn.Embedding(vocab_size, embed_dim)  # token embedding

        # Positional encoding
        pos = torch.arange(max_seq_len)[:, None]
        i = torch.arange(embed_dim)[None, :]
        # TODO: self.max_seq_len or 10_000 ?
        angles = pos/(10_000**((2 * (i//2))/embed_dim))  # if i is even, then i+1 has the same angle
        pe = torch.empty(max_seq_len, embed_dim, dtype=torch.float)
        pe[:, 0::2] = torch.sin(angles[:, 0::2])
        pe[:, 1::2] = torch.cos(angles[:, 1::2])
        self.register_buffer('pe', pe.unsqueeze(0))
        # https://fleuret.org/dlc/materials/dlc-handout-6-6-using-GPUs.pdf
        # https://fleuret.org/dlc/materials/dlc-handout-10-3-NVP.pdf

        self.layers = nn.Sequential(
            nn.Dropout(p=0.1),  # TODO: find dropout used in literature
            *[TransformerBlock(embed_dim=embed_dim, nb_heads=nb_heads, mlp_hidden_dim=mlp_hidden_dim) for _ in range(nb_layers)],
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, vocab_size)  # logits: batch_size x max_seq_len x vocab_size
        )

    def wpe(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, :x.shape[-1]]

    def generate(self,
            x: torch.Tensor,
            nb_tokens: int=50,
            sampling_mode: str="prob",
            temperature: float=1) -> torch.Tensor:  # TODO: add temperature
        y = x.new_empty((x.shape[0], nb_tokens), dtype=torch.int)
        match sampling_mode:
            case "argmax":
                for l in tqdm(range(nb_tokens)):
                    logits = self(x)[0, :, -1]  # V
                    y[:, l] = torch.argmax(logits)
                    x = torch.cat([x, y[:, l][:, None]], dim=1)[:, -self.max_seq_len:]
            case "top5":
                for l in tqdm(range(nb_tokens)):
                    logits = self(x)[0, :, -1]  # V
                    new_logits, indices = torch.topk(logits, 5)
                    y[:, l] = indices[Categorical(logits=new_logits).sample()]
                    x = torch.cat([x, y[:, l][:, None]], dim=1)[:, -self.max_seq_len:]
            case _:  # prob
                for l in tqdm(range(nb_tokens)):
                    logits = self(x)[0, :, -1]  # V
                    y[:, l] = Categorical(logits=logits).sample()
                    x = torch.cat([x, y[:, l][:, None]], dim=1)[:, -self.max_seq_len:]
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch of seq. of token ids N x L
        # -> batch of seq. of token embeddings N x L x C
        # -> batch of seq. of logits N x L x V
        # -> N x V x L
        return self.layers(self.wte(x) + self.wpe(x)).transpose(-1, -2)