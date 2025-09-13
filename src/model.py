from torch import nn as nn
import torch
from src.config import Config



class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, drop_out=0.1):
        super().__init__()

        assert embed_dim % num_heads == 0

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(drop_out)
        self.scale = (embed_dim // num_heads) ** 0.5
        self.embed_dim = embed_dim
        self.num_heads = num_heads

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor = None,
        pad_mask: torch.Tensor = None,
    ):
        bs = q.shape[0]
        q_len = q.shape[1]
        k_len = k.shape[1]

        Q: torch.Tensor = self.q_proj(q)
        K: torch.Tensor = self.k_proj(k)
        V: torch.Tensor = self.v_proj(v)

        q_state = Q.view(bs, q_len, self.num_heads, -1).transpose(1, 2)
        k_state = K.view(bs, k_len, self.num_heads, -1).transpose(1, 2)
        v_state = V.view(bs, k_len, self.num_heads, -1).transpose(1, 2)

        attn = q_state @ k_state.transpose(
            -1, -2
        )  # [bs, head, q_len, dim] @ [bs, head, dim, k_len] = [bs, head, q_len, k_len]
        attn: torch.Tensor = attn / self.scale

        if mask is not None:
            attn = attn.masked_fill(~mask, -1e8)

        if pad_mask is not None:
            attn = attn.masked_fill(~pad_mask.unsqueeze(1).unsqueeze(2), -1e8)

        attn = torch.softmax(attn, dim=-1)

        attn = self.dropout(attn)

        out = (
            attn @ v_state
        )  # [bs, head, q_len, k_len] @ [bs, head, k_len, dim] = [bs, head, q_len, dim]

        out = out.transpose(1, 2).contiguous().view(bs, q_len, -1)

        out = self.out_proj(out)

        return out


class FFN(nn.Module):
    def __init__(self, embed_dim, drop_out=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(drop_out),
        )

    def forward(self, x):
        return self.mlp(x)


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, drop_out=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(
            embed_dim=embed_dim, num_heads=num_heads, drop_out=drop_out
        )
        self.ffn = FFN(embed_dim=embed_dim, drop_out=drop_out)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, pad_mask=None):
        x = x + self.mha(x, x, x, pad_mask=pad_mask)
        x = self.norm1(x)

        x = x + self.ffn(x)

        x = self.norm2(x)

        return x


class Encoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(config.encoder_layer):
            self.layers.append(
                EncoderLayer(config.embed_dim, config.num_heads, config.drop_out)
            )

    def forward(self, x, pad_mask=None):
        for layer in self.layers:
            x = layer(x, pad_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, drop_out=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(
            embed_dim=embed_dim, num_heads=num_heads, drop_out=drop_out
        )
        self.cross_attn = MultiHeadAttention(
            embed_dim=embed_dim, num_heads=num_heads, drop_out=drop_out
        )
        self.ffn = FFN(embed_dim=embed_dim, drop_out=drop_out)
        self.norm0 = nn.LayerNorm(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, memory, src_pad_mask=None, tgt_pad_mask=None):

        x_len = x.shape[1]
        mask = torch.ones(size=(1, 1, x_len, x_len), device=x.device, dtype=torch.bool).tril()

        x = x + self.self_attn(x, x, x, mask=mask, pad_mask=tgt_pad_mask)

        x = self.norm0(x)

        x = x + self.cross_attn(x, memory, memory, pad_mask=src_pad_mask)
        x = self.norm1(x)

        x = x + self.ffn(x)

        x = self.norm2(x)

        return x


class Decoder(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(config.decoder_layer):
            self.layers.append(
                DecoderLayer(config.embed_dim, config.num_heads, config.drop_out)
            )

    def forward(self, x: torch.Tensor, memory, src_pad_mask=None, tgt_pad_mask=None):
        for layer in self.layers:
            x = layer(x, memory, src_pad_mask=src_pad_mask, tgt_pad_mask=tgt_pad_mask)
        return x


class PositionEmbedding(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        pe = torch.zeros(config.max_len, config.embed_dim)
        pos = torch.arange(0, config.max_len, 1).float().unsqueeze(1)
        _2i = torch.arange(0, config.embed_dim, 2)
        pe[:, 0::2] = torch.sin(pos / (10000 ** (_2i / config.embed_dim)))
        pe[:, 1::2] = torch.cos(pos / (10000 ** (_2i / config.embed_dim)))

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x_len = x.shape[1]
        return x + self.pe[:, :x_len].to(dtype=x.dtype)


class TranslateModel(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.position_embedding = PositionEmbedding(config=config)
        self.encoder = Encoder(config=config)
        self.decoder = Decoder(config=config)
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size)
        self.drop = nn.Dropout(config.drop_out)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_pad_mask=None,
        tgt_pad_mask=None,
    ):

        ## encoder

        src_embedding = self.embedding(src)
        src_embedding = self.position_embedding(src_embedding)
        memory = self.encoder.forward(src_embedding, src_pad_mask)

        tgt_embedding = self.embedding(tgt)
        tgt_embedding = self.position_embedding(tgt_embedding)

        output = self.decoder.forward(tgt_embedding, memory, src_pad_mask, tgt_pad_mask)

        output = self.drop(output)

        output = self.head(output)

        return output
