import torch
import torch.nn as nn

from src.MultiHeadAttention import MultiHeadAttention
from src import LayerNorm
from src.FeedForward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        print('--- Initializing Transformer Block ---')
        self.att = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length = cfg["context_length"],
            num_heads = cfg["n_heads"],
            dropout =  cfg['drop_rate'],
            qkv_bias = cfg["qkv_bias"]
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm.LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm.LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x

GPT_CONFIG_124M = {"vocab_size": 50257, # Vocabulary size
                   "context_length": 1024, # Context length
                   "emb_dim": 768, # Embedding dimension
                   "n_heads": 12, # Number of attention heads
                   "n_layers": 12, # Number of layers
                   "drop_rate": 0.1, # Dropout rate
                   "qkv_bias": False # Query-Key-Value bias
                   }
def test_transformer_block():
    torch.manual_seed(123)
    x = torch.randn(2,4,768)
    block = TransformerBlock(cfg=GPT_CONFIG_124M)
    output = block(x)
    print('intput:',x.shape)
    print('output',output)
    return

if __name__ == '__main__':
    test_transformer_block()