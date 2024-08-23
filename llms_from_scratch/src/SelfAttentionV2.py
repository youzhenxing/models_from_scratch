import torch
import torch.nn as nn

class SelfAttention_v2(nn.Module):
    def __init__(self,d_in, d_out,qkv_bias):
        super().__init__();
        self.W_query = nn.Linear(d_in, d_out,bias=qkv_bias)
        self.W_keys = nn.Linear(d_in, d_out,bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out,bias = qkv_bias)

    def forward(self, x):
        keys = self.W_keys(x)
        values = self.W_value(x)
        query = self.W_query(x)
        attn_scores = query @ keys.T
        attn_weights = torch.softmax(attn_scores/keys.shape[-1] ** 0.5, dim=-1)
        context_vector = attn_weights @ values
        return context_vector