import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self,d_in, d_out):
        super().__init__();
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        values = x @ self.W_value
        query = x @ self.W_query
        attn_scores = query @ keys.T
        attn_weights = torch.softmax(attn_scores/keys.shape[-1] ** 0.5, dim=-1)
        context_vector = attn_weights @ values
        return context_vector