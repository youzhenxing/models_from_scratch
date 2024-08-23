import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):
        super().__init__()
        assert(d_out % num_heads == 0),'d_out must be divisible by num_heads'

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out, d_out)
        self.register_buffer('mask',torch.triu(torch.ones(context_length,context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        keys = keys.view(b,num_tokens, self.num_heads, self.head_dim)
        values = values.view(b,num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b,num_tokens,self.num_heads,self.head_dim)

        keys = keys.transpose(1,2)
        values = values.transpose(1,2)
        queries = queries.transpose(1,2)

        attns_scores = queries @ keys.transpose(2,3)
        mask_bool = self.mask.bool()[:num_tokens,:num_tokens]
        attns_scores = attns_scores.masked_fill(mask_bool,-torch.inf)

        attns_weights = torch.softmax(attns_scores/keys.shape[-1] ** 0.5, dim=-1)
        attns_weights = self.dropout(attns_weights)

        context_vec = (attns_weights @ values).transpose(1,2)

        context_vec = context_vec.contiguous().view(b,num_tokens, self.d_out)

        context_vec = self.out_proj(context_vec)
        return context_vec


