import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self,d_in, d_out, context_length, dropout, qkv_bias = False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_keys = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask',torch.triu(torch.ones(context_length,context_length),diagonal=1))
    def forward(self,x):
         b, num_tokens, d_in = x.shape
         keys = self.W_keys(x)
         values = self.W_value(x)
         queries = self.W_query(x)
         print('keys:',keys)
         print('keys.transpose(1,2):',keys.transpose(1,2))
         attn_scores = queries @ keys.transpose(1,2)
         attn_scores.masked_fill(self.mask.bool()[:num_tokens,:num_tokens], - torch.inf)
         attn_weights = self.dropout(attn_scores)

         context_vec = attn_weights @ values
         return context_vec

class MultiHeadAttentionWrapper(nn.Module):
        def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):
            super().__init__()
            self.heads = nn.ModuleList([CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
                                       for _ in range(num_heads)])
        def forward(self, x):
            return torch.cat([head(x) for head in self.heads], dim = -1)
