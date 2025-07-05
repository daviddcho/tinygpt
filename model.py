import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, n_heads):
    super().__init__()
    assert d_model % n_heads == 0
    
    self.d_model = d_model
    self.n_heads = n_heads
    self.d_k = d_model // n_heads
    
    self.w_q = nn.Linear(d_model, d_model, bias=False)
    self.w_k = nn.Linear(d_model, d_model, bias=False)
    self.w_v = nn.Linear(d_model, d_model, bias=False)
    self.w_o = nn.Linear(d_model, d_model, bias=False)
    
  def attention(self, q, k, v, mask=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
    
    if mask is not None:
      scores = scores.masked_fill(mask == 0, -1e9)
      #for mps autocasting to half precision
      #scores = scores.masked_fill(mask == 0, -1e4)
      
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, v)
    
    return output, attention_weights
  
  def forward(self, x, mask=None):
    batch_size, seq_len, d_model = x.size()
    
    q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
    k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
    v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
    
    attention_output, attention_weights = self.attention(q, k, v, mask)
    
    attention_output = attention_output.transpose(1, 2).contiguous().view(
      batch_size, seq_len, d_model
    )
    
    output = self.w_o(attention_output)
    return output

class FeedForward(nn.Module):
  def __init__(self, d_model, d_ff):
    super().__init__()
    self.linear1 = nn.Linear(d_model, d_ff)
    self.linear2 = nn.Linear(d_ff, d_model)
    self.dropout = nn.Dropout(0.1)
    
  def forward(self, x):
    return self.linear2(self.dropout(F.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
  def __init__(self, d_model, n_heads, d_ff):
    super().__init__()
    self.attention = MultiHeadAttention(d_model, n_heads)
    self.feed_forward = FeedForward(d_model, d_ff)
    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.dropout = nn.Dropout(0.1)
    
  def forward(self, x, mask=None):
    # Self-attention with residual connection
    attention_output = self.attention(x, mask)
    x = self.norm1(x + self.dropout(attention_output))
    
    # Feed forward with residual connection
    ff_output = self.feed_forward(x)
    x = self.norm2(x + self.dropout(ff_output))
    
    return x

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_seq_len=5000):
    super().__init__()
    
    pe = torch.zeros(max_seq_len, d_model)
    position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
    
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
               (-math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    self.register_buffer('pe', pe.unsqueeze(0))
    
  def forward(self, x):
    return x + self.pe[:, :x.size(1)]

class Transformer(nn.Module):
  def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=8, d_ff=2048, max_seq_len=1024):
    super().__init__()
    self.d_model = d_model
    self.vocab_size = vocab_size
    self.max_seq_len = max_seq_len
    
    self.embedding = nn.Embedding(vocab_size, d_model)
    nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
    
    self.transformer_blocks = nn.ModuleList([
      TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
    ])
    
    self.ln_f = nn.LayerNorm(d_model)
    self.head = nn.Linear(d_model, vocab_size, bias=False)
    
    self.dropout = nn.Dropout(0.1)
    
    # Parameter count: ~10M
    print(f"Model parameters: {sum(p.numel() for p in self.parameters()):,}")
    
  def create_causal_mask(self, seq_len):
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask.unsqueeze(0).unsqueeze(0)
  
  def forward(self, x):
    seq_len = x.size(1)
    
    # Token embeddings + positional encoding
    token_embeddings = self.embedding(x) * math.sqrt(self.d_model)
    x = self.dropout(self.pos_encoding(token_embeddings))
    
    # Causal mask for autoregressive generation
    mask = self.create_causal_mask(seq_len).to(x.device)
    
    # Pass through transformer blocks
    for transformer_block in self.transformer_blocks:
      x = transformer_block(x, mask)
      
    # Final layer norm and output projection
    x = self.ln_f(x)
    logits = self.head(x)
    
    return logits

