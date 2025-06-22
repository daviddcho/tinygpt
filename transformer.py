import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from tqdm import trange
import os
import zipfile

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

class EnWik8Dataset:
    def __init__(self, data_path, seq_len=256, train_split=0.9):
        self.seq_len = seq_len
        
        # Load and preprocess data
        if data_path.endswith('.zip'):
            with zipfile.ZipFile(data_path, 'r') as zip_file:
                with zip_file.open('enwik8') as file:
                    text = file.read().decode('utf-8', errors='ignore')
        else:
            with open(data_path, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
        
        print(f"Dataset size: {len(text):,} characters")
        
        # Create character vocabulary
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        print(f"Vocabulary size: {self.vocab_size}")
        
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        
        # Convert text to indices
        data = np.array([self.char_to_idx[ch] for ch in text], dtype=np.int32)
        
        # Create train/test split
        split_idx = int(len(data) * train_split)
        self.train_data = data[:split_idx]
        self.test_data = data[split_idx:]
        
        print(f"Train size: {len(self.train_data):,}")
        print(f"Test size: {len(self.test_data):,}")
    
    def get_batch(self, split, batch_size=32):
        data = self.train_data if split == 'train' else self.test_data
        
        # Random starting positions
        ix = np.random.randint(0, len(data) - self.seq_len, size=batch_size)
        
        # Create input sequences and targets
        x = np.stack([data[i:i+self.seq_len] for i in ix])
        y = np.stack([data[i+1:i+self.seq_len+1] for i in ix])
        
        return torch.from_numpy(x).long(), torch.from_numpy(y).long()

def train_model():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 
                          'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = EnWik8Dataset('data/enwik8.zip', seq_len=256)
    
    # Initialize model
    model = Transformer(
        vocab_size=dataset.vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=8,
        d_ff=2048,
        max_seq_len=256
    ).to(device)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()
    
    # Training parameters
    max_iters = 10000
    eval_interval = 500
    batch_size = 32
    
    # Training loop
    model.train()
    for iter_num in trange(max_iters, desc="Training"):
        # Get batch
        X_train, Y_train = dataset.get_batch('train', batch_size)
        X_train, Y_train = X_train.to(device), Y_train.to(device)
        
        # Forward pass
        logits = model(X_train)
        loss = criterion(logits.view(-1, logits.size(-1)), Y_train.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Evaluation
        if iter_num % eval_interval == 0:
            model.eval()
            with torch.no_grad():
                # Training accuracy
                train_acc = calculate_accuracy(logits, Y_train)
                
                # Validation loss and accuracy
                X_test, Y_test = dataset.get_batch('test', batch_size)
                X_test, Y_test = X_test.to(device), Y_test.to(device)
                
                val_logits = model(X_test)
                val_loss = criterion(val_logits.view(-1, val_logits.size(-1)), Y_test.view(-1))
                val_acc = calculate_accuracy(val_logits, Y_test)
                
                print(f"\nIter {iter_num}: Train Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.4f}")
                
                # Generate sample text
                sample_text = generate_sample(model, dataset, device, length=100)
                print(f"Sample: {sample_text[:50]}...")
                
            model.train()
    
    return model, dataset

def calculate_accuracy(logits, targets):
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == targets).float()
    return correct.mean().item()

def generate_sample(model, dataset, device, length=100, temperature=0.8):
    model.eval()
    with torch.no_grad():
        # Start with a random character
        context = torch.randint(0, dataset.vocab_size, (1, 1)).to(device)
        
        generated = []
        for _ in range(length):
            if context.size(1) > model.max_seq_len:
                context = context[:, -model.max_seq_len:]
                
            logits = model(context)
            logits = logits[0, -1, :] / temperature
            
            # Sample from the distribution
            probs = F.softmax(logits, dim=-1)
            next_char = torch.multinomial(probs, 1)
            
            generated.append(next_char.item())
            context = torch.cat([context, next_char.unsqueeze(0)], dim=1)
        
        # Convert indices back to characters
        return ''.join([dataset.idx_to_char[idx] for idx in generated])

def load_model_and_chat(model_path, max_length=200, temperature=0.8):
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.mps.is_available() else 
                          'cpu')
    print(f"Loading model on {device}...")
    
    checkpoint = torch.load(model_path, map_location=device)
    vocab_size = checkpoint['vocab_size']
    char_to_idx = checkpoint['char_to_idx']
    idx_to_char = checkpoint['idx_to_char']
    
    model = Transformer(
        vocab_size=vocab_size,
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_seq_len=256
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("Model loaded successfully! Type 'quit' to exit.")
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() == 'quit':
            break
            
        if not user_input:
            continue
            
        try:
            input_indices = [char_to_idx.get(ch, 0) for ch in user_input]
            context = torch.tensor([input_indices], dtype=torch.long).to(device)
            
            generated_text = generate_text(model, context, idx_to_char, max_length, temperature)
            print(f"Bot: {generated_text}")
            
        except Exception as e:
            print(f"Error generating response: {e}")

def generate_text(model, context, idx_to_char, max_length=200, temperature=0.8):
    model.eval()
    generated = []
    
    with torch.no_grad():
        for _ in range(max_length):
            if context.size(1) > model.max_seq_len:
                context = context[:, -model.max_seq_len:]
                
            logits = model(context)
            logits = logits[0, -1, :] / temperature
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            next_char = idx_to_char.get(next_token.item(), '')
            if next_char in ['\n', '.', '!', '?'] and len(generated) > 10:
                generated.append(next_char)
                break
                
            generated.append(next_char)
            context = torch.cat([context, next_token.unsqueeze(0)], dim=1)
    
    return ''.join(generated)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'chat':
        model_path = sys.argv[2] if len(sys.argv) > 2 else 'transformer_enwik8.pth'
        load_model_and_chat(model_path)
    else:
        print("Starting transformer training on enwik8...")
        model, dataset = train_model()
        
        # Save the model
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab_size': dataset.vocab_size,
            'char_to_idx': dataset.char_to_idx,
            'idx_to_char': dataset.idx_to_char
        }, 'transformer_enwik8.pth')
        
        print("Training completed! Model saved as 'transformer_enwik8.pth'")
