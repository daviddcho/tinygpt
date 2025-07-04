#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
from model import Transformer
from dataloader import EnWik8Dataset
from tqdm import trange
import os

N_EPOCHS = 50000
D_MODEL = 768
N_HEADS = 12
N_LAYERS = 10
D_FF = 2048
MAX_SEQ_LEN = 512
BS = 8
LR = 3e-4

def train_model():
    run = wandb.init(entity='davidcho', project='llm-wiki', config={
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "n_layers": N_LAYERS,
        "d_ff": D_FF,
        "max_seq_len": MAX_SEQ_LEN,
        "bs": BS,
        "learning_rate": LR,
    })
    device = torch.device('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 
                          'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = EnWik8Dataset('data/enwik8.zip', seq_len=MAX_SEQ_LEN)
    
    # Initialize model
    model = Transformer(
        vocab_size=dataset.vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN
    ).to(device)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    criterion = nn.CrossEntropyLoss()
    
    # Training parameters
    eval_interval = 500
    batch_size = BS
    
    # Training loop
    model.train()
    for iter_num in trange(N_EPOCHS, desc="Training"):
        # Get batch
        X_train, Y_train = dataset.get_batch('train', batch_size)
        X_train, Y_train = X_train.to(device), Y_train.to(device)
        
        # Forward pass
        logits = model(X_train)
        loss = criterion(logits.view(-1, logits.size(-1)), Y_train.view(-1))
        accuracy = calculate_accuracy(logits, Y_train)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        run.log({'accuracy': accuracy, 'loss': loss.item()})
        
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
            probs = F.softmax(logits / temperature, dim=-1)
            # Top-k sampling 
            top_k = 10
            top_probs, top_indices = torch.topk(probs, top_k)
            top_probs = top_probs / top_probs.sum()
            next_token = top_indices[torch.multinomial(top_probs, 1)]
            
            generated.append(next_token.item())
            context = torch.cat([context, next_token.unsqueeze(0)], dim=1)
        
        # Convert indices back to characters
        return ''.join([dataset.idx_to_char[idx] for idx in generated])

if __name__ == "__main__":
    print("Starting transformer training on enwik8...")
    model, dataset = train_model()
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': dataset.vocab_size,
        'char_to_idx': dataset.char_to_idx,
        'idx_to_char': dataset.idx_to_char
    }, 'weights/transformer_enwik8.pth')
    
    print("Training completed! Model saved as 'transformer_enwik8.pth'")
