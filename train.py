#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb
from model import Transformer
from dataloader import EnWik8Dataset, TinyShakespeare
import tiktoken
from tqdm import trange
import os
import math

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

N_EPOCHS = 5000
MAX_LR = 1e-4
MIN_LR = MAX_LR * 0.1
WARMUP_STEPS = 500

from config.tiny import cfg
import sys 
if len(sys.argv) > 1:
  exec(open(f"config/{sys.argv[1]}.py").read())
globals().update(vars(cfg))

device = torch.device(
  'cuda' if torch.cuda.is_available() else 
  'mps' if torch.backends.mps.is_available() else 
  'cpu'
)
print(f"Using device: {device}")

def train_model():
  run = wandb.init(entity='davidcho', project='llm-wiki', config={
    "d_model": D_MODEL,
    "n_heads": N_HEADS,
    "n_layers": N_LAYERS,
    "d_ff": D_FF,
    "max_seq_len": MAX_SEQ_LEN,
    "bs": BS,
    "max_lr": MAX_LR,
  })
  
  dataset = TinyShakespeare(MAX_SEQ_LEN)
  
  model = Transformer(
    vocab_size=dataset.vocab_size,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    d_ff=D_FF,
    max_seq_len=MAX_SEQ_LEN
  ).to(device)
  
  optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=0.1)
  criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
  
  eval_interval = 500
  batch_size = BS
  
  def get_lr(iter_num):
    # linear warmup
    if iter_num < WARMUP_STEPS:
      return MAX_LR * iter_num / WARMUP_STEPS
    # cosine decay
    if iter_num > N_EPOCHS:
      return MIN_LR
    decay_ratio = (iter_num - WARMUP_STEPS) / (N_EPOCHS - WARMUP_STEPS)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + (MAX_LR - MIN_LR) * cosine_decay

  model.train()
  for iter_num in (t := trange(N_EPOCHS)):
    X_train, Y_train = dataset.get_batch('train', batch_size)
    X_train, Y_train = X_train.to(device), Y_train.to(device)
    
    lr = get_lr(iter_num)
    for param_group in optimizer.param_groups:
      param_group['lr'] = lr
    
    logits = model(X_train)

    # entropy regularization
    probs = torch.softmax(logits, dim=-1)
    entropy = - (probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()

    loss = criterion(logits.view(-1, logits.size(-1)), Y_train.view(-1)) + 0.01 * entropy
    accuracy = calculate_accuracy(logits, Y_train)
    
    optimizer.zero_grad()
    loss.backward()
    run.log({'accuracy': accuracy, 'loss': loss.item(), 'lr': lr})
    t.set_description("loss %.2f accuracy %.2f" % (loss.item(), accuracy))
    
    # gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    
    # eval
    if iter_num % eval_interval == 0:
      model.eval()
      with torch.no_grad():
        train_acc = calculate_accuracy(logits, Y_train)
        
        X_test, Y_test = dataset.get_batch('test', batch_size)
        X_test, Y_test = X_test.to(device), Y_test.to(device)
        
        val_logits = model(X_test)
        val_loss = criterion(val_logits.view(-1, val_logits.size(-1)), Y_test.view(-1))
        val_acc = calculate_accuracy(val_logits, Y_test)
        
        print(f"\niter: {iter_num}, val loss: {val_loss.item():.4f}, val acc: {val_acc:.4f}")
        run.log({'val_acc': val_acc, 'val_loss': val_loss})
        
        sample_text = generate_sample(model, dataset.vocab_size, device, length=100)
        print(f"Sample: {sample_text}")
        
      model.train()
  
  return model, dataset

def calculate_accuracy(logits, targets):
  predictions = torch.argmax(logits, dim=-1)
  correct = (predictions == targets).float()
  return correct.mean().item()

def generate_sample(model, vocab_size, device, length=100, temperature=0.3):
  model.eval()
  tokenizer = tiktoken.get_encoding("gpt2")
  with torch.no_grad():
    # Start with a random token
    context = torch.randint(0, vocab_size, (1, 1)).to(device)
    print(f"context: {tokenizer.decode([context.item()])}")
    
    generated = []
    for _ in range(length):
      if context.size(1) > model.max_seq_len:
        context = context[:, -model.max_seq_len:]
        
      logits = model(context)
      logits = logits[0, -1, :]
      
      # Sample from the distribution
      probs = F.softmax(logits / temperature, dim=-1)
      
      next_token = torch.multinomial(probs, 1)
      
      generated.append(next_token.item())
      context = torch.cat([context, next_token.unsqueeze(0)], dim=1)
    
    return tokenizer.decode(generated)

if __name__ == "__main__":
  print("Starting transformer training...")
  model, dataset = train_model()
  
  try:
    os.mkdir('weights')
  except FileExistsError:
    pass
  torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_size': dataset.vocab_size,
  }, 'weights/transformer_enwik8.pth')
  
  print("Training completed! Model saved as 'transformer_enwik8.pth'")
