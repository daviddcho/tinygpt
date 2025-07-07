#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
from model import Transformer
from dataloader import TinyShakespeare
from tqdm import trange
import os
import math

N_EPOCHS = 5000
MAX_LR = 1e-4
MIN_LR = MAX_LR * 0.1
WARMUP_STEPS = 500

from config.tiny import cfg
import sys 
if len(sys.argv) > 1:
  exec(open(f"config/{sys.argv[1]}.py").read())
  print(f"loaded sys.argv[1]")
globals().update(vars(cfg))


def train(rank, world_size):
  print(f"starting... {rank}")
  dist.init_process_group(
    backend="nccl" if torch.cuda.is_available() else 'gloo',
    init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
    rank=rank,
    world_size=world_size,
  )

  if torch.cuda.is_available():
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
  else: 
    device = torch.device('cpu')
  print(f"Using device: {device}\n")

  dataset = TinyShakespeare(MAX_SEQ_LEN)
  
  model = Transformer(
    vocab_size=dataset.vocab_size,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    d_ff=D_FF,
    max_seq_len=MAX_SEQ_LEN
  ).to(device)
  if torch.cuda.is_available():
    model = DDP(model, device_ids=[local_rank])
  else:
    model = DDP(model)
  
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
  progress_bar = trange(N_EPOCHS) if rank == 0 else range(N_EPOCHS) 
  for iter_num in (t := progress_bar):
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
    
    # gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    optimizer.step()
    
    dist.barrier()
    if dist.get_rank() == 0:
      wandb.log({'accuracy': accuracy, 'loss': loss.item(), 'lr': lr})
      t.set_description("loss %.2f accuracy %.2f" % (loss.item(), accuracy))
   
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
        
        dist.barrier()
        if dist.get_rank() == 0:
          print(f"\niter: {iter_num}, val loss: {val_loss.item():.4f}, val acc: {val_acc:.4f}")
          wandb.log({'val_acc': val_acc, 'val_loss': val_loss})
      model.train()

  dist.barrier()
  if dist.get_rank() == 0:
    os.makedirs('weights', exist_ok=True)
    torch.save({
      'model_state_dict': model.state_dict(),
      'vocab_size': dataset.vocab_size,
    }, f"weights/model-{wandb.run.name}.pth")
    print(f"Training completed! Model saved as weights/model-{wandb.run.name}.pth")
  dist.destroy_process_group()

def calculate_accuracy(logits, targets):
  predictions = torch.argmax(logits, dim=-1)
  correct = (predictions == targets).float()
  return correct.mean().item()

if __name__ == "__main__":
  #mp.set_start_method("spawn", force=True)
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  local_rank = int(os.environ["LOCAL_RANK"]) # useful on multi-gpu boxes

  if rank == 0:
    import wandb
    print(wandb.__file__)
    wandb.init(entity='davidcho', project='transformer-multigpu', config={
      "d_model": D_MODEL,
      "n_heads": N_HEADS,
      "n_layers": N_LAYERS,
      "d_ff": D_FF,
      "max_seq_len": MAX_SEQ_LEN,
      "bs": BS,
      "max_lr": MAX_LR,
    })

  train(rank, world_size)
