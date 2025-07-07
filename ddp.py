#!/usr/bin/env python3
import os
import sys
import tempfile
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import trange
import tempfile, pathlib

class ToyModel(nn.Module):
  def __init__(self):
    super(ToyModel, self).__init__()
    self.net1 = nn.Linear(10, 10)
    self.relu = nn.ReLU()
    self.net2 = nn.Linear(10, 5)

  def forward(self, x):
    return self.net2(self.relu(self.net1(x)))

def setup(rank, world_size):
  #os.environ['MASTER_ADDR'] = 'localhost'
  #os.environ['MASTER_PORT'] = '12355'
  #dist.init_process_group("gloo", rank=rank, world_size=world_size)

  tmp = pathlib.Path(tempfile.gettempdir()) / "ddp_example"
  init_method = f"file://{tmp.as_posix()}"
  dist.init_process_group(
    backend="gloo",
    init_method=init_method,
    rank=rank,
    world_size=world_size,
  )
 
def cleanup():
  dist.destroy_process_group()

def train(rank, world_size):
  print("running basic ddp example on rank", rank)
  setup(rank, world_size)

  print("setup complete")

  device = torch.device("cpu")
  model = ToyModel().to(device)
  ddp_model = DDP(model)

  loss_fn = nn.MSELoss()
  optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

  n_steps = 100
  #progress_bar = trange(n_steps) if rank == 0 else range(n_steps) 

  for i in range(n_steps):
    print(i)
    optimizer.zero_grad()
    out = ddp_model(torch.randn(20, 10)).to(device)
    targets = torch.randn(20, 5).to(device)
    loss_fn(out, targets).backward()
    optimizer.step()

  cleanup()
  print("finished running DDP on rank", rank)

def run():
  rank = int(os.environ["RANK"])
  world_size = int(os.environ["WORLD_SIZE"])
  local_rank = int(os.environ["LOCAL_RANK"]) # useful on multi-GPU boxes

  train(rank, world_size)

if __name__ == "__main__":
  mp.set_start_method("spawn", force=True)
  run()
