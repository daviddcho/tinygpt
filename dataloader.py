import torch
import numpy as np
import zipfile

class TinyShakespeare:
  def __init__(self, block_size):
    self.block_size = block_size

    train_data = np.memmap('data/train.bin', dtype=np.int64)
    val_data = np.memmap('data/val.bin', dtype=np.int64)
    self.vocab_size = max(train_data.max(), val_data.max()) + 1
    print(f"vocab size: {self.vocab_size}")

  def get_batch(self, split, batch_size):
    if split == 'train':
      data = np.memmap('data/train.bin')
    else:
      data = np.memmap('data/val.bin')

    #block_size = MAX_SEQ_LEN
    ix = torch.randint(len(data) - self.block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+self.block_size]).astype(np.int64)) for i in ix]).long()
    y = torch.stack([torch.from_numpy((data[i+1:i+1+self.block_size]).astype(np.int64)) for i in ix]).long()

    return x, y

# TODO: use bpe
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
    
    chars = sorted(list(set(text)))
    self.vocab_size = len(chars)
    print(f"Vocabulary size: {self.vocab_size}")
    
    self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
    self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
    
    # text to indices
    data = np.array([self.char_to_idx[ch] for ch in text], dtype=np.int32)
    
    split_idx = int(len(data) * train_split)
    self.train_data = data[:split_idx]
    self.test_data = data[split_idx:]
    
    print(f"Train size: {len(self.train_data):,}")
    print(f"Test size: {len(self.test_data):,}")
  
  def get_batch(self, split, batch_size=32):
    data = self.train_data if split == 'train' else self.test_data
    
    ix = np.random.randint(0, len(data) - self.seq_len, size=batch_size)
    # input sequences and targets
    x = np.stack([data[i:i+self.seq_len] for i in ix])
    y = np.stack([data[i+1:i+self.seq_len+1] for i in ix])
    
    return torch.from_numpy(x).long(), torch.from_numpy(y).long()
