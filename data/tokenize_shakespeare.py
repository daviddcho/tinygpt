#!/usr/bin/env python3
import os 
import numpy as np
import tiktoken

# load data
input_path = os.path.join(os.path.dirname(__file__), 'tiny_shakespeare')
with open(input_path, 'r', encoding='utf-8') as f:
  data = f.read()

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode with tiktoken gpt2 bpe
# https://github.com/openai/tiktoken
enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_ordinary(train_data)
val_ids = enc.encode_ordinary(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# save as binary files
train_ids = np.array(train_ids, dtype=np.int64)
val_ids = np.array(val_ids, dtype=np.int64)

train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))
