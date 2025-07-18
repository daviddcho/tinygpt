#!/usr/bin/env python3
import os
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets
from tqdm import trange

num_proc = 8

enc = tiktoken.get_encoding("gpt2")

if __name__ == "__main__":
  dataset = load_dataset("openwebtext", num_proc=num_proc)

  split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
  split_dataset['val'] = split_dataset.pop('test')

  def process(example):
    ids = enc.encode_ordinary(example['text'])
    ids.append(enc.eot_token)
    out = {'ids': ids, 'len': len(ids)}
    return out
  
  tokenized = split_dataset.map(
    process,
    remove_columns=['text'],
    desc="tokenizing the splits",
    num_proc=num_proc,
  )

  for split, dset in tokenized.items():
    arr_len = np.sum(dset['len'], dtype=np.uint64)
    filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
    dtype = np.uint16
    arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
    total_batches = 1024

    idx = 0
    for batch_idx in trange(total_batches):
      batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
      arr_batch = np.concatenate(batch['ids'])
      arr[idx : idx + len(arr_batch)] = arr_batch
      idx += len(arr_batch)
    arr.flush()
