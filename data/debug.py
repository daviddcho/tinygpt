#!/usr/bin/env python3
import tiktoken
import numpy as np 

tokenizer = tiktoken.get_encoding('gpt2')
data = np.memmap('train.bin', dtype=np.int64)
count = np.bincount(data)

#x = tokenizer.encode('!')[0]
#print("! token id", x)
#print("n !", count[x])

k = 20
top_ids = np.argpartition(-count, k)[:k] 
top_ids = top_ids[np.argsort(-count[top_ids])]

for rank, tid in enumerate(top_ids, 1):
  token = tokenizer.decode([tid])
  print(rank, tid, count[tid], repr(token))
