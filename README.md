# tinygpt

Training a transformer on a tiny shakespeare dataset.

## Usage 
```
# create train and val dataset with gpt2 BPE
./data/tokenize_shakespeare.py

# To train tiny model (27M) 
caffeinate ./train.py tiny

# train small model (124M)
caffeinate ./train.py small
```

Trained a 124M transformer on 301K tokens (50k vocab size) for 10k steps on a RTX 4090. Implemented linear learning rate warmup with cosine decay, entropy regularization, and gradient clipping.

Generated sample from our 124M model:
```
William serve this gentleman,
For what you know your father's house.
I'll tell me.
The same ancient feast, and there'st on the prince my young prince,
What!
What's welcome homely.
What is not Harry Percy,
The next way to make King Henry'st and his highness curst thou that raised with his son,
The royal fruit-night to get aboard him.
What'st thou spoken like a prophet could never brook-fellow
What is left behind,
What dangers on the princely mast?
The royal graced with his leadenuate,
That every clouded gaoler meeting noses?
What'st thou shalt thou diadem upon my father'st thou not that is Dee;
Thou hast thou to beat Aumerle plants with eager cry thee;
What'st order ta'Anon desires access to get aboard!
What is Caius Marcius,
```

TODO:
* Add mixed precision training https://arxiv.org/abs/1710.03740
* Improve coherence with SentencePiece or BPE for tokenization
* Finetune with RLHF https://arxiv.org/pdf/2203.02155

Train multi-gpu
```
torchrun --nproc_per_node=4 ddp.py
python3 -m torch.distributed.run --nproc_per_node=4 train_multigpu.py
```
