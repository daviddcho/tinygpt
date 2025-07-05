#!/usr/bin/env python3
import torch
import torch.nn.functional as F
import tiktoken
from model import Transformer

tokenizer = tiktoken.get_encoding("gpt2")

def chat(model_path, max_length=200, temperature=0.5):
  device = torch.device(
    'cuda' if torch.cuda.is_available() else 
    'mps' if torch.mps.is_available() else 
    'cpu'
  )
  print(f"Loading model on {device}...")
  
  checkpoint = torch.load(model_path, map_location=device, weights_only=False)
  state_dict = checkpoint['model_state_dict']
  vocab_size = checkpoint['vocab_size']

  d_model = state_dict['embedding.weight'].shape[1]
  n_layers = len(set([k.split('.')[1] for k in state_dict.keys() if k.startswith('transformer_blocks.') and '.' in k]))
  n_heads = d_model // (state_dict['transformer_blocks.0.attention.w_q.weight'].shape[0] // d_model)
  d_ff = state_dict['transformer_blocks.0.feed_forward.linear1.weight'].shape[0]
  max_seq_len = state_dict['pos_encoding.pe'].shape[1]
  
  print(f"Inferred architecture: d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}, d_ff={d_ff}, max_seq_len={max_seq_len}")

  model = Transformer(
    vocab_size=vocab_size,
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers,
    d_ff=d_ff,
    max_seq_len=max_seq_len,
  ).to(device)

  model.load_state_dict(state_dict)
  model.eval()
  
  print("Model loaded successfully! Type 'quit' to exit.")
  
  while True:
    user_input = input("\nYou: ").strip()
    if user_input.lower() == 'quit':
      break
      
    if not user_input:
      continue
      
    try:
      x = tokenizer.encode(user_input)[0]
      context = torch.tensor([[x]], dtype=torch.long).to(device)
      
      generated_text = generate_text(model, context, max_length, temperature)
      print(f"Bot: {generated_text}")
      
    except Exception as e:
      print(f"Error generating response: {e}")

def generate_text(model, context, max_length=200, temperature=0.8):
  model.eval()
  generated = []
  with torch.no_grad():
    for _ in range(max_length):
      if context.size(1) > model.max_seq_len:
        context = context[:, -model.max_seq_len:]
        
      logits = model(context)
      logits = logits[0, -1, :]
      
      probs = F.softmax(logits / temperature, dim=-1)
      next_token = torch.multinomial(probs, 1)
      
      generated.append(next_token.item())
      context = torch.cat([context, next_token.unsqueeze(0)], dim=1)
  
  return tokenizer.decode(generated)

if __name__ == "__main__":
  import sys
  model_path = sys.argv[2] if len(sys.argv) > 2 else 'weights/transformer_enwik8.pth'
  chat(model_path)
