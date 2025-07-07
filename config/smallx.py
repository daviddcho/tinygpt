from types import SimpleNamespace

cfg = SimpleNamespace(
  N_EPOCHS = 200,
  D_MODEL = 768,
  N_HEADS = 12,
  N_LAYERS = 10,
  D_FF = 3072,
  MAX_SEQ_LEN = 1024,
  BS = 32, # 8 per gpu
)
