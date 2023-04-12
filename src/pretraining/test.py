import sys
from utils import load_data, load_model
import torch
from hyperparams import *
from encoder import Encoder

# Get the arguments passed to the script
path = sys.argv[1]

# Print the arguments
data = load_data("input.txt")

encoder = Encoder(None, load_tokens=True, encode_text=False)

vocab_size = encoder.get_vocab_size()

model = load_model(path, N_EMBD, vocab_size, BLOCK_SIZE, N_HEAD, N_LAYER, DROPOUT)

while True:
    context = input("Message: ")
    enc_context = torch.tensor([encoder.encode(context)], dtype=torch.long)
    response = model.generate(enc_context, max_new_tokens=200)
    print(encoder.decode(response[0].tolist()))
