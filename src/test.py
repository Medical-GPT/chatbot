import sys
from utils import load_data, load_model, get_tokens, get_token_coders
import torch
from hyperparams import *

# Get the arguments passed to the script
path = sys.argv[1]

# Print the arguments
data = load_data("input.txt")

tokens = get_tokens(data)
encode, decode = get_token_coders(tokens)
vocab_size = len(tokens)

model = load_model(path, N_EMBD, vocab_size, BLOCK_SIZE, N_HEAD, N_LAYER, DROPOUT)


while True:
    context = input("Message: ")
    enc_context = torch.tensor([encode(context)], dtype=torch.long)
    response = model.generate(enc_context, max_new_tokens=200)
    print(decode(response[0].tolist()))
