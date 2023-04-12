import sys
import torch
from .hyperparams import *
from .model import BigramLanguageModel

# Get the arguments passed to the script
path = sys.argv[1]

# Print the arguments
model, encode, decode = BigramLanguageModel.load(path)

while True:
    context = input("Message: ")
    enc_context = torch.tensor([encode(context)], dtype=torch.long)
    response = model.generate(enc_context, max_new_tokens=200)
    print(decode(response[0].tolist()))
