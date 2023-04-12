import time
import torch
from encoder import Encoder
from model import BigramLanguageModel
from utils import estimate_loss, save_model
from dataloader import DataLoader
from hyperparams import *
from constants import CHECKPOINT_DIR, PRETRAINED_DIR, INPUT_DIR, ENCODER_ENCTEXT

torch.manual_seed(1337)

encoder = Encoder(INPUT_DIR, encode_text=True)

VOCAB_SIZE = encoder.get_vocab_size()

device = "cuda" if torch.cuda.is_available() else "cpu"

data_loader = DataLoader(ENCODER_ENCTEXT, device, BLOCK_SIZE, BATCH_SIZE)
print("Loaded data loader")

model = BigramLanguageModel(
    N_EMBD,
    VOCAB_SIZE,
    BLOCK_SIZE,
    N_HEAD,
    N_LAYER,
    DROPOUT,
).to(device)

print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

start = time.time()
for iter in range(MAX_ITERS):
    # every once in a while evaluate the loss on train and val sets
    if iter % EVAL_INTERVAL == 0 or iter == MAX_ITERS - 1:
        losses = estimate_loss(model, data_loader, EVAL_ITERS)
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}. Took {time.time() - start:.2f} seconds"
        )
        start = time.time()
        save_model(model, CHECKPOINT_DIR / f"ch_{iter}_{losses['val']:.4f}.pt")
    # sample a batch of data
    xb, yb = data_loader.get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


# save the model
save_model(model, PRETRAINED_DIR / "pretrained.pt")

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(encoder.decode(model.generate(context, max_new_tokens=2000)[0].tolist()))
