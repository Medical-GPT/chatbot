import torch
from model import BigramLanguageModel


@torch.no_grad()
def estimate_loss(model, data_loader, eval_iters):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = data_loader.get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def save_model(model, path):
    torch.save(model.state_dict(), path)


def load_model(path, n_embd, vocab_size, block_size, n_head, n_layer, dropout):
    model = BigramLanguageModel(
        n_embd,
        vocab_size,
        block_size,
        n_head,
        n_layer,
        dropout,
    )
    model.load_state_dict(torch.load(path))
    return model


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text
