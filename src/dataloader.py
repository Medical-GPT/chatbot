import torch


class DataLoader:
    def __init__(self, data, device, block_size, batch_size, split=0.9):
        self.device = device
        self.block_size = block_size
        self.batch_size = batch_size

        self.data = torch.tensor(data, dtype=torch.long)

        n = int(split * len(self.data))

        self.train_data = self.data[:n]
        self.validation_data = self.data[n:]

    def get_batch(self, split):
        data = self.train_data if split == "train" else self.validation_data
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y
