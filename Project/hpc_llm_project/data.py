import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

class SyntheticTokenDataset(Dataset):
    def __init__(self, num_samples=10000, seq_len=128, vocab_size=5000, seed=1234):
        g = torch.Generator().manual_seed(seed)
        self.data = torch.randint(
            low=0,
            high=vocab_size,
            size=(num_samples, seq_len + 1),
            generator=g,
            dtype=torch.long,
        )

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx, :-1]
        y = self.data[idx, 1:]
        return x, y

def build_loader(dataset, batch_size, distributed=False, rank=0, world_size=1):
    sampler = None
    if distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    return loader, sampler