# fecot/data.py
from __future__ import annotations
from typing import Tuple
import torch
from torch.utils.data import Dataset, random_split, Subset
from torchvision import datasets, transforms

# You can add more datasets later (e.g., DomainNet) with the same interface.
def _cifar10():
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)),
    ])
    tr = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
    te = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)
    full = torch.utils.data.ConcatDataset([tr, te])
    num_classes = 10
    return full, num_classes

def get_federated_partitions(cid: int, dataset: str, unlabeled_per_client: int):
    if dataset.lower() == "cifar10":
        full, num_classes = _cifar10()
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Simple deterministic split per client id
    N = len(full)
    chunk = N // 5  # assumes 5 clients; simulation will reuse ranges if fewer/more
    start = (cid % 5) * chunk
    end = start + chunk
    client_slice = Subset(full, range(start, min(end, N)))

    # From client slice → split labeled train/val and a tiny unlabeled pool
    L = len(client_slice)
    u = min(unlabeled_per_client, L // 10 if L >= 10 else L)
    l = L - u
    val = int(0.1 * l)
    train = l - val

    labeled_subset, unlabeled_subset = random_split(client_slice, [l, u], generator=torch.Generator().manual_seed(123+cid))
    train_subset, val_subset = random_split(labeled_subset, [train, val], generator=torch.Generator().manual_seed(321+cid))

    # For unlabeled, we ignore labels by returning target=-1
    class Unlabeled(Dataset):
        def __init__(self, subset):
            self.subset = subset
        def __len__(self): return len(self.subset)
        def __getitem__(self, idx):
            x, _ = self.subset[idx]
            return x, -1

    return train_subset, val_subset, Unlabeled(unlabeled_subset), num_classes
