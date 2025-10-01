# data.py
from __future__ import annotations
from typing import Tuple
import torch
from torch.utils.data import Subset, random_split, Dataset
from torchvision import datasets, transforms

def _cifar10():
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465), (0.2023,0.1994,0.2010)),
    ])
    tr = datasets.CIFAR10("./data", train=True,  download=True, transform=tfm)
    te = datasets.CIFAR10("./data", train=False, download=True, transform=tfm)
    full = torch.utils.data.ConcatDataset([tr, te])
    num_classes = 10
    return full, num_classes

def get_partitions(cid: int, dataset: str, unlabeled_per_client: int):
    if dataset.lower() != "cifar10":
        raise ValueError("Only CIFAR-10 implemented in this minimal baseline.")
    full, num_classes = _cifar10()

    # Simple split into 5 equal parts (also works for 3 clients)
    C = 5
    N = len(full)
    part = N // C
    start, end = (cid % C) * part, min((cid % C + 1) * part, N)
    client_slice = Subset(full, range(start, end))

    # From this slice: U=20, the rest is labeled → train/val
    L = len(client_slice)
    u = min(unlabeled_per_client, 20)  # ensure length 20
    l = L - u
    val = max(1, int(0.1 * l))
    train = l - val

    labeled_subset, unlabeled_subset = random_split(client_slice, [l, u], generator=torch.Generator().manual_seed(123+cid))
    train_subset, val_subset = random_split(labeled_subset, [train, val], generator=torch.Generator().manual_seed(321+cid))

    class Unlabeled(Dataset):
        def __init__(self, subset): self.subset = subset
        def __len__(self): return len(self.subset)
        def __getitem__(self, i):
            x, _ = self.subset[i]
            return x, -1

    return train_subset, val_subset, Unlabeled(unlabeled_subset), num_classes
