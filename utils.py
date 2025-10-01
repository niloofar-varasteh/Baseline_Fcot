# fecot/utils.py
from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from typing import Tuple

def device_auto(cuda_flag: str):
    if cuda_flag == "on":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cuda_flag == "off":
        return torch.device("cpu")
    # auto
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, loader: DataLoader, opt, device):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = torch.nn.functional.cross_entropy(logits, yb)
        loss.backward()
        opt.step()

def evaluate_acc(model, loader: DataLoader, device, return_loss=False):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = torch.nn.functional.cross_entropy(logits, yb)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
            loss_sum += loss.item() * yb.size(0)
    acc = correct / max(total, 1)
    if return_loss:
        return loss_sum / max(total, 1), acc
    return acc

def softmax_conf(logits):
    probs = torch.softmax(logits, dim=1)
    confs, preds = torch.max(probs, dim=1)
    return preds, confs
