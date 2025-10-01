# client.py
from __future__ import annotations
import json
import torch
import flwr as fl
from torch.utils.data import DataLoader
from data import get_partitions
from model import get_model

def device_auto(flag: str):
    if flag == "on":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if flag == "off":
        return torch.device("cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, loader, opt, device):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(xb), yb)
        loss.backward()
        opt.step()

def eval_acc(model, loader, device):
    model.eval()
    tot, ok = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            ok += (pred == yb).sum().item()
            tot += yb.size(0)
    return ok / max(1, tot)

class FecotClient(fl.client.NumPyClient):
    def __init__(self, cid: str, dataset: str, private_batch: int, unlabeled_per_client: int, cuda: str, lr: float):
        self.cid = int(cid)
        (self.trainset, self.valset, self.unlabeled, self.num_classes) = get_partitions(self.cid, dataset, unlabeled_per_client)
        self.device = device_auto(cuda)
        self.net = get_model(self.num_classes).to(self.device)
        self.trainloader = DataLoader(self.trainset, batch_size=private_batch, shuffle=True, num_workers=2)
        self.valloader   = DataLoader(self.valset,   batch_size=private_batch, shuffle=False, num_workers=2)
        # U is exactly 20 and fed in one go (global batch = 20)
        self.uloader     = DataLoader(self.unlabeled, batch_size=len(self.unlabeled), shuffle=False)
        self.lr = lr

    # --- required signatures for current Flower versions ---
    def get_parameters(self, config):
        return [p.detach().cpu().numpy() for _, p in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        sd = {k: torch.tensor(v) for k, v in zip(self.net.state_dict().keys(), parameters)}
        self.net.load_state_dict(sd, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        epochs = int(config.get("local_epochs", 4))
        lr     = float(config.get("lr", self.lr))

        opt = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        for _ in range(epochs):
            train_one_epoch(self.net, self.trainloader, opt, self.device)

        # predict on U=20
        self.net.eval()
        with torch.no_grad():
            xb, _ = next(iter(self.uloader))
            xb = xb.to(self.device)
            preds = self.net(xb).argmax(1).cpu().tolist()  # length 20

        metrics = {
            "val_acc": float(eval_acc(self.net, self.valloader, self.device)),
            "num_classes": self.num_classes,
            "u_preds": json.dumps(preds),
        }
        return self.get_parameters({}), len(self.trainset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        acc = eval_acc(self.net, self.valloader, self.device)
        return float(0.0), len(self.valset), {"val_acc": float(acc)}

def client_fn(cid: str, dataset: str, private_batch: int, unlabeled_per_client: int, cuda: str, lr: float):
    # return a Client (not NumPyClient) to avoid deprecation warning
    return FecotClient(cid, dataset, private_batch, unlabeled_per_client, cuda, lr).to_client()
