# fecot/client.py
from __future__ import annotations
import json
from typing import Dict, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import flwr as fl

from data import get_federated_partitions
from model import get_model
from utils import device_auto, train_one_epoch, evaluate_acc, softmax_conf

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, dataset: str, private_batch: int, unlabeled_per_client: int, lr: float, cuda: str):
        self.cid = cid
        (
            self.trainset,
            self.valset,
            self.unlabeled_set,
            self.num_classes,
        ) = get_federated_partitions(int(cid), dataset, unlabeled_per_client)

        self.device = device_auto(cuda)
        self.net = get_model(self.num_classes).to(self.device)

        self.trainloader = DataLoader(self.trainset, batch_size=private_batch, shuffle=True, num_workers=2)
        self.valloader = DataLoader(self.valset, batch_size=private_batch, shuffle=False, num_workers=2)

        self.lr = lr
        self.unlabeled_loader = DataLoader(self.unlabeled_set, batch_size=len(self.unlabeled_set), shuffle=False)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = {k: torch.tensor(v) for k, v in zip(self.net.state_dict().keys(), parameters)}
        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        epochs = int(config.get("local_epochs", 4))
        lr = float(config.get("lr", self.lr))

        opt = torch.optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        for _ in range(epochs):
            train_one_epoch(self.net, self.trainloader, opt, self.device)

        # Predict unlabeled pool and send predictions + confidences in metrics
        self.net.eval()
        with torch.no_grad():
            for xb, _ in self.unlabeled_loader:
                xb = xb.to(self.device)
                logits = self.net(xb)
                probs = torch.softmax(logits, dim=1)
                confs, preds = torch.max(probs, dim=1)
                u_preds = preds.cpu().numpy().tolist()
                u_confs = confs.cpu().numpy().tolist()

        metrics = {
            "val_acc": float(evaluate_acc(self.net, self.valloader, self.device)),
            "num_classes": self.num_classes,
            "u_preds": json.dumps(u_preds),
            "u_confs": json.dumps(u_confs),
        }
        return self.get_parameters({}), len(self.trainset), metrics

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, acc = evaluate_acc(self.net, self.valloader, self.device, return_loss=True)
        return float(loss), len(self.valset), {"val_acc": float(acc)}

def client_fn(cid: str, dataset: str, private_batch: int, unlabeled_per_client: int, cuda: str, lr: float):
    return FlowerClient(cid, dataset, private_batch, unlabeled_per_client, lr, cuda)
