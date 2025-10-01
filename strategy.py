# fecot/strategy.py
from __future__ import annotations
import json
import numpy as np
from typing import Dict, List, Optional, Tuple
import flwr as fl
from flwr.common import EvaluateRes, FitRes, Metrics

CONF_THRESHOLD = 0.7  # confidence gate for a valid vote

def majority_vote(preds: np.ndarray, confs: np.ndarray, conf_thresh: float = CONF_THRESHOLD) -> np.ndarray:
    # preds, confs: shape [num_clients, U]
    num_clients, U = preds.shape
    maj = np.zeros(U, dtype=int)
    for i in range(U):
        # consider only votes above confidence threshold
        valid_idx = confs[:, i] >= conf_thresh
        votes = preds[valid_idx, i]
        if votes.size == 0:
            # no confident votes → fall back to plain majority over all
            votes = preds[:, i]
        # argmax over counts
        labels, counts = np.unique(votes, return_counts=True)
        maj[i] = labels[np.argmax(counts)]
    return maj

class FecotStrategy(fl.server.strategy.FedAvg):
    """FedAvg + server-side co-training consensus logging."""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ):
        # Call FedAvg to aggregate weights
        agg_parameters, agg_metrics = super().aggregate_fit(server_round, results, failures)

        # Collect per-client predictions on the unlabeled pool (sent via fit metrics)
        client_preds, client_confs = [], []
        for _, fit_res in results:
            m: Metrics = fit_res.metrics or {}
            if "u_preds" in m and "u_confs" in m:
                client_preds.append(np.array(json.loads(m["u_preds"]), dtype=int))
                client_confs.append(np.array(json.loads(m["u_confs"]), dtype=float))

        if client_preds:
            P = np.stack(client_preds)      # [C, U]
            C = np.stack(client_confs)      # [C, U]
            maj = majority_vote(P, C, CONF_THRESHOLD)

            # Debug prints per round
            print(f"\n=== Round {server_round} FCoT Debug ===")
            for ci, (p, c) in enumerate(zip(P, C), start=1):
                print(f"[Client {ci}] preds={p.tolist()} confs={[round(x,3) for x in c.tolist()]}")
            print(f"[Majority] {maj.tolist()}\n")

            # Save final consensus on last round
            if self._num_rounds is not None and server_round == self._num_rounds:
                with open(f"fecot_consensus_round{server_round}.json", "w", encoding="utf-8") as f:
                    json.dump({"majority": maj.tolist()}, f, ensure_ascii=False, indent=2)

        return agg_parameters, agg_metrics
