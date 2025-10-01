# strategy.py
from __future__ import annotations
import json
from typing import List, Tuple
import numpy as np
import flwr as fl
from flwr.common import FitRes, Metrics

def majority_vote(pred_matrix: np.ndarray) -> np.ndarray:
    # pred_matrix: [C, U]  -> output: [U]
    C, U = pred_matrix.shape
    out = np.zeros(U, dtype=int)
    for i in range(U):
        labels, counts = np.unique(pred_matrix[:, i], return_counts=True)
        out[i] = labels[np.argmax(counts)]
    return out

class FecotStrategy(fl.server.strategy.FedAvg):
    """FedAvg + log predictions of U and perform majority voting on the server."""

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ):
        # Aggregate weights same as FedAvg
        agg_params, agg_metrics = super().aggregate_fit(server_round, results, failures)

        # Collect predictions of U from client metrics
        preds = []
        for _, fit_res in results:
            m: Metrics = fit_res.metrics or {}
            if "u_preds" in m:
                preds.append(np.array(json.loads(m["u_preds"]), dtype=int))

        if preds:
            P = np.stack(preds)             # [C, U=20]
            maj = majority_vote(P)          # [U=20]

            # Debug log (exactly what you requested)
            print(f"\n=== Round {server_round} FCoT Debug ===")
            for ci, p in enumerate(P, start=1):
                print(f"[Client {ci}] preds={p.tolist()}")
            print(f"[Majority] {maj.tolist()}\n")

            # On the final round, save the final output
            if getattr(self, "_num_rounds", None) == server_round:
                with open(f"fecot_consensus_round{server_round}.json", "w", encoding="utf-8") as f:
                    json.dump({"majority": maj.tolist()}, f, ensure_ascii=False, indent=2)

        return agg_params, agg_metrics
