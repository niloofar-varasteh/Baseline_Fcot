# run.py
import argparse
import flwr as fl
from flwr.common import Context
from strategy import FecotStrategy
from client import client_fn


def parse_args():
    p = argparse.ArgumentParser("FCoT baseline (Flower simulation)")
    p.add_argument("--num_clients", type=int, default=5)            # 5 (fallback 3)
    p.add_argument("--rounds", type=int, default=100)               # total rounds
    p.add_argument("--local_epochs", type=int, default=4)           # communicate every 4 local epochs
    p.add_argument("--private_batch", type=int, default=32)         # local labeled batch size
    p.add_argument("--unlabeled_per_client", type=int, default=20)  # |P_i| = 20 (global batch = 20)
    p.add_argument("--dataset", type=str, default="cifar10")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--cuda", type=str, default="auto", choices=["auto", "on", "off"])
    return p.parse_args()


def main():
    args = parse_args()

    # Per-round config passed to clients (Flower 1.x uses on_fit_config_fn)
    def fit_cfg_fn(rnd: int):
        return {
            "lr": args.lr,
            "local_epochs": args.local_epochs,
            "dataset": args.dataset,
            "private_batch": args.private_batch,
            "unlabeled_per_client": args.unlabeled_per_client,
            "cuda": args.cuda,
        }

    strategy = FecotStrategy(
        fraction_fit=1.0,
        min_fit_clients=args.num_clients,
        min_available_clients=args.num_clients,
        on_fit_config_fn=fit_cfg_fn,
    )

    # New-style client_fn signature for Flower >=1.22
    def sim_client_fn(context: Context):
        cid = str(context.node_id)  # use node_id as client id in simulation
        return client_fn(
            cid=cid,
            dataset=args.dataset,
            private_batch=args.private_batch,
            unlabeled_per_client=args.unlabeled_per_client,
            cuda=args.cuda,
            lr=args.lr,
        )

    fl.simulation.start_simulation(
        client_fn=sim_client_fn,
        num_clients=args.num_clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
