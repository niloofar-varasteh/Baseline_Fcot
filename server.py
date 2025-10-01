# fecot/server.py
import argparse
import flwr as fl
from strategy import FecotStrategy

def parse_args():
    p = argparse.ArgumentParser("FCoT Server")
    p.add_argument("--address", type=str, default="[::]:8080")
    p.add_argument("--rounds", type=int, default=100)
    p.add_argument("--num_clients", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--local_epochs", type=int, default=4)
    p.add_argument("--dataset", type=str, default="cifar10")
    p.add_argument("--private_batch", type=int, default=32)
    p.add_argument("--unlabeled_per_client", type=int, default=20)
    return p.parse_args()

def main():
    args = parse_args()
    strategy = FecotStrategy(
        fraction_fit=1.0,
        min_fit_clients=args.num_clients,
        min_available_clients=args.num_clients,
        on_round_config_fn=lambda rnd: {
            "lr": args.lr,
            "local_epochs": args.local_epochs,
            "dataset": args.dataset,
            "private_batch": args.private_batch,
            "unlabeled_per_client": args.unlabeled_per_client,
            "cuda": "auto",
        },
    )

    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
