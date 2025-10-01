# FCoT Baseline (Flower)

Federated Co-Training baseline for Flower simulation.

## Key settings
- Clients: 5 (fallback 3)
- Unlabeled per client: 20 (global batch=20)
- Private batch: 32
- Local epochs: 4 (communication after 4 epochs)
- Rounds: 100
- Dataset: CIFAR-10 (auto `num_classes=10`)
- LR: param

## Run
```bash
pip install flwr==1.5.0 torch torchvision
python -m fecot.run --num_clients 5 --rounds 100 --local_epochs 4 --lr 0.01 --cuda auto
