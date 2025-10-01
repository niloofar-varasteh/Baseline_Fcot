# model.py
import torch.nn as nn
import torch.nn.functional as F

class SmallCifarCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.c3 = nn.Conv2d(64, 128, 3, padding=1)
        self.p  = nn.MaxPool2d(2,2)
        self.f1 = nn.Linear(128*4*4, 256)
        self.f2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.p(F.relu(self.c1(x)))  # 16x16
        x = self.p(F.relu(self.c2(x)))  # 8x8
        x = self.p(F.relu(self.c3(x)))  # 4x4
        x = x.view(x.size(0), -1)
        x = F.relu(self.f1(x))
        return self.f2(x)

def get_model(num_classes: int):
    return SmallCifarCNN(num_classes)
