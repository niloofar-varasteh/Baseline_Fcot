# fecot/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SmallCifarCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128*4*4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 16x16
        x = self.pool(F.relu(self.conv2(x)))   # 8x8
        x = self.pool(F.relu(self.conv3(x)))   # 4x4
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def get_model(num_classes: int):
    return SmallCifarCNN(num_classes)
