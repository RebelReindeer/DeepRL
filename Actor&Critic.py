import torch
import torch.nn as nn
import torch.nn.functional as F

policy = nn.Sequential(
    nn.Linear(2, 128),
    F.relu(),
    nn.Linear(128, 128),
    F.relu(),
    nn.Linear(128, 2),
    F.softmax(dim = -1)
)


