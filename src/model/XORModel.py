import torch
from torch import nn

# Define model
class XORNN(nn.Module):
    def __init__(self):
        super().__init__()
        #self.flatten = nn.Flatten()
        self.linear_stack = nn.Sequential(
            nn.Linear(2, 2),
            #nn.Sigmoid(),
            nn.Tanh(),
            nn.Linear(2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.linear_stack(x)
        return logits