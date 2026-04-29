import torch
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.net(x)

model = SimpleMLP()

print(model.state_dict().keys())
for name, param in model.state_dict().items():
    print(name, param.shape)