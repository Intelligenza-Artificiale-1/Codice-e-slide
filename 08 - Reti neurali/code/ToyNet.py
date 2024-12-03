import torch.nn as nn
import torch

class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x

    def fit(self, x, y, epochs=100, lr=0.001):
        losses = []
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        for _ in range(epochs):
            optimizer.zero_grad()
            output = self(x)
            loss = criterion(output, y)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
        return losses

class ToyNet2(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        return x
    
    def fit(self, x, y, epochs=100, lr=0.01):
        losses = []
        return losses
