import torch
import torch.nn as nn
import torch.nn.functional as F
import utils 

class Net(nn.Module):
    def __init__(self):
        super().__init__()

    def fit(self, x, y, lr=0.01, epochs=300, show=False, decay=True):
        losses = []
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        if decay:
            #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.999)
        for epoch in range(epochs):
            loss = loss_fn(self.forward(x).squeeze(), y)
            optimizer.zero_grad()
            loss.backward()
            losses.append(loss.item())  
            optimizer.step()
            if show :
                utils.plot(x, y, self, f"Epoch {epoch}, loss {loss.item():.2f}", pause=False)
            if decay:
                scheduler.step()
        return losses

class WideNet(Net):
    def __init__(self, hidden_size):
        super().__init__()
        hidden_size = max(1, hidden_size)
        self.fc1 = nn.Linear(1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x

class DeepNet(Net):
    def __init__(self, depth, hidden_size):
        super().__init__()
        hidden_size = max(1, hidden_size)
        depth = max(1, depth)
        self.activation = nn.ReLU()
        layers = [nn.Linear(1, hidden_size), self.activation]
        for _ in range(depth-2):
            layers += [nn.Linear(hidden_size, hidden_size), self.activation]
        self.layers = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.layers(x)
        x = self.output(x)
        return x