import torch
import torch.nn as nn
import torch.nn.functional as F
import utils 

class Net(nn.Module):
    def __init__(self):
        super().__init__()

    def train(self, x, y, lr=0.01, epochs=300, show=False):
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        for epoch in range(epochs):
            loss = loss_fn(self.forward(x).squeeze(), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if show and epoch % 20 == 0:
                utils.plot(x, y, self, f"Epoch {epoch}, loss {loss.item():.2f}", pause=False)
        return loss.item()

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
        self.fc1 = nn.Linear(1, hidden_size)
        for i in range(2,depth):
            setattr(self, f'fc{i}', nn.Linear(hidden_size, hidden_size))
        setattr(self, f'fc{depth}', nn.Linear(hidden_size, 1))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        for i in range(2, len(self._modules)):
            x = F.relu(getattr(self, f'fc{i}')(x))
        x = getattr(self, f'fc{len(self._modules)}')(x)
        return x

class TaylorNet(Net):
    def __init__(self, degree):
        super().__init__()
        self.degree = degree
        self.fc1 = nn.Linear(degree, 1)

    def forward(self, x):
        x = torch.cat([(x**i-self.mean[i-1])/self.std[i-1] \
                for i in range(1, self.degree+1)], dim=1)
        x = self.fc1(x)
        return x

    def train(self, x, y, lr=0.01, epochs=300, show=False):
        self.set_scaler(x)
        return super().train(x, y, lr, epochs, show)

    def set_scaler(self, x):
        self.mean = []
        self.std = []
        for i in range(1, self.degree+1):
            self.mean.append(torch.mean(x**i))
            self.std.append(torch.std(x**i))

class FourierNet(Net):
    def __init__(self, degree):
        super().__init__()
        self.degree = degree
        self.fc1 = nn.Linear(2*degree, 1)

    def forward(self, x):
        x = (x-self.bias)*self.scale
        x = torch.cat([torch.sin(x*i) for i in range(1, self.degree+1)] +\
                [torch.cos(x*i) for i in range(1, self.degree+1)], dim=1)
        x = self.fc1(x)
        return x

    def set_scaler(self, x):
        min_x = torch.min(x)
        max_x = torch.max(x)
        #make the wole domain [-pi, pi]
        self.bias = -min_x
        self.scale = torch.pi/(max_x-min_x)
        

    def train(self, x, y, lr=0.01, epochs=300, show=False):
        self.set_scaler(x)
        return super().train(x, y, lr, epochs, show)

class CoulombNet(Net):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 10)
        self.fc5 = nn.Linear(10, 1)
        self.activation = nn.ReLU()

    def set_scaler(self, x,y):
        self.mean = []
        self.std = []
        for i in range(3):
            self.mean.append(torch.mean(x[:,i]))
            self.std.append(torch.std(x[:,i]))
        self.miny = torch.min(y)
        self.maxy = torch.max(y)

    def train(self, x, y, lr=0.01, epochs=300):
        self.set_scaler(x,y)
        return super().train(x, y, lr, epochs)

    def forward(self, x):
        x = torch.stack([(x[:,i]-self.mean[i])/self.std[i] for i in range(3)], dim=1)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = nn.Sigmoid()(self.fc5(x))
        x = x*(self.maxy-self.miny)+self.miny
        return x
