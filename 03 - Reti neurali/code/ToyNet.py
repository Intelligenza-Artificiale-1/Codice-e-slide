import torch
import torch.nn as nn
from utils import plot

class ToyNet(nn.Module):
    def __init__(self):
        super(ToyNet, self).__init__()
        self.fc1 = nn.Linear(2, 1)

    def forward(self, x):
        x = self.fc1(x)
        return x

    def train_sample_perceptron(self, x, y):
        y_hat = torch.sign(self.forward(x))
        delta = (y - y_hat)/2
        self.fc1.weight.data += delta * x
        self.fc1.bias.data += delta

    def manual_sdg_train(self, x, y, epochs=40, lr=0.05, show_plot=True):
        # train using SGD optimizer and MSE loss
        losses = []
        for _ in range(epochs):
            loss = 0
            for i in range(len(x)):
                y_hat = self.forward(x[i])
                error = y_hat - y[i]  
                self.fc1.weight.data -= lr * error * x[i]
                self.fc1.bias.data -= lr * error * 1
                loss+=(error.detach()**2)
            losses.append(loss.item())
            if show_plot:
                plot(x, y, self, title="Manual SGD")
        return losses

    def train(self, x, y, epochs=40, lr=0.05, show_plot=True):
        # train using SGD optimizer
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        losses = []
        for _ in range(epochs):
            optimizer.zero_grad()
            y_hat = self.forward(x).flatten()
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.detach())
            if show_plot:
                plot(x, y, self, title="PyTorch SGD Optimizer")
        return losses

