import numpy as np
import torch
import matplotlib.pyplot as plt


def load_data(file):
    data = np.loadtxt(file)
    x = data[:, :-1]
    y = data[:, -1]
    return torch.tensor(x, dtype=torch.float32),\
            torch.tensor(y, dtype=torch.float32)


def plot(X,y,model, title="", pause=False):
    plt.clf()
    xmin = min(X)
    xmax = max(X)
    x = np.linspace(xmin, xmax, 100)
    plt.scatter(X, y)
    plt.plot(x, model.forward(torch.tensor(x, dtype=torch.float32)).detach().numpy(), color="red")
    plt.title(title)
    plt.pause(0.25)
    if pause:
        plt.show()
