import numpy as np
import torch
import matplotlib.pyplot as plt
import os


def load_data(file):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    file= os.path.join(dir_path, file)
    data = np.loadtxt(file)
    x = data[:, :-1]
    y = data[:, -1]
    return torch.tensor(x, dtype=torch.float32),\
            torch.tensor(y, dtype=torch.float32)


def plot(X,y,model, title="", pause=False):
    plt.clf()
    xmin, xmax = min(X), max(X)
    ymin, ymax = min(y), max(y)
    x = np.linspace(xmin, xmax, 100)
    plt.ylim(ymin, ymax)
    plt.scatter(X, y)
    plt.plot(x, model.forward(torch.tensor(x, dtype=torch.float32)).detach().numpy(), color="red")
    plt.title(title)
    plt.pause(0.0125)
    if pause:
        plt.show()
