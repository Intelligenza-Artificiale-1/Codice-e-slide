import torch
from ToyNet import ToyNet, ToyNet2
import matplotlib.pyplot as plt
from utils import *

torch.manual_seed(42)

def main1():
    dataset = torch.Tensor([ [0.5, 35.7],
                    [14.0, 55.9],
                    [15.0, 58.2],
                    [28.0, 81.9],
                    [11.0, 56.3],
                    [8.0, 48.9],
                    [3.0, 33.9],
                    [-4.0, 21.8],
                    [6.0, 48.4],
                    [13.0, 60.4],
                    [21.0, 68.4]])


    model = ToyNet()
    X = dataset[:, 0].unsqueeze(1)
    y = dataset[:, 1].unsqueeze(1)
    plot(X, y, model, "Before training", temp=False)
    plot(X, y, model, "Before training", temp=True, clip=True)
    losses = []
    W = [model.fc1.weight.item()]
    B = [ model.fc1.bias.item()]
    for _ in range(20):
        l = model.fit(X, y, epochs=100, lr=0.001)
        plot(X, y, model, f"After training, loss={l[-1]}", temp=True, clip=True)
        losses += l
        W.append(model.fc1.weight.item())
        B.append(model.fc1.bias.item())
    plot(X, y, model, f"After training, loss={l[-1]}", temp=False, clip=True)
    plt.plot(losses)
    plt.show()

    print(f"Farenheit = Celsius * {model.fc1.weight.item():.2f} + {model.fc1.bias.item():.2f}")

    draw_mse_gradient_heatmap(0, 5, 0, 40, dataset)
    plt.plot(W, B, 'o-')
    plt.scatter([1.8], [36], c='g')
    plt.show()

def main2():
    dataset = torch.Tensor([[2, 1, 1],
                         [6, 0.5, 0],
                         [2.5, -1, 1],
                         [5, 0, 0],
                         [0, 0, 1],
                         [4, -1, 0],
                         [1, 0.5, 1],
                         [3, 1.5, 0]])
    model = ToyNet2()
    X = dataset[:, :2]
    y = dataset[:, 2].long()
    plot2(X, y, model, title="Before training", temp=False)
    losses = []
    for _ in range(15):
        l = model.fit(X, y, epochs=150, lr=0.01)
        plot2(X, y, model, title=f"After training, loss={l[-1]}", temp=True)
        losses += l
    plot2(X, y, model, title=f"After training, loss={l[-1]}", temp=False)
    plt.plot(losses)
    plt.show()

if __name__ == "__main__":
    main1()
