import torch
import ToyNet
import matplotlib.pyplot as plt
from utils import *

DATA = "./dataset.dat"

def train_perceptron(x, y, net, show_plot=True):
    losses = []
    while True:
        # shuffle x and y
        indices = torch.randperm(len(x))
        x = x[indices]
        y = y[indices]

        loss = 0
        for i in range(len(x)):
            net.train_sample_perceptron(x[i], y[i])
            loss += (y[i] - net.forward(x[i])).detach()**2
        if show_plot:
            plot(x, y, net, title="Perceptron Algorithm")
        losses.append(loss.detach().item())
        y_hat = torch.sign(net.forward(x))


        if torch.all(y_hat.flatten() == y):
            break
    return losses

def main():
    # train using perceptron algorithm
    net = ToyNet.ToyNet()
    x, y = load_data(DATA)
    losses = train_perceptron(x, y, net, show_plot=True)
    plot(x, y, net, title="Perceptron Algorithm", temp=False)
    plt.plot(losses)
    plt.show()
    # print parameters
    print("Perceptron Algorithm")
    print("w: ", net.fc1.weight.data)
    print("b: ", net.fc1.bias.data)

    # manual SGD training
    net = ToyNet.ToyNet()
    losses = net.manual_sdg_train(x, y, epochs=50, lr=0.01, show_plot=True)
    plot(x, y, net, title="Manual SGD", temp=False)
    plt.plot(losses)
    plt.show()

    print("\nManual SGD")
    print("w: ", net.fc1.weight.data)
    print("b: ", net.fc1.bias.data)

    # train using SGD optimizer
    net = ToyNet.ToyNet()
    losses = net.train(x, y, epochs=40, lr=0.05, show_plot=True)
    plot(x, y, net, title="PyTorch SGD Optimizer", temp=False)
    plt.plot(losses)
    plt.show()

    print("\nPyTorch SGD Optimizer")
    print("w: ", net.fc1.weight.data)
    print("b: ", net.fc1.bias.data)

if __name__ == "__main__":
    main()
