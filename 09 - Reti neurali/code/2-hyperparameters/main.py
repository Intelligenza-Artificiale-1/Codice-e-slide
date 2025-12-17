import nets
import torch.nn as nn
import utils
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

#set random seed
torch.manual_seed(1)

def widenet(dataset, width=10, lr=0.1, epochs=300):
    x, y = utils.load_data(dataset)
    x_train, y_train = x[:len(x)//2], y[:len(y)//2]
    x_test, y_test = x[len(x)//2:], y[len(y)//2:]
    net = nets.WideNet(width)
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    train_losses = net.fit(x_train, y_train, lr=lr, epochs=epochs, show=True)
    val_loss = nn.MSELoss()(net(x_test).squeeze(), y_test)
    utils.plot(x_test, y_test, net, f"Hidden size {width}, train loss {train_losses[-1]:.2f}, val loss {val_loss:.2f}, params {pytorch_total_params}", pause=True)
    plt.plot(train_losses)
    plt.show()

def deepnet(dataset, depth=3, lr=0.1, epochs=300):
    x, y = utils.load_data(dataset)
    x_train, y_train = x[:len(x)//2], y[:len(y)//2]
    x_test, y_test = x[len(x)//2:], y[len(y)//2:]
    net = nets.DeepNet(depth, 10)
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    train_losses = net.fit(x_train, y_train, lr=lr, epochs=epochs, show=True)
    val_loss = nn.MSELoss()(net(x_test).squeeze(), y_test)
    utils.plot(x_test, y_test, net, f"Depth {depth}, train loss {train_losses[-1]:.2f}, val loss {val_loss:.2f}, params {pytorch_total_params}", pause=True)
    plt.plot(train_losses)
    plt.show()

if __name__ == "__main__":
    dataset = "dataset1.dat"
    widenet(dataset, lr=0.01, width=100, epochs=600)
    #deepnet(dataset, lr=0.25, depth=7, epochs=800)
