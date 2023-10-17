import numpy as np
import torch
import matplotlib.pyplot as plt


def load_data(file):
    data = np.loadtxt(file)
    x = data[:, :-1]
    y = data[:, -1]
    return torch.tensor(x, dtype=torch.float32),\
            torch.tensor(y, dtype=torch.float32)


def plot(x,y, net, title="Linear classifier", temp=True):
    # plot data
    plt.scatter(x[:, 0], x[:, 1], c=y)
    # plot decision boundary
    w = net.fc1.weight.data
    b = net.fc1.bias.data
    x1 = np.linspace(min(x[:, 0]), max(x[:, 0]), 100)
    x2 = -(w[0, 0]*x1 + b[0])/w[0, 1]
    plt.plot(x1, x2)
    plt.ylim(min(x[:, 1])-1, max(x[:, 1])+1)
    plt.title(title)
    #plt.show(block=False)
    if temp:
        plt.pause(1)
        plt.clf()
    else:
        plt.show()


def plot2(X,y, model, title="", temp=True):
    # define bounds of the domain
    min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
    min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1
    # define the x and y scale
    x1grid = np.arange(min1, max1, 0.025)
    x2grid = np.arange(min2, max2, 0.025)
    # create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)
    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    # horizontal stack vectors to create x1,x2 input for the model
    grid = [[x1, x2] for x1, x2 in zip(r1,r2)]
    # make predictions for the grid
    yhat = model.forward(torch.tensor(grid, dtype=torch.float32)).detach().numpy()
    # reshape the predictions back into a grid
    zz = yhat.reshape(xx.shape)
    # plot the grid of x, y and z values as a surface
    plt.contourf(xx, yy, zz, cmap='viridis')
    # create scatter plot for samples from each class
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='black')
    plt.title(title)
    # show the plot
    if temp:
        plt.pause(2)
        plt.clf()
    else:
        plt.show()


