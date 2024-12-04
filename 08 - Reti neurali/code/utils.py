import numpy as np
import torch
import os
import matplotlib.pyplot as plt


def plot(X,y, model, title="", temp=True, clip=False):
    plt.scatter(X, y)
    x = np.linspace(min(X), max(X), 100)
    if clip:
        r = max(y) - min(y)
        plt.ylim((min(y)-0.05*r, max(y)+0.05*r))
    plt.plot(x, model(torch.Tensor(x)).detach().numpy(), color='red')
    plt.title(title)
    # show the plot
    if temp:
        plt.pause(1)
        plt.clf()
    else:
        plt.show()

def plot2(X,y, model, title="", temp=False):
    # define bounds of the domain
    with torch.no_grad():
        X, y = np.array(X), np.array(y)
        min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
        min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1
        # define the x and y scale
        x1grid = np.arange(min1, max1, (max1-min1)/1000)
        x2grid = np.arange(min2, max2, (max2-min2)/1000)
        # create all of the lines and rows of the grid
        xx, yy = np.meshgrid(x1grid, x2grid)
        # flatten each grid to a vector
        r1, r2 = xx.flatten(), yy.flatten()
        # horizontal stack vectors to create x1,x2 input for the model
        grid = torch.Tensor([[x1, x2] for x1, x2 in zip(r1,r2)])
        # make predictions for the grid
        yhat = np.argmax(model(grid),axis=1)
        # reshape the predictions back into a grid
        zz = yhat.reshape(xx.shape)
        # plot the grid of x, y and z values as a surface
        plt.contourf(xx, yy, zz, cmap='Paired')
        # create scatter plot for samples from each class
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Paired', edgecolors='black')
        plt.title(title)
        # show the plot
        if temp:
            plt.pause(1)
            plt.clf()
        else:
            plt.show()

def draw_mse_gradient_heatmap(x_min, x_max, y_min, y_max, dataset):
    def f(x,y, dataset):
        loss = 0
        return sum((x * s1 + y - s2) ** 2 for s1, s2 in dataset) / len(dataset)

    X = torch.linspace(x_min, x_max, 100)
    Y = torch.linspace(y_min, y_max, 100)
    Z = torch.Tensor([[f(x, y, dataset) for x in X] for y in Y]).T
    X, Y = torch.meshgrid(X, Y)
    plt.contourf(X, Y, Z, 50, cmap='RdGy')
    plt.colorbar()