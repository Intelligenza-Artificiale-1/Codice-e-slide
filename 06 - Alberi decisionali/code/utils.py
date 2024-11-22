import numpy as np
import os
import matplotlib.pyplot as plt

def classification_stats(y, yhat):
    y, yhat = np.array(y), np.array(yhat)
    accuracy = (yhat == y).sum() / len(y)
    precision = ((yhat == 1) * (y == 1)).sum() / (yhat == 1).sum()
    recall = ((yhat == 1) * (y == 1)).sum() / (y == 1).sum()
    f1 = 2 * precision * recall / (precision + recall)
    print( f"Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1: {f1:.3f}\n")

def load_data(file):
    dir_path = os.path.dirname(os.path.abspath(__file__))
    file= os.path.join(dir_path, file)
    data = np.loadtxt(file)
    x = data[:, :-1]
    y = data[:, -1]
    return x.tolist(), y.tolist()

def plot(X,y, model, title="", pause=False):
    # define bounds of the domain
    X, y = np.array(X), np.array(y)
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
    yhat = np.array(model(grid))
    # reshape the predictions back into a grid
    zz = yhat.reshape(xx.shape)
    # plot the grid of x, y and z values as a surface
    plt.contourf(xx, yy, zz, cmap='Paired')
    # create scatter plot for samples from each class
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Paired', edgecolors='black')
    plt.title(title)
    # show the plot
    plt.pause(2)
    if pause:
        plt.show()
    plt.clf()
