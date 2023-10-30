import numpy as np
import torch
import matplotlib.pyplot as plt


def confusion_matrix(y, yhat):
    """Plot the confusion matrix for a classification problem
    :y: ground truth
    :yhat: predictions
    """
    classes = np.unique(y)
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes))
    for i, c in enumerate(classes):
        for j, c2 in enumerate(classes):
            cm[i, j] = ((yhat == c2) * (y == c)).sum()
            # add labels
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.imshow(cm)
    plt.xticks(range(n_classes), classes)
    plt.yticks(range(n_classes), classes)
    plt.xlabel("Predicted")
    plt.ylabel("Ground truth")
    plt.show()
    
def plot_2d(X, f1=0, f2=1):
    """Plot a 2D representation of a dataset
    :X: dataset
    :f1: index of first feature
    :f2: index of second feature
    """
    plt.scatter(X[:, f1], X[:, f2], c=X[:, -1], edgecolors='black', cmap='viridis')
    plt.show()

def plot_3d(X, labels):
    """Plot a 3D representation of a dataset
    :X: dataset
    :labels: labels of the dataset
    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=labels, cmap='viridis')
    plt.show()

def plot_mse(X, error, labels, f1=0, f2=1):
    X = X.detach().numpy()
    error = error.detach().numpy()
    #show error levels
    plt.tricontourf(X[:, f1], X[:, f2], error)
    #show colorbar
    plt.colorbar()
    #show points
    plt.scatter(X[:, f1], X[:, f2], c=labels, edgecolors='black', cmap='viridis')
    plt.show()


