import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

#set seed
np.random.seed(42)
torch.manual_seed(42)

class NeuralNet(nn.Module):
    def __init__(self):
        """Da completare"""
        pass

    def forward(self, x):
        """Da completare"""
        pass

    def fit(self, X_train, y_train, epochs=None, lr=None):
        """Da completare"""
        pass

def load_dataset():
    #loads dataset and puts it into torch tensors (test and train split)
    dir_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(dir_path, 'dataset.csv')
    data = np.loadtxt(file_path, delimiter=',')
    X, y = data[:, :-1], data[:, -1]
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()
    #split into test and train
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, X_test, y_train, y_test


def main():
    X_train, X_test, y_train, y_test = load_dataset()
    model = NeuralNet()
    losses = model.fit(X_train, y_train)
    plt.plot(losses)
    #check accuracy on test set
    with torch.no_grad():
        y_pred = model(X_test)
        correct = (y_pred.argmax(dim=1) == y_test).type(torch.FloatTensor)
        accuracy = correct.mean()
    plt.title(f'Accuracy: {accuracy.item():.3f}')
    plt.show()



if __name__ == '__main__':
    main()
