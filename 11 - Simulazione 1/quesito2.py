from sklearn.datasets import load_breast_cancer
import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

#set seed
np.random.seed(42)
torch.manual_seed(42)

def dataset_to_file():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(dir_path, 'breast_cancer.csv')
    data = load_breast_cancer()
    #random order
    indices = np.random.permutation(len(data.data))
    X = data.data[indices]
    # scale data
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    y = data.target[indices]
    with open(file_path, 'w') as f:
        for i in range(len(y)):
            f.write(','.join([str(x) for x in X[i]]) + ',' + str(y[i]) + '\n')

def load_dataset():
    #loads dataset and puts it into torch tensors (test and train split)
    dir_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(dir_path, 'breast_cancer.csv')
    if not os.path.exists(file_path):
        dataset_to_file()
    data = np.loadtxt(file_path, delimiter=',')
    X, y = data[:, :-1], data[:, -1]
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()
    #split into test and train
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    return X_train, X_test, y_train, y_test


class NeuralNet(nn.Module):
    def __init__(self):
        """Da completare"""
        pass

    def forward(self, x):
        """Da completare"""
        pass

    def train(model, X_train, y_train, epochs=None, lr=None):
        """Da completare"""
        pass

def main():
    X_train, X_test, y_train, y_test = load_dataset()
    model = NeuralNet()
    #Determinare numero di epoche e learning rate
    losses = NeuralNet.train(model, X_train, y_train, epochs=0, lr=0)
    plt.plot(losses)
    plt.show()
    with torch.no_grad():
        y_pred = model(X_test)
        correct = (y_pred.argmax(dim=1) == y_test).type(torch.FloatTensor)
        accuracy = correct.mean()
        print('Accuracy: ', accuracy.item())

if __name__ == '__main__':
    main()
