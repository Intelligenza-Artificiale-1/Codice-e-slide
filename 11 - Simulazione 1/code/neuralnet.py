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
        #super(NeuralNet, self
        super().__init__()
        self.fc1 = nn.Linear(30, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128,64)
        self.fc4 = nn.Linear(64, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.softmax(self.fc4(x))

    def train(model, X_train, y_train, epochs=400, lr=0.2):
        losses = []
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = model(X_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses

def main():
    X_train, X_test, y_train, y_test = load_dataset()
    model = NeuralNet()
    losses = NeuralNet.train(model, X_train, y_train)
    plt.plot(losses)
    plt.show()
    #check accuracy on test set
    with torch.no_grad():
        y_pred = model(X_test)
        correct = (y_pred.argmax(dim=1) == y_test).type(torch.FloatTensor)
        accuracy = correct.mean()
        print('Accuracy: ', accuracy.item())

if __name__ == '__main__':
    main()
