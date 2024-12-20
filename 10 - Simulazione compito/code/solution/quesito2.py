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
        #super(NeuralNet, self
        super().__init__()
        width=64
        self.model = nn.Sequential(
            nn.Linear(30, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 2)
        )

    def forward(self, x):
        return self.model(x)

    def fit(self, X_train, y_train, epochs=200, lr=0.1):
        losses = []
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            y_pred = self(X_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        return losses

def load_dataset():
    #loads dataset and puts it into torch tensors (test and train split)
    dir_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(dir_path, 'breast_cancer.csv')
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
    plt.show()
    #check accuracy on test set
    with torch.no_grad():
        y_pred = model(X_test)
        correct = (y_pred.argmax(dim=1) == y_test).type(torch.FloatTensor)
        accuracy = correct.mean()
        print('Accuracy: ', accuracy.item())

if __name__ == '__main__':
    main()
