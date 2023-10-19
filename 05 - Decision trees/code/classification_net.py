import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 50)
        self.linear2 = nn.Linear(50, 50)
        self.linear3 = nn.Linear(50, 2)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.activation(self.linear(x))
        x = self.activation(self.linear2(x))
        x = self.linear3(x)
        x = self.softmax(x)
        return x

    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float)
        with torch.no_grad():
            return self.forward(x).argmax(dim=1).numpy()


    def train_loop(self, X, y, epochs=900, lr=0.1):
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = loss_fn(y_pred, y)
            if epoch % 100 == 0:
                print(f"Epoch: {epoch}, Loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
