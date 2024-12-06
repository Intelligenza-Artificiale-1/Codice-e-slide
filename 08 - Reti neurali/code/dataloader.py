import torch
import torch.nn as nn
import numpy as np
import os
from utils import *

torch.manual_seed(42)

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        super().__init__()
        dir_path = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(dir_path, 'dataset.dat')
        ds = np.loadtxt(file_path, delimiter=' ', dtype=np.float32)
        l = int(0.7 * len(ds))
        self.data = ds[:l] if train else ds[l:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx, :-1], int(self.data[idx, -1])

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 2)

    def forward(self, x):
        x = self.fc1(x)
        return x
    
    def fit(self, dataloader, epochs=100, lr=0.01):
        losses = []
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        for _ in range(epochs):
            epoch_loss = 0
            for x, y in dataloader:
                optimizer.zero_grad()
                output = self(x)
                loss = criterion(output, y)
                loss.backward()
                epoch_loss += loss.item()
                optimizer.step()
            average_loss = epoch_loss / len(dataloader)
            losses.append(average_loss)
        return losses





def main():
    trainset = MyDataset(train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, drop_last=False)

    testset = MyDataset(train=False)
    test_x, test_y = zip(*testset)
    model = Classifier()


    plot2(test_x, test_y, model, title="Before training", temp=False)
    losses = []
    for _ in range(15):
        l = model.fit(trainloader, epochs=50, lr=0.01)
        plot2(test_x, test_y, model, title=f"After training, loss={l[-1]}", temp=True)
        losses += l
    plot2(test_x, test_y, model, title=f"After training, loss={l[-1]}", temp=False)
    plt.plot(losses)
    plt.show()


if __name__ == "__main__":
    main()