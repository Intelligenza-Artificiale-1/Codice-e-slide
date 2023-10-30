import torch
import torch.nn as nn

class PlainAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(5,3)
        self.decoder = nn.Linear(3,5)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.activation(x)
        x = self.decoder(x)
        return x

    def train_autoencoder(self, x, lr=0.01, epochs=200):
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self(x)
            loss = criterion(output, x)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        return losses


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 3)
        self.fc3 = nn.Linear(3, 3)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        return self.softmax(self.fc3(x))

    def classify(self, x):
        return torch.argmax(self(x), dim=1)

    def train_classifier(self, x, y, lr=0.1, epochs=400):
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self(x)
            loss = criterion(output, y)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        return losses


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 3)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        return self.activation(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 4)

    def forward(self, x):
        return self.fc1(x)

class AEClassifier(nn.Module):
    """Same architecture as the classifier, but the first layer is
    pre-trained as an autoencoder
    """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fc1 = nn.Linear(3, 3)
        self.fc2 = nn.Linear(3, 3)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return self.softmax(x)

    def autoencode(self, x):
        return self.decoder(self.encoder(x))
        
    def classify(self, x):
        return torch.argmax(self(x), dim=1)

    def pretrain_autoencoder(self, x, lr=0.01, epochs=400):
        #optimize both encoder and decoder
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self.autoencode(x)
            loss = criterion(output, x)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        return losses


    def train_classifier(self, x, y, lr=0.1, epochs=400):
        ae_loss = self.pretrain_autoencoder(x)
        #freeze the encoder 
        for param in self.encoder.parameters():
            param.requires_grad = False
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()), lr=lr)
        criterion = nn.CrossEntropyLoss()
        losses = []
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self(x)
            loss = criterion(output, y)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        return losses
