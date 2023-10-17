import nets
import torch.nn as nn
import utils
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

#set random seed
torch.manual_seed(1)

def widenet(dataset):
    x,y = utils.load_data(dataset)
    x_train, y_train = x[:len(x)//2], y[:len(y)//2]
    x_test, y_test = x[len(x)//2:], y[len(y)//2:]
    for i in range(0, 101, 20):
        net = nets.WideNet(i)
        pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        train_loss = net.train(x_train, y_train, show=True)
        val_loss = nn.MSELoss()(net(x_test).squeeze(), y_test)
        utils.plot(x_test, y_test, net, f"Hidden size {i}, train loss {train_loss:.2f}, val loss {val_loss:.2f}, params {pytorch_total_params}", pause=True)

def deepnet(dataset):
    x,y = utils.load_data(dataset)
    x_train, y_train = x[:len(x)//2], y[:len(y)//2]
    x_test, y_test = x[len(x)//2:], y[len(y)//2:]
    for i in range(2, 8, 1):
        net = nets.DeepNet(i,10)
        pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        #dataset2 lr=0.1
        train_loss = net.train(x_train, y_train, lr=0.1, show=True)
        val_loss = nn.MSELoss()(net(x_test).squeeze(), y_test)
        utils.plot(x_test, y_test, net, f"Depth {i}, train loss {train_loss:.2f}, val loss {val_loss:.2f}, params {pytorch_total_params}", pause=True)

def taylornet(dataset):
    x,y = utils.load_data(dataset)
    x_train, y_train = x[:len(x)//2], y[:len(y)//2]
    x_test, y_test = x[len(x)//2:], y[len(y)//2:]
    for i in range(1, 10, 2):
        net = nets.TaylorNet(i)
        pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        train_loss = net.train(x_train, y_train, lr=0.01, show=True)
        val_loss = nn.MSELoss()(net(x_test).squeeze(), y_test)
        utils.plot(x_test, y_test, net, f"Degree {i}, train loss {train_loss:.2f}, val loss {val_loss:.2f}, params {pytorch_total_params}", pause=True)

def fouriernet(dataset):
    x,y = utils.load_data(dataset)
    x_train, y_train = x[:len(x)//2], y[:len(y)//2]
    x_test, y_test = x[len(x)//2:], y[len(y)//2:]
    for i in range(1, 10, 2):
        net = nets.FourierNet(i)
        pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        train_loss = net.train(x_train, y_train, lr=0.01, show=True)
        val_loss = nn.MSELoss()(net(x_test).squeeze(), y_test)
        utils.plot(x_test, y_test, net, f"Degree {i}, train loss {train_loss:.2f}, val loss {val_loss:.2f}, params {pytorch_total_params}", pause=True)

def coulombnet():
    x,y = utils.load_data("coulomb.dat")
    x_train, y_train = x[:3*len(x)//4], y[:3*len(y)//4]
    x_test, y_test = x[3*len(x)//4:], y[3*len(y)//4:]
    net = nets.CoulombNet()
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    train_loss = net.train(x_train, y_train, lr=0.1, epochs=2000)
    val_loss = nn.MSELoss()(net(x_test).squeeze(), y_test)
    print(f"Train loss {train_loss:.3f}, val loss {val_loss:.3f}, params {pytorch_total_params}")

    for q1 in np.linspace(-1e-7, 1e-7, 10):
        q2 = np.linspace(-1e-8, 1e-8, 10)
        r  = np.linspace(0.01, 0.005, 10)
        x = np.array([[q1, q2_s, r_s] for q2_s in q2 for r_s in r])
        y_hat = net(torch.tensor(x, dtype=torch.float32)).squeeze().detach().numpy()
        print(np.mean(y_hat), np.std(y_hat))
        #coulomb law ground truth
        epsilon_0 = 8.8541878128e-12
        y = np.array([1/(4*np.pi*epsilon_0) * q1*q2_s/r_s**2 for q1, q2_s, r_s in x])
        #3d mesh plot with q2, r, y
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(x[:,1], x[:,2], y,     label="ground truth")
        ax.plot_trisurf(x[:,1], x[:,2], y_hat, label="prediction")
        ax.title.set_text(f"q1={q1} C")
        plt.show()


if __name__ == "__main__":
    #widenet(sys.argv[1])
    #deepnet(sys.argv[1])
    #taylornet(sys.argv[1])
    #fouriernet(sys.argv[1])
    coulombnet()



