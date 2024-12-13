
import numpy as np
import torch
import torch.nn as nn
import os
import matplotlib.pyplot as plt

#set seed
np.random.seed(42)
torch.manual_seed(42)

class NN(nn.Module):
    def __init__(self):
        """
        DA COMPLETARE
        """

    def forward(self, x):
        """
        DA COMPLETARE
        """

    def fit(self, X_train, Y_train, epochs=5000, lr=0.1):
        losses = []
        """
        DA COMPLETARE
        """
        return losses

def load_dataset():
    #loads dataset and puts it into torch tensors (test and train split)
    dir_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(dir_path, 'dataset3.dat')
    ds = np.loadtxt(file_path, delimiter=',')
    ds = torch.from_numpy(ds).float()
    #split into test and train
    split = int(0.7 * len(ds))
    ds_train, ds_test = ds[:split], ds[split:]
    x_train, y_train = ds_train[:, :-1], ds_train[:, -1:]
    x_test, y_test = ds_test[:, :-1], ds_test[:, -1:]
    return x_train, y_train.squeeze().long(), x_test, y_test.squeeze().long()

def main():
    X_train, Y_train, X_test, Y_test = load_dataset()
    model = NN()
    #Determinare numero di epoche e learning rate
    losses = model.fit(X_train, Y_train)
    with torch.no_grad():
        Y_pred = model(X_test).detach()
        plt.figure()
        plt.suptitle(f'Test Accuracy: {100*torch.mean((torch.argmax(Y_pred, dim=1) == Y_test).float()).item():.2f}%')
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.yscale('log')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Function')

        Y_pred = Y_pred.numpy()

        #plot confusion matrix
        plt.subplot(2, 2, 2)
        cm = np.zeros((3, 3))
        for i in range(len(Y_test)):
            cm[int(Y_test[i]), np.argmax(Y_pred[i])] += 1
        plt.imshow(cm, cmap='Blues')
        #plt.colorbar()
        #set min and max values
        #plt.clim(0, 100)
        plt.xticks([0, 1, 2])
        plt.yticks([0, 1, 2])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        #add numbers
        for i in range(3):
            for j in range(3):
                plt.text(j, i, str(int(cm[i, j])), ha='center', va='center', color='red')
        #plot decision boundary
        plt.subplot(2, 2, 4)
    
        X, y = np.array(X_test), np.array(Y_test)
        min1, max1 = X[:, 0].min()-0.1, X[:, 0].max()+0.1
        min2, max2 = X[:, 1].min()-0.1, X[:, 1].max()+0.1
        # define the x and y scale
        x1grid = np.arange(min1, max1, (max1-min1)/1000)
        x2grid = np.arange(min2, max2, (max2-min2)/1000)
        # create all of the lines and rows of the grid
        xx, yy = np.meshgrid(x1grid, x2grid)
        # flatten each grid to a vector
        r1, r2 = xx.flatten(), yy.flatten()
        # horizontal stack vectors to create x1,x2 input for the model
        grid = torch.Tensor([[x1, x2] for x1, x2 in zip(r1,r2)])
        # make predictions for the grid
        yhat = np.argmax(model(grid),axis=1)
        # reshape the predictions back into a grid
        zz = yhat.reshape(xx.shape)
        # plot the grid of x, y and z values as a surface
        plt.contourf(xx, yy, zz, cmap='Accent')
        # create scatter plot for samples from each class
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap='Set3', edgecolors='black')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()