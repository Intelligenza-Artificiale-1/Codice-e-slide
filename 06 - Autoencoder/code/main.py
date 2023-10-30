import dataset, nets, utils, torch
import matplotlib.pyplot as plt
import sys

#set random seed
torch.manual_seed(3)


def iris_ae():
    """Train an autoencoder on the iris dataset and plot the loss
    """
    ae = nets.PlainAutoencoder()
    iris = dataset.load_torch_iris()
    loss=ae.train_autoencoder(iris)
    plt.plot(loss)
    plt.show()
    #plot 2D representation
    utils.plot_2d((iris).detach().numpy(), f1=0, f2=1)
    error =(iris-ae(iris)).pow(2).sum(dim=1)
    utils.plot_mse(iris, error, iris[:, -1], f1=0, f2=1)
    #plot 3D representation
    utils.plot_3d(ae.encoder(iris).detach().numpy(), iris[:, -1].detach().numpy())

def iris_classifier(pretrain=True):
    """Train a classifier on the iris dataset and plot the loss
    """
    iris = dataset.load_torch_iris()
    #shuffle
    iris = iris[torch.randperm(iris.size()[0])]
    #split train/test
    iris_train, iris_test = iris[:100], iris[100:]
    x = iris_train[:, :-1]
    y = iris_train[:, -1].long()

    x_test = iris_test[:, :-1]
    y_test = iris_test[:, -1].long()
    #choose the model
    if pretrain:
        classifier = nets.AEClassifier()
    else:
        classifier = nets.Classifier()
    loss = classifier.train_classifier(x, y)
    plt.plot(loss)
    plt.show()
    utils.confusion_matrix(y_test, classifier.classify(x_test))



if __name__ == "__main__":
    iris_ae()
    #iris_classifier(pretrain=int(sys.argv[1]))
