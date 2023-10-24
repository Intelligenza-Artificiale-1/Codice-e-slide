# Import delle librerie
from sklearn import tree  
import numpy as np
import matplotlib.pyplot as plt
import utils
import classification_net
import torch
import sys
import manual_tree

#set random seed
np.random.seed(0)


X,y = utils.load_data("dataset.dat")
def net():
    X_t = torch.tensor(X, dtype=torch.float)
    y_t = torch.tensor(y, dtype=torch.long)

    net = classification_net.Net()
    net.train_loop(X_t, y_t)
    utils.plot(X,y, net, title="Neural Network", pause=True)
    utils.classification_stats(y_t.numpy(), net.predict(X_t))

def manual():
    clf = manual_tree.DecisionTree(max_depth=3)
    clf.fit(X, y)
    clf.predict(X)
    print(clf)
    utils.plot(X,y, clf, pause=True)
    utils.classification_stats(y, clf.predict(X))


def classification_tree():
    clf = tree.DecisionTreeClassifier(
            max_depth=3, 			    # Profondità massima dell'albero
            )                           # Creazione del classificatore
    clf = clf.fit(X, y) 			    # Addestramento del classificatore
    clf.predict(X) 					    # Predizione
    tree.plot_tree(clf) 			    # Visualizzazione dell'albero
    plt.show()
    print(tree.export_text(clf)) 	    # Visualizzazione dell'albero in testo
    utils.plot(X,y, clf, pause=True)
    utils.classification_stats(y, clf.predict(X))


    for depth in range(1,15):
        clf = tree.DecisionTreeClassifier(
                max_depth=depth, 			# Profondità massima dell'albero
                )                           # Creazione del classificatore
        clf = clf.fit(X, y) 			    # Addestramento del classificatore
        utils.plot(X,y, clf, title=f"Depth: {depth}")
        utils.classification_stats(y, clf.predict(X))

if __name__ == "__main__":
    if sys.argv[1] == "tree":
        classification_tree()
    elif sys.argv[1] == "net":
        net()
    elif sys.argv[1] == "manual":
        manual()
