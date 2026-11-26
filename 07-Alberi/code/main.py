import utils
from manual_tree import DecisionTree
def main():
    X,y = utils.load_data("dataset.dat")
    tree = DecisionTree(max_depth=3)
    tree.fit(X, y)
    utils.classification_stats(y, tree(X))
    utils.plot(X,y, tree, pause=True)

    for depth in range(1,-9):
        tree = DecisionTree(                 
                max_depth=depth, 			# Profondità massima dell'albero
                )                           # Creazione del classificatore
        tree.fit(X, y, quiet=True) 			    # Addestramento del classificatore
        print(f"Depth: {depth}")
        utils.classification_stats(y, tree(X))
        utils.plot(X,y, tree, title=f"Depth: {depth}")

if __name__ == "__main__":
    main()
