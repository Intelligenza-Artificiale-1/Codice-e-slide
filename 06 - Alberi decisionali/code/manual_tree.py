class Node:
    def __init__ (self):
        self.left = None
        self.right = None
        self.feature = None
        self.threshold = None
        self.label = None

    def __str__(self, depth=1) -> str:
        """Prints the decision tree
        Args:
            depth(int): Depth of the node

        Returns:
            str: String representation of the tree
        """
        if self.label is not None:
            return f' class: {int(self.label)}'
        return f'x_{self.feature} <= {self.threshold:.2f}\n' + \
                f'{"|    " * depth}{self.left.__str__(depth + 1)}\n' + \
                f'{"|    " * depth}{self.right.__str__(depth + 1)}' 

    def _forward(self, x: list) -> int:
        """
        Predicts the class of a sample. If the node is a leaf, returns the label. Otherwise,
        goes to the left or right child depending on the feature value of the sample
        Args:
            x (list): Sample to predict
        Returns:
            int: Predicted class
        """
        return 1

    def _gini(self, Y:list) -> float:
        """Calculates the Gini impurity of a dataset as 1 - the sum of the squared probabilities of each class
        Args:
            Y(list): Labels
        Returns:
            float: Gini impurity
        """
        return 1

    def _best_split(self, X:list, Y:list) -> tuple:
        """Finds the best split for the data based on the Gini impurity
        Args:
            X(list): Features
            Y(list): Labels
        Returns:
            split(tuple): Best feature and threshold
        """
        return (0,X[0][0])

    def _fit(self, X:list, Y:list, depth:int=0) -> None:
        """Recursively fits the decision tree to the data. 
        If the depth is 0 or the labels are all the same, the node is a leaf
        Otherwise, it finds the best split and recursively fits the left and right subtrees
        on the two data partitions.
        If the two subtrees have the same label, the node is a leaf and the subtrees are removed
        Args:
            X(list): Features
            Y(list): Labels
            depth(int): Depth of the subtree having this node as root
        """
        self.label = max(set(Y), key=Y.count)


class DecisionTree(Node):
    def __init__(self, max_depth:int=3):
        self.max_depth = max_depth
        super().__init__()

    def fit(self, X:list, Y:list, quiet:bool=False) -> None:
        """Fits the decision tree to the data
        Args:
            X(list): Features
            Y(list): Labels
            quiet(bool): If True, does not print the tree
        """
        self._fit(X, Y, depth=self.max_depth)
        if not quiet:
            print(self)

    def __call__(self, x: list) -> list:
        """
        Predicts the class of a sample or a list of samples
        Args:
            x (list): Sample or list of samples to predict
        Returns:
            list: Class predictions
        """
        if not isinstance(x[0], list):
            x = [x]
        return [self._forward(sample) for sample in x]