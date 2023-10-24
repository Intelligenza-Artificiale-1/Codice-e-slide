#Manual implementation of decision tree
import numpy as np

class Node():
    def __init__(self, feature, threshold, left=None, right=None, prediction=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.prediction = prediction

    def predict(self, x):
        """Predicts the target value for the given input feature

        :x: Input feature
        :returns: Predicted target value
        """
        if self.prediction is not None:
            return self.prediction
        if x[self.feature] <= self.threshold:
            return self.left.predict(x)
        else:
            return self.right.predict(x)

    def __str__(self, depth=0):
        """Prints the decision tree

        :returns: String representation of the tree
        """
        if self.prediction is not None:
            return str(self.prediction)
        return f'x{self.feature} <= {self.threshold:.2f}\n' + \
                f'{"    " * depth}{self.left.__str__(depth + 1)}\n' + \
                f'{"    " * depth}{self.right.__str__(depth + 1)}' 


class DecisionTree():
    def __init__(self, max_depth=3): 
        self.max_depth = max_depth

    def fit(self, X, y):
        """Trains the decision tree on the given dataset

        :X: Input features
        :y: Target values
        """
        self.n_classes = len(set(y))
        self.n_features = X.shape[1]
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        """Predicts the target values for the given input features

        :X: Input features
        :returns: Predicted target values
        """
        return np.array([self.tree.predict(x) for x in X])

    def _grow_tree(self, X, y, depth=0):
        """Grows the decision tree recursively

        :X: Input features
        :y: Target values
        :depth: Current depth of the tree
        :returns: Root node of the tree
        """
        # Stop growing the tree if the maximum depth is reached
        num_classes = len(set(y))
        if depth < self.max_depth and num_classes > 1:
            best_feature, best_threshold = self._best_criteria(X, y)
            # Split the data based on the best feature and threshold
            left_data    = X[X[:, best_feature] <= best_threshold]
            left_labels  = y[X[:, best_feature] <= best_threshold]
            right_data   = X[X[:, best_feature] >  best_threshold]
            right_labels = y[X[:, best_feature] >  best_threshold]
            # Grow the left and right subtrees
            left_node  = self._grow_tree(left_data,  left_labels, depth + 1)
            right_node = self._grow_tree(right_data, right_labels, depth + 1)
            if left_node.prediction == right_node.prediction != None:
                node = Node(None, None, None, None, left_node.prediction)
            else:
                node = Node(best_feature, best_threshold, left_node, right_node)
        else:
            n_samples_per_class = [np.sum(y == i) for i in range(self.n_classes)]
            # Predict the majority class
            prediction = np.argmax(n_samples_per_class)
            node = Node(None, None, None, None, prediction)
        return node

    def _best_criteria(self, X, y):
        """Finds the best feature and threshold to split the data

        :X: Input features
        :y: Target values
        :returns: Best feature and threshold
        """
        lowest_entropy = float('inf')
        split_index, split_threshold = None, None
        for feature in range(self.n_features):
            # All unique values of the feature
            feature_values = np.expand_dims(X[:, feature], axis=1)
            unique_values = np.unique(feature_values)
            # Try all possible thresholds
            for threshold in unique_values:
                left_indices  = np.argwhere(feature_values <= threshold).flatten()
                right_indices = np.argwhere(feature_values > threshold).flatten()
                # Calculate the entropy of the split
                left_entropy = 0
                for c in range(self.n_classes):
                    p = np.sum(y[left_indices] == c) / len(left_indices)
                    if p > 0:
                        left_entropy += -p * np.log2(p)
                right_entropy = 0
                for c in range(self.n_classes):
                    p = np.sum(y[right_indices] == c) / len(right_indices)
                    if p > 0:
                        right_entropy += -p * np.log2(p)
                entropy = (len(left_indices) * left_entropy + len(right_indices) * right_entropy) / len(y)
                # Update the best split if necessary
                if entropy < lowest_entropy:
                    lowest_entropy = entropy
                    split_index = feature
                    split_threshold = threshold
        return split_index, split_threshold

    def __str__(self):
        """Prints the decision tree

        :returns: String representation of the tree
        """
        return str(self.tree)
