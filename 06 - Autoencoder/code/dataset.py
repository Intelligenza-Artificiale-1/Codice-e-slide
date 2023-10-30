import torch
from sklearn.datasets import load_iris
import numpy as np

def load_torch_iris():
	dataset = load_iris()
	X = dataset.data
	y = dataset.target
	full_dataset = np.concatenate((X, y.reshape(-1,1)), axis=1)
	return torch.tensor(full_dataset, dtype=torch.float32)
