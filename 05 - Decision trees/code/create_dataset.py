from sklearn.datasets import make_moons

X, y = make_moons(n_samples=200, noise=0.25, random_state=3)

for i in range(0,200):
    print(*X[i], y[i])
