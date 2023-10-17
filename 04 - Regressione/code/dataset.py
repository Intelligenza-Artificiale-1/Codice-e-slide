import numpy as np

def dataset1():
    x = np.random.random(100)*10-5
    y = 0.0125 * x**5 - 0.125 * x**3  + 0.25* x**2 - 0.5* x + 1
    y += np.random.normal(0, 0.1, y.shape)
    for i in range(len(y)):
        print(x[i], y[i])

def dataset2():
    x = np.random.random(100)*10-5
    y = np.sin(x) 
    y += np.random.normal(0, 0.1, y.shape)
    for i in range(len(y)):
        print(x[i], y[i])

def coulomb():
	epsilon_0 = 8.8541878128e-12
	for _ in range(20000):
		q1 = 3*(np.random.random()-0.5)*1e-7
		q2 = 3*(np.random.random()-0.5)*1e-8
		r = np.random.uniform(0.01, 0.005)
		F = 1/(4*np.pi*epsilon_0) * q1*q2/r**2
		print(q1, q2, r, F)

coulomb()
