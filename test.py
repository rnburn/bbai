import peak_engines
import numpy as np
np.random.seed(0)

n = 15
k = 2

def make_random_problem():
    X = np.random.random_sample((n,k))
    for j in range(X.shape[1]):
        col = X[:, j]
        X[:, j] = (col - np.mean(col)) / np.std(col)
    beta = np.array([1.0, 0.0])
    y = np.dot(X, beta) + np.random.normal(size=n)
    return X, y

X, y = make_random_problem()
model = peak_engines.WarpedLinearRegressionModel()
model.fit(X, y)
assert model.within_tolerance_
print(model.regressors_)
