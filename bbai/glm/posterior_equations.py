import numpy as np

def log_jeffreys_posterior(X, y, w):
    n = len(y)
    u = np.dot(X, w)
    a = np.zeros(n)
    like = 0
    for i in range(n):
        ui = u[i][0]
        p = 1 / (1.0 + np.exp(-ui))
        q = 1 - p
        if y[i] == 1:
            like += np.log(p)
        else:
            like += np.log(q)
        a[i] = p * q
    H = np.dot(X.T, np.dot(np.diag(a), X))
    L = np.linalg.cholesky(H)
    prior = np.sum(np.log(np.diag(L)))
    return like + prior
