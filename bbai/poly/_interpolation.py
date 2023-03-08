import numpy as np

def interpolate_2d(X, y, degree):
    X = np.array(X)
    y = np.array(y)
    n = (degree + 1)**2
    assert len(X) == n
    assert len(y) == n
    A = np.zeros((n, n))
    for data_index in range(n):
        x1 = X[data_index, 0]
        x2 = X[data_index, 1]
        index = 0
        for i in range(degree+1):
            x1i = x1**i
            for j in range(degree+1):
                x2j = x2**j
                A[data_index, index] = x1i * x2j
                index += 1
    coef = np.linalg.solve(A, y)
    return coef.reshape(degree+1, degree+1)
