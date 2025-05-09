import numpy as np

class RandomRegressionDataset:
    def __init__(self,
                 n = 10,
                 p = 5,
                 cor = 0.5,
                 err = .1,
                 active_prob = 0.5,
                 with_intercept = False,
        ):
        self.with_intercept = with_intercept

        # X
        K = np.zeros((p, p))
        for j1 in range(p):
            for j2 in range(p):
                K[j1, j2] = cor ** np.abs(j1 - j2)
        self.X = np.random.multivariate_normal(np.zeros(p), K, size=n)

        # beta
        beta = np.zeros(p)
        for j in range(p):
            if not np.random.binomial(1, active_prob):
                continue
            beta[j] = np.random.laplace()
        self.beta = beta
        self.t = np.sum(np.abs(beta))

        # y
        y = np.dot(self.X, self.beta)
        if with_intercept:
            y += np.random.laplace()
        y += np.random.normal(0, err, size=n)
        self.y = y

        # lambda_max
        u = np.dot(self.X.T, self.y)
        if with_intercept:
            u -= np.mean(self.y) 
        self.lambda_max = np.max(np.abs(u))

