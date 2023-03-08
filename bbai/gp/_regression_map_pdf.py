import numpy as np

class RegressionMAPPDF(object):
    def __init__(self, packed_Al, b, N, y_mean, n):
        m = len(y_mean)
        Al = np.zeros((m, m))
        index = 0
        for j in range(m):
            for i in range(j, m):
                Al[i, j] = packed_Al[index]
                index += 1
        self.Al_ = Al
        self.b_ = b
        self.N_ = N
        self.y_mean_ = y_mean
        self.n_ = n

    def __call__(self, y):
        t = np.dot(self.Al_.T, y - self.y_mean_)
        v = (self.n_ + len(self.y_mean_)) / 2.0
        z = -v * np.log(np.dot(t, t) + self.b_)
        return np.exp(z - self.N_)
