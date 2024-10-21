from bbai._computation._bridge import mdlbn

import numpy as np

class BoundedNormal:
    def __init__(self, a, b):
        assert a < b
        self.a_ = a
        self.b_ = b

    def fit_mean_var(self, n, mean, variance):
        self.model_ = mdlbn.bounded_normal_fit(self.a_, self.b_, n, mean, variance)

    def fit(self, y):
        n = len(y)
        mean = np.mean(y)
        sigma = np.std(y)
        self.fit_mean_var(n, mean, sigma**2)

    def pdf(self, x):
        assert self.model_ is not None
        return mdlbn.pdf(self.model_, x)

    def cdf(self, x):
        assert self.model_ is not None
        return mdlbn.cdf(self.model_, x)

    def ppf(self, t):
        assert 0 <= t and t <= 1
        assert self.model_ is not None
        return mdlbn.ppf(self.model_, t)
