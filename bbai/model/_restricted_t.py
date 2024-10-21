import numpy as np
import scipy

class RestrictedT:
    def __init__(self, a, b):
        assert a < b
        self.a_ = a
        self.b_ = b

    def fit_mean_var(self, n, mean, variance):
        s = np.sqrt(variance * n / (n - 1))
        s /= np.sqrt(n)
        self.model_ = scipy.stats.t(n-1, loc=mean, scale=s)

        self.cdf_a_ = self.model_.cdf(self.a_)
        self.norm_ = self.model_.cdf(self.b_) - self.cdf_a_

    def fit(self, y):
        mean = np.mean(y)
        var = np.std(y)**2
        self.fit_mean_var(len(y), mean, var)

    def pdf(self, x):
        if x < self.a_ or x > self.b_:
            return 0
        return self.model_.pdf(x) / self.norm_

    def cdf(self, x):
        if x < self.a_:
            return 0
        if x > self.b_:
            return 1
        return (self.model_.cdf(x) - self.cdf_a) / self.norm_

    def ppf(self, t):
        assert 0 <= t and t <= 1
        t *= self.norm_
        t += self.cdf_a_
        return self.model_.ppf(t)
