from scipy.optimize import root_scalar
import numpy as np
import scipy.stats

class MarginalRegressor:
    def __init__(self, weight_vector, beta_hat_matrix, axi_diagonals, s2_vector, df, j):
        self.df_ = df
        self.j_ = j
        self.weight_vector_ = weight_vector
        self.beta_hat_matrix_ = beta_hat_matrix
        
        num_points = len(weight_vector)
        scale_vector = np.zeros(num_points)
        mean = 0
        for point_index, weight in enumerate(weight_vector):
            scale_vector[point_index] = \
                    np.sqrt(axi_diagonals[j, point_index] * s2_vector[point_index] / df)
            mean += weight * beta_hat_matrix[j, point_index]
        self.mean_ = mean
        self.scale_vector_ = scale_vector

    def __call__(self, bj):
        return self.pdf(bj)

    def pdf(self, bj):
        res = 0.0
        for point_index, weight in enumerate(self.weight_vector_):
            mean = self.beta_hat_matrix_[self.j_, point_index]
            scale = self.scale_vector_[point_index]
            res += weight * scipy.stats.t.pdf(bj, df = self.df_, loc = mean, scale = scale);
        return res

    def cdf(self, bj):
        res = 0.0
        for point_index, weight in enumerate(self.weight_vector_):
            mean = self.beta_hat_matrix_[self.j_, point_index]
            scale = self.scale_vector_[point_index]
            res += weight * scipy.stats.t.cdf(bj, df = self.df_, loc = mean, scale = scale);
        return res

    def ppf(self, p):
        assert 0 < p and p < 1
        def f(t):
            return self.cdf(t) - p
        def fp(t):
            return self.pdf(t)
        step = 1
        low = self.mean_ - step
        while self.cdf(low) > p:
            step *= 2
            low -= step
        step = 1
        high = self.mean_ + step
        while self.cdf(high) < p:
            step *= 2
            high += step
        res = root_scalar(
                f,
                x0 = self.mean_,
                bracket = (low, high),
                fprime = fp,
        )
        return res.root
