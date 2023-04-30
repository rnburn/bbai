import numpy as np

from scipy.optimize import root_scalar
from scipy.special import hyp2f1, loggamma
import scipy.stats

class BayesianRegressionPDF(object):
    def __init__(self, pdf_b_vector, pdf_matrix, df, mean, weight_vector):
        assert len(pdf_b_vector) == pdf_matrix.shape[1]
        assert pdf_matrix.shape[0] == 4
        self.pdf_b_vector_ = pdf_b_vector
        self.pdf_matrix_ = pdf_matrix
        self.df_ = df
        self.mean_ = mean
        self.weight_vector_ = weight_vector

        num_points = len(weight_vector)
        self.scale_vector_ = np.zeros(num_points)
        for point_index, weight in enumerate(weight_vector):
            if weight == 0:
                continue
            b = self.pdf_b_vector_[point_index]
            _, _, _, a = self.pdf_matrix_[:, point_index]
            scale = np.sqrt(b / self.df_) / a
            self.scale_vector_[point_index] = scale

    def __call__(self, y):
        return self.pdf(y)

    def pdf(self, y):
        res = 0.0
        for point_index, weight in enumerate(self.weight_vector_):
            if weight == 0:
                continue
            _, _, mean, _ = self.pdf_matrix_[:, point_index]
            scale = self.scale_vector_[point_index]
            res += weight * scipy.stats.t.pdf(y, df = self.df_, loc = mean, scale = scale);
        return res

    def cdf(self, y):
        res = 0.0
        for point_index, weight in enumerate(self.weight_vector_):
            if weight == 0:
                continue
            _, _, mean, _ = self.pdf_matrix_[:, point_index]
            scale = self.scale_vector_[point_index]
            res += weight * scipy.stats.t.cdf(y, df = self.df_, loc = mean, scale = scale);
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
