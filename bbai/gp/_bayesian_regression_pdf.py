import numpy as np

from scipy.optimize import root_scalar
from scipy.special import hyp2f1, loggamma

def compute_cdf_term1(a, b, v):
    log_a = np.log(a)
    log_b = np.log(b)
    res = 0.0
    res += - 2 * v * log_a 
    res += (v - 0.5) * (2 * log_a - log_b)
    res += 0.5 * np.log(np.pi)
    res += loggamma(v - 0.5)
    res -= np.log(2) + loggamma(v)
    return res 

def compute_cdf_term2(a, b, v, t, N):
    res = 0
    res += -v * np.log(b + a*a * t * t)
    res += v * np.log(1 + a * a * t * t / b)
    res += np.log(hyp2f1(0.5, v, 1.5, -a * a * t * t / b))
    return t * np.exp(res - N)

class BayesianRegressionPDF(object):
    def __init__(self, pdf_b_vector, pdf_matrix, n, mean):
        assert len(pdf_b_vector) == pdf_matrix.shape[1]
        assert pdf_matrix.shape[0] == 4
        self.pdf_b_vector_ = pdf_b_vector
        self.pdf_matrix_ = pdf_matrix
        self.n_ = n
        self.mean_ = mean

    def __call__(self, y):
        return self.pdf(y)

    def pdf(self, y):
        res = 0.0
        num_points = len(self.pdf_b_vector_)
        for point_index in range(num_points):
            b = self.pdf_b_vector_[point_index]
            sign_multiplier, log_abs_multiplier, mean, a = self.pdf_matrix_[:, point_index]
            if sign_multiplier == 0:
                continue
            delta = y - mean 
            t = 0.5 * (self.n_ + 1 ) * np.log((delta * a)**2 + b)
            res += sign_multiplier * np.exp(log_abs_multiplier - t)
        return res

    def cdf(self, y):
        res = 0.0
        num_points = len(self.pdf_b_vector_)
        v = 0.5 * (self.n_ + 1)
        for point_index in range(num_points):
            b = self.pdf_b_vector_[point_index]
            sign_multiplier, log_abs_multiplier, mean, a = self.pdf_matrix_[:, point_index]
            if sign_multiplier == 0:
                continue
            t1 = compute_cdf_term1(a, b, v)
            t2 = compute_cdf_term2(a, b, v, y - mean, t1)
            z = 1.0 + t2
            if z <= 0:
                continue
            res += sign_multiplier * np.exp(log_abs_multiplier + t1) * z
        return res

    def ppf(self, p):
        assert 0 < p and p < 1
        def f(t):
            if t <= 0.0:
                return -p
            return self.cdf(t) - p
        def fp(t):
            return self.pdf(t)
        res = root_scalar(
                f,
                x0 = self.mean_,
                fprime = fp,
        )
        return res.root
