import scipy.interpolate
import numpy as np
from scipy.optimize import root_scalar

class Marginal:
    def __init__(
            self,
            t0,
            a, b,
            point_vector,
            integral_point_vector,
            value_vector,
            integral_vector):
        self.t0_ = t0
        self.bracket_low_ = t0 + a
        self.bracket_high_ = t0 + b
        self.a_ = a
        self.b_ = b
        self.point_vector_ = point_vector
        self.integral_point_vector_ = integral_point_vector
        self.value_vector_ = value_vector
        self.integral_vector_ = integral_vector

    def pdf(self, t):
        t -= self.t0_
        in_range = (t >= self.a_) * (t <= self.b_)
        mid = (self.a_ + self.b_) / 2.0
        mult = (self.b_ - self.a_) / 2.0
        tp = (t - mid) / mult
        return in_range * scipy.interpolate.barycentric_interpolate(
                self.point_vector_,
                self.value_vector_,
                tp,
        ) / mult

    def cdf(self, t):
        t -= self.t0_
        if t <= self.a_:
            return 0
        if t >= self.b_:
            return 1
        mid = (self.a_ + self.b_) / 2.0
        mult = (self.b_ - self.a_) / 2.0
        tp = (t - mid) / mult
        return scipy.interpolate.barycentric_interpolate(
                self.integral_point_vector_,
                self.integral_vector_,
                tp,
        )

    def ppf(self, p):
        assert 0 <= p and p <= 1
        low = self.t0_ + self.a_
        high = self.t0_ + self.b_
        def f(t):
            return self.cdf(t) - p
        def fp(t):
            return self.pdf(t)
        res = root_scalar(
                f,
                x0 = self.t0_,
                bracket = (low, high),
                fprime = fp,
        )
        return res.root

class LogMarginal:
    def __init__(self, marginal):
        self.marginal_ = marginal
        self.bracket_low_ = np.exp(marginal.bracket_low_)
        self.bracket_high_ = np.exp(marginal.bracket_high_)

    def pdf(self, t):
        return self.marginal_.pdf(np.log(t)) / t

    def cdf(self, t):
        assert t >= 0
        if t == 0:
            return 0
        return self.marginal_.cdf(np.log(t))

    def ppf(self, p):
        assert 0 <= p and p <= 1
        return np.exp(self.marginal_.ppf(p))
