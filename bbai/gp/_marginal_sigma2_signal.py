import numpy as np
import math
from scipy.special import loggamma, gammaincc
from scipy.optimize import root_scalar

def compute_pdf_multipliers(weight_vector, s2_vector, alpha):
    res = []
    log_divisor = loggamma(alpha)
    for i, w in enumerate(weight_vector):
        beta = s2_vector[i] / 2
        sgn = math.copysign(1.0, w)
        log_abs_w = np.log(np.abs(w))
        val = log_abs_w + alpha * np.log(beta) - log_divisor
        res.append((sgn, val))
    return res

class MarginalSigma2Signal:
    def __init__(self, weight_vector, s2_vector, n):
        alpha = n / 2.0
        self.alpha_ = alpha
        self.weight_vector_ = weight_vector
        self.s2_vector_ = s2_vector
        self.pdf_multipliers_ = compute_pdf_multipliers(weight_vector, s2_vector, alpha)
        mean_s2_vector = 0
        for w, s2val in zip(weight_vector, s2_vector):
            mean_s2_vector += w * s2val
        self.mean_s2_vector = mean_s2_vector

    def pdf(self, t):
        is_zero = (t == 0)
        t += is_zero * 1.0e-5
        res = 0
        term1 = (-self.alpha_ - 1) * np.log(t)
        for i, (sgn, log_mult) in enumerate(self.pdf_multipliers_):
            if sgn == 0:
                continue
            beta = self.s2_vector_[i] / 2
            val = sgn * np.exp(log_mult + term1 - beta / t)
            res += val
        return res * (t != 0)

    def cdf(self, t):
        assert t >= 0
        if t == 0:
            return 0
        res = 0
        for i, w in enumerate(self.weight_vector_):
            beta = self.s2_vector_[i] / 2
            res += w * gammaincc(self.alpha_, beta / t)
        return res

    def ppf(self, p):
        assert 0 < p and p < 1
        def f(t):
            if t <= 0.0:
                return -p
            return self.cdf(t) - p
        def fp(t):
            if t <= 0:
                return 0.0
            return self.pdf(t)
        high = 10
        while self.cdf(high) < p:
            high *= 2
        res = root_scalar(
                f,
                x0 = self.mean_s2_vector / self.alpha_ / 2.0,
                bracket = (0, high),
                fprime = fp,
        )
        return res.root
