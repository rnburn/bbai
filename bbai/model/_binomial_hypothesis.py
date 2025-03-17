from scipy.stats import beta
from scipy.special import betaln
import numpy as np

INTEGRAL_WEIGHT = 0.73085

class _WrappedPDF:
    def __init__(self, dist, a, b):
        self.dist_ = dist
        self.a_ = a
        self.b_ = b
        self.cdf_a_ = dist.cdf(a)
        self.Z_ = dist.cdf(b) - self.cdf_a_

    def pdf(self, t):
        if t < self.a_ or t > self.b_:
            return 0.0
        return self.dist_.pdf(t) / self.Z_

    def cdf(self, t):
        if t < self.a_:
            return 0
        if t > self.b_:
            return 1
        return (self.dist_.cdf(t) - self.cdf_a_) / self.Z_

class _MixedPDF:
    def __init__(self, dist1, dist2, w):
        self.dist1_ = dist1
        self.dist2_ = dist2
        self.w_ = w

    def pdf(self, t):
        return self.w_ * self.dist1_.pdf(t) + (1 - self.w_) * self.dist2_.pdf(t)

    def cdf(self, t):
        return self.w_ * self.dist1_.cdf(t) + (1 - self.w_) * self.dist2_.cdf(t)

def make_dist(a, b):
    ap = a + 1
    bp = b + 1
    log_norm = betaln(ap, bp)
    return beta(ap, bp), log_norm

def make_prior_dists(prior):
    if prior == 'laplace':
        dist = make_dist(0, 0)[0]
        return [
            _WrappedPDF(dist, 0, 0.5),
            _WrappedPDF(dist, 0.5, 1),
        ]
    if prior == 'jeffreys':
        dist = make_dist(-0.5, -0.5)[0]
        return [
            _WrappedPDF(dist, 0, 0.5),
            _WrappedPDF(dist, 0.5, 1),
        ]
    res = []
    dist1, _ = make_dist(-0.5, 0.5)
    dist2, _ = make_dist(0.5, -0.5)
    weights = [1 - INTEGRAL_WEIGHT, INTEGRAL_WEIGHT]
    ranges = [(0, 0.5), (0.5, 1)]
    for w, (a, b) in zip(weights, ranges):
        dist = _MixedPDF(
                _WrappedPDF(dist1, a, b), 
                _WrappedPDF(dist2, a, b), 
        w)
        res.append(dist)
    return res


class BinomialHypothesis2:
    def __init__(self, y0=None, y1=None, prior='integral'):
        self.priors_ = make_prior_dists(prior)
        if y0 is None or y1 is None:
            return
        if prior == 'laplace':
            self._fit_laplace(y0, y1)
        elif prior == 'jeffreys':
            self._fit_jeffreys(y0, y1)
        else:
            self._fit_integral(y0, y1)

    def _fit_laplace(self, y0, y1):
        dist, _ = make_dist(y0, y1)
        self.left_ = dist.cdf(0.5)
        self.right_ = 1 - self.left_

    def _fit_jeffreys(self, y0, y1):
        dist, _ = make_dist(y0 - 0.5, y1 - .5)
        self.left_ = dist.cdf(0.5)
        self.right_ = 1 - self.left_

    def _fit_integral(self, y0, y1):
        dist1, log_norm1 = make_dist(y0 - 0.5, y1 + .5)
        dist2, log_norm2 = make_dist(y0 + .5, y1 - .5)

        r = np.exp(log_norm1 - log_norm2)

        w = INTEGRAL_WEIGHT

        p1 = (1 - w) * 4 / (np.pi + 2) * dist1.cdf(0.5) * r
        p1 += w * 4 / (np.pi - 2) * dist2.cdf(0.5)

        p2 = w * 4 / (np.pi - 2) * (1 - dist1.cdf(0.5)) * r
        p2 += (1 - w) * 4 / (np.pi + 2) * (1 - dist2.cdf(0.5))
        
        N = p1 + p2
        self.left_ = p1 / N
        self.right_ = p2 / N
