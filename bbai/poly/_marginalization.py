import numpy as np
import scipy.special
import scipy.interpolate

def compute_normalizer(poly, mult_p, num_points):
    roots, weights = scipy.special.roots_hermite(num_points // 2 + 1)
    m = np.sqrt(mult_p)
    res = 0.0
    for r, w in zip(roots, weights):
        res += poly(r / m) * w
    return res / m

class MarginalFn:
    def __init__(self, pts, vals, t0, mult_p):
        self.poly_ = scipy.interpolate.BarycentricInterpolator(pts, vals)
        self.t0_ = t0
        self.mult_p_ = mult_p
        self.N_ = compute_normalizer(self.poly_, mult_p, len(pts))

    def __call__(self, t):
        t = t - self.t0_
        return self.poly_(t) * np.exp(-self.mult_p_ * t**2) / self.N_

def marginalize_slice(f, degree, mult, t0, dim):
    n = (degree + 1) // 2 + int(degree % 2 == 0)
    roots, weights = scipy.special.roots_hermite(n)
    m = np.sqrt(mult)
    res = 0
    for r, w in zip(roots, weights):
        t = np.array(t0)
        t[dim] += r / m
        res += w * f(t) * np.exp(r**2)
    return res / m

def marginalize(f, degree, hess, x0, dim):
    dim_p = None
    if dim == 0:
        dim_p = 1
    else:
        dim_p = 0
    mult = 0.5 * hess[dim, dim]
    mult_p = 0.5 * (hess[dim_p, dim_p] - hess[0, 1]**2 / hess[dim, dim])
    print('mult_p =', mult_p)

    pts = 2 * np.polynomial.chebyshev.chebpts1(2 * degree + 1) / np.sqrt(mult_p)
    vals = []
    alpha = hess[0, 1] / hess[dim, dim]
    for pt in pts:
        t0 = np.zeros(2)
        t0[dim] = x0[dim] + alpha * pt
        t0[dim_p] = x0[dim_p] + pt
        val = marginalize_slice(f, degree, mult, t0, dim) * np.exp(mult_p * pt**2)
        vals.append(val)
        print('marginal:', pt, '-->', val)
    return MarginalFn(pts, vals, x0[dim_p], mult_p) 
