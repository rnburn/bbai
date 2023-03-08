import numpy as np
import scipy.special
import scipy.stats

def compute_gaussian_kl(mean, std, pdf, num_quadrature_points=9):
    def f(x):
        logp = scipy.stats.norm.logpdf(x, loc=mean, scale=std)
        logq = np.log(pdf(x))
        return logp - logq
    res = 0.0
    ux, wx = scipy.special.roots_hermite(num_quadrature_points)
    for ui, wi in zip(ux, wx):
        xi = std * np.sqrt(2) * ui + mean
        res += f(xi) * wi
    res /= np.sqrt(np.pi)
    return res

def compute_gaussian_gaussian_kl(mean1, std1, mean2, std2):
    term1 = np.log(std2) - np.log(std1)
    term2 = (std1**2 + (mean1 - mean2)**2) / (2 * std2**2)
    return term1 + term2 - 0.5
