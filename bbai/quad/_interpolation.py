import numpy as np
from scipy.interpolate import interp2d
from bbai.poly import interpolate_2d

class Interpolation:
    def __init__(self, poly, x0, eigenvalues, eigenvectors):
        self.poly_ = poly
        self.x0_ = x0
        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = eigenvectors

    def __call__(self, x):
        t = x - self.x0_
        tp = np.dot(self.eigenvectors_.T, t)
        pval = np.polynomial.polynomial.polyval2d(tp[0], tp[1], self.poly_)
        m = np.exp(-0.5 * np.dot(self.eigenvalues_, tp**2))
        # print('tp =', tp)
        # print('inter:', pval, m)
        return pval * m

def interpolate_grid(grid):
    X = []
    y = []
    for point in grid.points:
        X.append(point.x)
        y.append(point.val / point.multiplier)
    poly = interpolate_2d(X, y, grid.k - 1)
    res = Interpolation(poly, grid.x0, grid.eigenvalues, grid.eigenvectors)

    return res
