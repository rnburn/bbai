from .._computation._bridge import numsg

import inspect
import numpy as np

class SparseGridInterpolator:
    """Fit an adapative sparse grid at Chebyshev-Gauss-Lobatto nodes to approximate a given target 
    function, f: R^k -> R, over a specified k-cell. The approximation can be used both for 
    interpolation and integration.

    For background on the algorithm see

    * Bathelmann Volker, Novak Erich, Ritter Klaus. High dimensional polynomial interpolation
    on sparse grids // Advances in Computation Mathematics. 2000. 273-288
    * Gerstner Thomas, Griebel Michael. Dimension–Adaptive Tensor–Product Quadrature // Computing. 
    09 2003. 71. 65–87
    * Klimke Andreas. Uncertainty Modeling using Fuzzy Arithmetic and Sparse Grids. 01 2006. 40–41.
    * Jakeman John D., Roberts Stephen G. Local and Dimension Adaptive Sparse Grid Interpolation and 
    Quadrature. 2011

    and for a general overview see Algorithm 5 of https://arxiv.org/abs/2307.08997.
    
    Parameters
    ----------
    tolerance : float, default=1.0e-4
        The desired accuracy to achieve.

    ranges: default=None
        Specify the k-cell, [(a1, b1), ..., (ak, bk)], to interpolate on. If unspecified,
        SparseGridInterpolator will default to [0, 1]^k where k will be determined by the
        arity of the function provided to fit.

    Example
    --------
    >>> from bbai.numeric import SparseGridInterpolator
    >>> import numpy as np
    >>> 
    >>> # An example function to interpolate
    >>> def f(x, y, z):
    >>>     t1 = 0.68 * np.abs(x - 0.3)
    >>>     t2 = 1.25 * np.abs(y - 0.15)
    >>>     t3 = 1.86 * np.abs(z - 0.09)
    >>>     return np.exp(-t1 - t2 - t3)
    >>> 
    >>> # Fit a sparse grid to approximate f
    >>> ranges = [(-2, 5), (1, 3), (-2, 2)]
    >>> interp = SparseGridInterpolator(tolerance=1.0e-4, ranges=ranges)
    >>> interp.fit(f)
    >>> 
    >>> # Test the accuracy at a random point of the domain
    >>> print(interp.evaluate(1.84, 2.43, 0.41), f(1.84, 2.43, 0.41))
    >>> #    prints 0.011190847391188667 0.011193746554063376
    >>> 
    >>> # Integrate the approximation over the range
    >>> print(interp.integral)
    >>> #    prints 0.6847335267327939
    """
    def __init__(self, tolerance=1.0e-4, ranges=None):
        self._grid = None
        self._options = dict(
                tolerance=tolerance
        )
        self._ranges = ranges

    def fit(self, f):
        """Fit a sparse grid to approximate the function f."""
        if self._ranges is None:
            num_dims = len(inspect.getfullargspec(f)[0])
        else:
            num_dims = len(self._ranges)
        err = None
        def fp(point_matrix):
            nonlocal err
            point_matrix = self._reshape(point_matrix)
            args = []
            for dim in range(num_dims):
                args.append(point_matrix[dim, :])
            try:
                res = f(*args)
            except Exception as e:
                err = e
                return np.zeros((0, 0))
            return res.reshape(1, res.shape[0])
        self._grid = numsg.fit_interpolation(
                fp, 
                self._options,
                1, num_dims)
        if not self._grid:
            raise err

    def evaluate(self, *args):
        """Evaluate the sparse grid interpolation at a given point."""
        assert self._grid is not None
        is_iterable = False
        try:
            it = iter(args[0])
            is_iterable = True
        except TypeError:
            is_iterable = False
        if is_iterable:
            X = np.array(args)
            X = self._inverse_reshape(X)
            return numsg.evaluate_interpolation(self._grid, X)[0, :]
        X = np.array([args]).T
        X = self._inverse_reshape(X)
        return numsg.evaluate_interpolation(self._grid, X)[0, 0]

    def for_each_subgrid(self, f):
        """Iterate over the points of the sparse grid."""
        assert self._grid is not None
        def fp(levels, points):
            f(levels, points)
            return True
        numsg.for_each_subgrid(self._grid, fp)

    @property
    def points(self):
        """Returns the points of the sparse grid interpolation."""
        res = None
        def f(levels, points):
            nonlocal res
            if res is None:
                res = points
            else:
                res = np.hstack((res, points))
        self.for_each_subgrid(f)
        return res

    @property
    def integral(self):
        """Returns the integral of fitted polynomial over the specified k-cell."""
        res = numsg.integrate(self._grid)
        res = res[0]
        if self._ranges is None:
            return res
        for (a, b) in self._ranges:
            res *= (b - a)
        return res

    def _reshape(self, point_matrix):
        ranges = self._ranges
        if ranges is None:
            return point_matrix
        assert point_matrix.shape[0] == len(ranges)
        for dim, (a, b) in enumerate(ranges):
            point_matrix[dim, :] = a + point_matrix[dim, :] * (b - a)
        return point_matrix

    def _inverse_reshape(self, point_matrix):
        ranges = self._ranges
        if ranges is None:
            return point_matrix
        assert point_matrix.shape[0] == len(ranges)
        for dim, (a, b) in enumerate(ranges):
            point_matrix[dim, :] = (point_matrix[dim, :] - a) / (b - a)
        return point_matrix
