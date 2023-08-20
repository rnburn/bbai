import pytest

import numpy as np
from bbai.numeric import SparseGridInterpolator
np.random.seed(0)

def test_we_can_interpolate_a_polynomial():
    def f(x):
        return x**3
    g = SparseGridInterpolator()
    g.fit(f)
    for x in np.linspace(0, 1, 25):
        assert np.isclose(g.evaluate(x), x**3)
    assert np.isclose(g.evaluate(0.5), 0.5**3)
    assert np.isclose(g.integral, 0.25)

def test_we_propagate_exceptions():
    def f(x):
        raise RuntimeError()
    g = SparseGridInterpolator()
    with pytest.raises(Exception):
        g.fit(f)

def test_we_can_interpolate_a_polynomial_over_a_nonstandard_range():
    def f(x):
        return x**3
    g = SparseGridInterpolator(ranges=[(1, 3)])
    g.fit(f)
    for x in np.linspace(1, 3, 25):
        assert np.isclose(g.evaluate(x), x**3)
    assert np.isclose(g.integral, 3**4 / 4.0 - 1 / 4.0)

def test_we_can_interpolate_a_nonpolynomial_function():
    def f(x):
        return np.exp(x)
    g = SparseGridInterpolator(ranges=[(1, 3)])
    g.fit(f)
    for x in np.linspace(1, 3, 25):
        assert np.isclose(g.evaluate(x), f(x))
    assert np.isclose(g.integral, f(3) - f(1))

def test_tolerance_can_be_used_to_control_accuracy():
    def f(x):
        return np.exp(x)
    g1 = SparseGridInterpolator(tolerance=1.0e-3)
    g1.fit(f)
    g2 = SparseGridInterpolator(tolerance=1.0e-7)
    g2.fit(f)

    x = np.random.uniform()
    err1 = np.abs(g1.evaluate(x) - f(x))
    err2 = np.abs(g2.evaluate(x) - f(x))
    assert err2 < err1

def test_we_can_interpolate_a_multivariable_function():
    rng1 = 1, 10
    rng2 = -2, 5
    def f(x, y):
        return 1.0e5 * np.exp(-0.2 * (x - 0.1)**2 - 0.5 * (y - 0.75)**2)
    g = SparseGridInterpolator(tolerance=1.0e-6, ranges=[rng1, rng2])
    g.fit(f)
    N = 25
    xp = np.random.uniform(*rng1, size=N)
    yp = np.random.uniform(*rng2, size=N)
    evals = g.evaluate(xp, yp)
    true_evals = f(xp, yp)
    assert np.isclose(evals, true_evals).all()
    assert np.isclose(g.integral, 281900)

def test_we_can_interpolate_a_function_of_3_variables():
    rng1 = 1, 10
    rng2 = -2, 5
    rng3 = -2, -1
    def f(x, y, z):
        return x **2 * y - y**3 + z * x
    g = SparseGridInterpolator(tolerance=1.0e-6, ranges=[rng1, rng2, rng3])
    g.fit(f)
    N = 25
    xp = np.random.uniform(*rng1, size=N)
    yp = np.random.uniform(*rng2, size=N)
    zp = np.random.uniform(*rng3, size=N)
    evals = g.evaluate(xp, yp, zp)
    true_evals = f(xp, yp, zp)
    assert np.isclose(evals, true_evals).all()
    assert np.isclose(g.integral, 1606.5)

def test_we_can_access_the_points_of_a_grid():
    def f(x, y):
        return np.exp(x - y)
    g = SparseGridInterpolator()
    g.fit(f)
    levels = []
    points = []
    def f(lvls, pts):
        levels.append(lvls)
        points.append(pts)
    g.for_each_subgrid(f)
    points = np.hstack(points)
    assert (points == g.points).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
