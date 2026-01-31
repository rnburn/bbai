import numpy as np
import pytest
from pytest import approx
from bbai.graphics import BezierPath


def test_linear():
    def f(t):
        return 3 * t
    path = BezierPath(dst_xmin=0, dst_xmax=1, dst_ymin=-1, dst_ymax=1).fit(f, -10, 10).segments_
    assert(len(path) == 2)

    assert(path[0][0] == pytest.approx(0.0))
    assert(path[0][1] == pytest.approx(-1.0))

    assert(path[1][0] == pytest.approx(1.0))
    assert(path[1][1] == pytest.approx(1.0))

def test_quadratic():
    def f(t):
        return 3 * t - t*t
    path = BezierPath(dst_xmin=0, dst_xmax=1, dst_ymin=-1, dst_ymax=1).fit(f, -10, 10).segments_
    assert(len(path) == 2)

def test_parametric():
    def fx(t):
        return 3 * t - t*t
    def fy(t):
        return t - t**3
    path = BezierPath(dst_xmin=0, dst_xmax=1, dst_ymin=-1, dst_ymax=1).fit(fx, fy, -10, 10).segments_
    assert(len(path) == 2)

def test_gaussian():
    def f(t):
        return np.exp(-t*t)
    path = BezierPath(dst_xmin=0, dst_xmax=1, dst_ymin=-1, dst_ymax=1).fit(f, -10, 10).segments_
    assert(len(path) > 2)

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
