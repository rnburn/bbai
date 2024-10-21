import pytest
import numpy as np
from bbai.model import BoundedNormal
import scipy

def test_bounded_normal_model():
    a = -0.123
    b = 4.5
    model = BoundedNormal(a, b)
    y = [0, 3, 4]
    model.fit(y)

    # PDF goes to zero at the endpoints
    assert model.pdf(a) == 0.0
    assert model.pdf(b) == 0.0

    # PDF is zero outside of the boundaries
    assert model.pdf(a - 0.01) == 0.0
    assert model.pdf(b + 0.01) == 0.0

    # PDF integrates to 1
    S = 0
    N = 2000
    step = (b - a) / N
    for i in range(N):
        S += model.pdf(a + i * step) * step
    assert S == pytest.approx(1.0)

    # cdf matches up with numeric integration
    t = 1.23
    S = 0
    step = (t - a) / N
    for i in range(N):
        S += model.pdf(a + i * step) * step
    assert S == pytest.approx(model.cdf(t), rel=1.0e-3)

    # cdf and ppf match up
    median = model.ppf(0.5)
    assert model.cdf(median) == pytest.approx(0.5, rel=1.0e-3)

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

