import pytest
import numpy as np
from bbai.glm import BayesianLogisticRegression1
import scipy

def log_prior(x, w):
    s = 0
    for xi in x:
        p = 1 / (1 + np.exp(-xi * w))
        q = 1 - p
        s += xi**2 * p * q
    return 0.5 * np.log(s)

def log_like(x, y, w):
    res = 0
    for xi, yi in zip(x, y):
        mult = 2 * yi - 1
        res += np.log(1 / (1 + np.exp(-xi * mult * w)))
    return res

def log_post(x, y, w):
    return log_like(x, y, w) + log_prior(x, w)

def test_bayesian_logistic_regression1():
    x = [1.2, -3.4, 5.1]
    y = [1.0, 1.0, 0.0]
    model = BayesianLogisticRegression1()
    model.fit(x, y)

    # test relative pdfs
    w1 = 0.123
    w2 = 0.321
    relative_pdf = model.pdf(w2) / model.pdf(w1)
    expected = np.exp(log_post(x, y, w2) - log_post(x, y, w1))
    assert relative_pdf == pytest.approx(expected)

    # pdf integrates to 1
    N = 1000
    wmin = -10
    wmax = 10
    step = (wmax - wmin) / N
    integral = 0
    for i in range(N):
        integral += model.pdf(wmin + i * step) * step
    assert integral == pytest.approx(1.0)

    # cdf matches the integral of pdf
    N = 2000
    wmin = -10
    wmax = 0.123
    step = (wmax - wmin) / N
    integral = 0
    for i in range(N):
        integral += model.pdf(wmin + i * step) * step
    assert integral == pytest.approx(model.cdf(wmax), rel=1.0e-3)

    # cdf and ppf match up
    assert 0.123 == pytest.approx(model.ppf(model.cdf(0.123)), rel=1.0e-3)

    # prediction pdf matches the transformed posterior
    xp = 0.67
    pred = model.predict(xp)
    def phi(p):
        return np.log(p / (1 - p)) / xp
    eps = 1.0e-9
    p = 0.123
    w = phi(p)
    phi_dot = (phi(p + eps) - phi(p)) / eps
    assert pred.pdf(p) == pytest.approx(model.pdf(w) * phi_dot)

    # prediction cdf matches w cdf
    p = 0.123
    xp = 0.67
    pred = model.predict(xp)
    w = np.log(p / (1 - p)) / xp
    assert p == pytest.approx(1 / (1 + np.exp(-xp * w)))
    median = pred.cdf(p)
    expected = model.cdf(w)
    assert median == pytest.approx(expected)

    # prediction cdf and ppf match up
    xp = 0.67
    pred = model.predict(xp)
    p = pred.ppf(0.123)
    assert pred.cdf(p) == pytest.approx(0.123)
    
    # we properly handle endpoints
    assert pred.pdf(0.0) == 0.0
    assert pred.pdf(1.0) == 0.0

    # prediction pdf integrates to 1
    N = 1000
    pred = model.predict(0.123)
    pmin = 0.0
    pmax = 1.0
    step = (pmax - pmin) / N
    integral = 0
    for i in range(N):
        integral += pred.pdf(pmin + i * step) * step
    assert integral == pytest.approx(1.0)

def test_bayesian_logistic_regression1_separable():
    x = [1.2, -3.4, 5.1]
    y = [1.0, 0.0, 1.0]
    model = BayesianLogisticRegression1()
    model.fit(x, y)

    xp = 0.67
    pred = model.predict(xp)
    
    assert pred.pdf(0.0) == 0.0
    assert pred.pdf(1.0) == np.inf

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
