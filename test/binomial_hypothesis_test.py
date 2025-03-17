import numpy as np
import pytest
from pytest import approx
import scipy.integrate
from scipy.stats import beta
from scipy.integrate import quad
from bbai.model import BinomialHypothesis2

def validate_hypothesis(hy, a, b):
    eps = 1.0e-8

    # confirm priors integrate to 1
    prior = hy.priors_[0]
    def f(p):
        return prior.pdf(p)
    assert f(0.51) == 0.0
    p1 = quad(f, eps, 0.5)[0]
    assert p1 == pytest.approx(1.0, rel=1.0e-3)

    prior = hy.priors_[1]
    def f(p):
        return prior.pdf(p)
    assert f(0.49) == 0.0
    p2 = quad(f, 0.5, 1-eps)[0]
    assert p2 == pytest.approx(1.0, rel=1.0e-3)

    # check posterior against numerical result
    def like(p):
        return p**a * (1-p)**b

    prior = hy.priors_[0]
    def f(p):
        return like(p) * prior.pdf(p)
    p1 = quad(f, eps, 0.5)[0]

    prior = hy.priors_[1]
    def f(p):
        return like(p) * prior.pdf(p)
    p2 = quad(f, 0.5, 1 - eps)[0]

    Z = p1 + p2
    assert hy.left_ == pytest.approx(p1/Z)
    assert hy.right_ == pytest.approx(p2/Z)

def test_priors():
    a = 3
    b = 2

    # Laplace's prior
    hy = BinomialHypothesis2(a, b, prior='laplace')
    assert hy.priors_[0].pdf(0.123) == pytest.approx(hy.priors_[0].pdf(0.456))
    assert hy.priors_[1].pdf(0.543) == pytest.approx(hy.priors_[1].pdf(0.654))
    validate_hypothesis(hy, a, b)

    # Jeffreys prior
    hy = BinomialHypothesis2(a, b, prior='jeffreys')

    p1 = 0.123
    p2 = 0.456
    prior = hy.priors_[0]
    expected = p1**-.5 * (1 - p1)**-.5
    expected /= p2**-0.5 * (1 - p2)**-.5
    assert prior.pdf(p1) / prior.pdf(p2) == pytest.approx(expected)

    p1 = 0.567
    p2 = 0.678
    prior = hy.priors_[1]
    expected = p1**-.5 * (1 - p1)**-.5
    expected /= p2**-0.5 * (1 - p2)**-.5
    assert prior.pdf(p1) / prior.pdf(p2) == pytest.approx(expected)

    validate_hypothesis(hy, a, b)

    # integral prior
    hy = BinomialHypothesis2(a, b, prior='integral')
    validate_hypothesis(hy, a, b)

def test_integral_prior():
    eps = 1.0e-8
    dist1 = beta(0.5 + 1, 0.5)
    dist2 = beta(0.5, 0.5 + 1)
    hy = BinomialHypothesis2(prior='integral')

    def f(t):
        return t * hy.priors_[0].pdf(t)
    m1 = quad(f, eps, 0.5)[0]

    t = 0.543
    val = dist1.pdf(t) * m1 / (1 - dist1.cdf(0.5)) + dist2.pdf(t) * (1 - m1) / (1 - dist2.cdf(0.5))
    assert val == pytest.approx(hy.priors_[1].pdf(t))

    def f(t):
        return t * hy.priors_[1].pdf(t)
    m2 = quad(f, 0.5, 1 - eps)[0]

    t = 0.123
    val = dist1.pdf(t) * m2 / dist1.cdf(0.5) + dist2.pdf(t) * (1 - m2) / dist2.cdf(0.5)
    assert val == pytest.approx(hy.priors_[0].pdf(t))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
