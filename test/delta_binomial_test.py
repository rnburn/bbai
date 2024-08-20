import pytest
import numpy as np
from bbai.model import DeltaBinomialModel
import scipy

def test_delta_binomial_model():
    for prior in ['uniform', 'jeffreys', 'reference']:
        a1 = 2
        b1 = 1
        a2 = 3
        b2 = 4
        model = DeltaBinomialModel(prior=prior)
        model.fit(a1, b1, a2, b2)

        # cdf and pdf match up
        t = 0.123
        h = 1.0e-6
        d = (model.cdf(t + h) - model.cdf(t)) / h
        expected = model.pdf(t)
        assert d == pytest.approx(expected, rel=1.0e-4)

        d = (model.cdf(-t + h) - model.cdf(-t)) / h
        expected = model.pdf(-t)
        assert d == pytest.approx(expected, rel=1.0e-4)

        # end point behavior
        assert model.cdf(-1.0) == 0.0
        assert model.cdf(1.0) == 1.0

        # prior integrates to 1
        delta = 1.0e-3
        tot = 0.0
        for t in np.arange(-0.9999, 0.9999, delta):
            tot += model.prior_pdf(t)
        assert tot * delta == pytest.approx(1.0, rel=1.0e-2)

        # marginal prior matches up with pdf
        delta = 1.0e-6
        tot = 0.0
        theta = 0.123
        val = model.prior_pdf(theta)
        eps = 1.0e-7
        def f(x):
          return model.prior_pdf(theta, x)
        res = scipy.integrate.quad(f, eps, 1.0 - theta - eps)
        assert res[0] == pytest.approx(val, rel=1.0e-3)

        # symmetry
        t = 0.123
        model.fit(a1, b1, a2, b2)
        val1 = model.cdf(t)
        model.fit(b1, a1, b2, a2)
        val2 = 1 - model.cdf(-t)
        assert val1 == pytest.approx(val2)

def test_precompute():
    a1 = 3
    b1 = 2
    a2 = 1
    b2 = 2

    # uniform
    model = DeltaBinomialModel(prior='uniform')
    model.fit(a1, b1, a2, b2)
    val = model.pdf(0.123)
    assert val == pytest.approx(1.38587)

    # Jeffreys
    model = DeltaBinomialModel(prior='jeffreys')
    model.fit(a1, b1, a2, b2)
    val = model.pdf(0.123)
    assert val == pytest.approx(1.22913, rel=1.0e-4)

    # Reference
    model = DeltaBinomialModel(prior='reference')
    model.fit(a1, b1, a2, b2)
    val = model.pdf(0.123)
    assert val == pytest.approx(1.08699, rel=1.0e-4)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

