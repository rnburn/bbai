import pytest
import numpy as np
from scipy.integrate import quadrature, nquad
from sklearn.preprocessing import StandardScaler
import bbai.gp
from bbai.gp import BayesianGaussianProcessRegression, Power1CovarianceFunction
from bbai.gp.reference_equations import *

np.random.seed(0)

@pytest.fixture
def dataset1():
    def target_generator(rng, X):
        target = 0.5 + np.sin(3 * X)
        target += rng.normal(0, 0.3, size=target.shape)
        return target.squeeze()
    rng = np.random.RandomState(0)
    Z = rng.uniform(0, 5, size=20).reshape(-1, 1)
    X = np.zeros((20, 0))
    y = target_generator(rng, Z)
    return Z, X, y

def test_bayesian_model(dataset1):
    Z, X, y = dataset1
    model = BayesianGaussianProcessRegression(
            kernel=bbai.gp.RbfCovarianceFunction(),
            length0=2.0,
            noise_ratio0=0.05)
    model.fit(Z, y)
    theta, eta = model.length_mode_, model.noise_ratio_mode_

    cvfn = RbfCovarianceFunction()

    def f(theta, eta):
        ws = compute_workspace(X, Z, cvfn, theta, eta)
        return compute_posterior(X, Z, y, cvfn, theta, eta) + np.log(theta) + np.log(eta)

    # check that we found a local optimum of the posterior by stepping to neighboring 
    # hyperparameters and verifying that the posterior doesn't increasing
    f_opt = f(theta, eta)
    d = 1.0e-3

    assert f_opt > f(theta + d, eta)
    assert f_opt > f(theta - d, eta)

    assert f_opt > f(theta, eta + d)
    assert f_opt > f(theta, eta - d)

    # Verify medians
    assert np.isclose(model.marginal_length_.cdf(model.length_median_), 0.5)
    assert np.isclose(model.marginal_noise_ratio_.cdf(model.noise_ratio_median_), 0.5)

    # Verify marginal properties for length and noise ratio
    marginals = [
            model.marginal_log_length_,
            model.marginal_log_noise_ratio_,
            model.marginal_length_,
            model.marginal_noise_ratio_,
    ]
    for marginal in marginals:
        integral, _ = quadrature(lambda t: marginal.pdf(t), 
                         marginal.bracket_low_,
                         marginal.bracket_high_,
                         vec_func=False,
                         maxiter=1000) 
        assert np.isclose(integral, 1.0, rtol=1.0e-4)
        mid = marginal.ppf(0.5)
        integral, _ = quadrature(lambda t: marginal.pdf(t), 
                         marginal.bracket_low_,
                         mid,
                         vec_func=False,
                         maxiter=1000) 
        assert np.isclose(integral, marginal.cdf(mid), rtol=1.0e-5)
        assert np.isclose(integral, 0.5)

    # Verify marginal properties for noise
    marginal = model.marginal_sigma2_signal_
    integral, _ = quadrature(lambda t: marginal.pdf(t), 
                     1.0e-5,
                     100,
                     vec_func=False,
                     maxiter=1000) 
    assert np.isclose(integral, 1.0, rtol=1.0e-3)
    integral, _ = quadrature(lambda t: marginal.pdf(t), 
                     1.0e-5,
                     10,
                     vec_func=False,
                     maxiter=1000) 
    assert np.isclose(integral, marginal.cdf(10))

    # Verify marginal properties for a prediction
    Z_test = np.array([[0.123]])
    X_test = np.zeros((1, 0))
    _, pdfs = model.predict(Z_test, X_test, with_pdf=True) 
    marginal = pdfs[0]
    integral, _ = quadrature(lambda t: marginal.pdf(t), 
                     -5,
                     5,
                     vec_func=False,
                     maxiter=1000) 
    assert np.isclose(integral, 1.0)
    integral, _ = quadrature(lambda t: marginal.pdf(t), 
                     -5,
                     0.75,
                     vec_func=False,
                     maxiter=1000) 
    assert np.isclose(integral, marginal.cdf(0.75))
    assert np.isclose(marginal.ppf(marginal.cdf(0.75)), 0.75)

def test_invalid_data(dataset1):
    Z, X, y = dataset1
    X = np.zeros((Z.shape[0], 1))
    model = BayesianGaussianProcessRegression(
            kernel=bbai.gp.RbfCovarianceFunction(),
            length0=2.0,
            noise_ratio0=0.05)
    with pytest.raises(RuntimeError):
        model.fit(Z, y, X)

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))

