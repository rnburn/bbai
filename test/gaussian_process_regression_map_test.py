import pytest
import numpy as np
from scipy.integrate import quadrature, nquad
from sklearn.preprocessing import StandardScaler
import bbai.gp
from bbai.gp import GaussianProcessRegressionMAP
from bbai.gp.reference_equations import *

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

@pytest.fixture
def dataset2():
    def target_generator(rng, X):
        target = 0.5 + np.sin(3 * X)
        target += rng.normal(0, 0.3, size=target.shape)
        return target.squeeze()
    rng = np.random.RandomState(0)
    Z = rng.uniform(0, 5, size=20).reshape(-1, 1)
    y = target_generator(rng, Z)
    X = np.ones((20, 1))
    return Z, X, y

@pytest.fixture
def dataset3():
    def target_generator(rng, X):
        target = -2.0 + 0.5*X**2 + np.sin(3 * X)
        target += rng.normal(0, 0.3, size=target.shape)
        return target.squeeze()
    rng = np.random.RandomState(0)
    Z = rng.uniform(0, 5, size=20).reshape(-1, 1)
    y = target_generator(rng, Z)
    X = np.hstack((Z**2, np.ones((20, 1))))
    return Z, X, y

# Test against a data set with no regressors
def test_interpolation1(dataset1):
    Z, X, y = dataset1
    model = GaussianProcessRegressionMAP(
            kernel=bbai.gp.RbfCovarianceFunction(),
            length0=2.0,
            noise_ratio0=0.05)
    model.fit(Z, y)
    theta, eta = model.length_, model.noise_ratio_

    cvfn = RbfCovarianceFunction()

    def f(theta, eta):
        ws = compute_workspace(X, Z, cvfn, theta, eta)
        return compute_posterior(X, Z, y, cvfn, theta, eta)

    # check that we found a local optimum of the posterior by stepping to neighboring 
    # hyperparameters and verifying that the posterior doesn't increasing
    f_opt = f(theta, eta)
    d = 1.0e-3

    assert f_opt > f(theta + d, eta)
    assert f_opt > f(theta - d, eta)

    assert f_opt > f(theta, eta + d)
    assert f_opt > f(theta, eta - d)

    # check the predicted mean against the equations from Ren, Sun, and He.
    rng = np.random.RandomState(123)
    num_test = 50
    Z_test = rng.uniform(0, 5, size=num_test).reshape(-1, 1)
    X_test = np.zeros((num_test, 0))
    expected_pred = compute_prediction_mean(
            X, Z, y, X_test, Z_test, cvfn, theta, eta)
    pred, pdf = model.predict(Z_test, with_pdf=True)
    assert np.allclose(pred, expected_pred, rtol=1.0e-7)

    # check that the prediction probability density function values are correct
    # up to a normalization constant
    pdf_p = compute_unormalized_pdf(
            X, Z, y, X_test, Z_test, cvfn, theta, eta)
    y1 = pred
    y2 = pred * 1.1
    r = pdf(y2) / pdf(y1)
    r_p = pdf_p(y2) / pdf_p(y1)
    assert np.isclose(r, r_p)

    # a PDF distribution integrates to 1
    Z_test = rng.uniform(0, 5, size=1).reshape(-1, 1)
    pred, pdf = model.predict(Z_test, with_pdf=True)
    def f(tx):
        return [pdf(ti) for ti in tx]
    p_int, _ = quadrature(f, pred[0]-5.0, pred[0] + 5.0, maxiter=100)
    assert np.isclose(p_int, 1.0)

    Z_test = rng.uniform(0, 5, size=2).reshape(-1, 1)
    pred, pdf = model.predict(Z_test, with_pdf=True)
    def f(u, v):
        return pdf([u, v])
    a = (pred[0]-5.0, pred[0]+5.0)
    b = (pred[1]-5.0, pred[1]+5.0)
    p_int, _ = nquad(f, (a, b), opts=dict(limit=100))
    assert np.isclose(p_int, 1.0)


# Test against a data set with a single constant regressors
def test_interpolation2(dataset2):
    Z, X, y = dataset2
    model = GaussianProcessRegressionMAP(
            kernel=bbai.gp.RbfCovarianceFunction(),
            length0=2.0,
            noise_ratio0=0.05)
    model.fit(Z, y, design_matrix = X)
    theta, eta = model.length_, model.noise_ratio_

    cvfn = RbfCovarianceFunction()

    def f(theta, eta):
        ws = compute_workspace(X, Z, cvfn, theta, eta)
        return compute_posterior(X, Z, y, cvfn, theta, eta)

    # check that we found a local optimum of the posterior by stepping to neighboring 
    # hyperparameters and verifying that the posterior doesn't increasing
    f_opt = f(theta, eta)
    d = 1.0e-3

    assert f_opt > f(theta + d, eta)
    assert f_opt > f(theta - d, eta)

    assert f_opt > f(theta, eta + d)
    assert f_opt > f(theta, eta - d)

    # check the predicted mean against the equations from Ren, Sun, and He.
    rng = np.random.RandomState(123)
    num_test = 50
    Z_test = rng.uniform(0, 5, size=num_test).reshape(-1, 1)
    X_test = np.ones((num_test, 1))
    expected_pred = compute_prediction_mean(
            X, Z, y, X_test, Z_test, cvfn, theta, eta)
    pred, pdf = model.predict(Z_test, design_matrix=X_test, with_pdf=True)
    assert np.allclose(pred, expected_pred, rtol=1.0e-7)

    # check that the prediction probability density function values are correct
    # up to a normalization constant
    pdf_p = compute_unormalized_pdf(
            X, Z, y, X_test, Z_test, cvfn, theta, eta)
    y1 = pred
    y2 = pred * 1.1
    r = pdf(y2) / pdf(y1)
    r_p = pdf_p(y2) / pdf_p(y1)
    assert np.isclose(r, r_p)

    # a PDF distribution integrates to 1
    Z_test = rng.uniform(0, 5, size=1).reshape(-1, 1)
    X_test = np.ones((1, 1))
    pred, pdf = model.predict(Z_test, design_matrix=X_test, with_pdf=True)
    def f(tx):
        return [pdf(ti) for ti in tx]
    p_int, _ = quadrature(f, pred[0]-5.0, pred[0] + 5.0, maxiter=100)
    assert np.isclose(p_int, 1.0)

    Z_test = rng.uniform(0, 5, size=2).reshape(-1, 1)
    X_test = np.ones((2, 1))
    pred, pdf = model.predict(Z_test, design_matrix=X_test, with_pdf=True)
    def f(u, v):
        return pdf([u, v])
    a = (pred[0]-5.0, pred[0]+5.0)
    b = (pred[1]-5.0, pred[1]+5.0)
    p_int, _ = nquad(f, (a, b), opts=dict(limit=100))
    assert np.isclose(p_int, 1.0)

# Test against a data set with regressors
def test_interpolation3(dataset3):
    Z, X, y = dataset3
    model = GaussianProcessRegressionMAP(
            kernel=bbai.gp.RbfCovarianceFunction(),
            length0=2.0,
            noise_ratio0=0.05)
    model.fit(Z, y, design_matrix = X)
    theta, eta = model.length_, model.noise_ratio_

    cvfn = RbfCovarianceFunction()

    def f(theta, eta):
        ws = compute_workspace(X, Z, cvfn, theta, eta)
        return compute_posterior(X, Z, y, cvfn, theta, eta)

    # check that we found a local optimum of the posterior by stepping to neighboring 
    # hyperparameters and verifying that the posterior doesn't increasing
    f_opt = f(theta, eta)
    d = 1.0e-3
    
    assert f_opt > f(theta + d, eta)
    assert f_opt > f(theta - d, eta)

    assert f_opt > f(theta, eta + d)
    assert f_opt > f(theta, eta - d)

    # check the predicted mean against the equations from Ren, Sun, and He.
    rng = np.random.RandomState(123)
    num_test = 50
    Z_test = rng.uniform(0, 5, size=num_test).reshape(-1, 1)
    X_test = np.hstack((Z_test**2, np.ones((num_test, 1))))
    expected_pred = compute_prediction_mean(
            X, Z, y, X_test, Z_test, cvfn, theta, eta)
    pred, pdf = model.predict(Z_test, design_matrix=X_test, with_pdf=True)
    assert np.allclose(pred, expected_pred, rtol=1.0e-7)

    # check that the prediction probability density function values are correct
    # up to a normalization constant
    pdf_p = compute_unormalized_pdf(
            X, Z, y, X_test, Z_test, cvfn, theta, eta)
    y1 = pred
    y2 = pred * 1.1
    r = pdf(y2) / pdf(y1)
    r_p = pdf_p(y2) / pdf_p(y1)
    assert np.isclose(r, r_p)

    # a PDF distribution integrates to 1
    Z_test = rng.uniform(0, 5, size=1).reshape(-1, 1)
    X_test = np.hstack((Z_test**2, np.ones((1, 1))))
    pred, pdf = model.predict(Z_test, design_matrix=X_test, with_pdf=True)
    def f(tx):
        return [pdf(ti) for ti in tx]
    p_int, _ = quadrature(f, pred[0]-5.0, pred[0] + 5.0, maxiter=100)
    assert np.isclose(p_int, 1.0)

    Z_test = rng.uniform(0, 5, size=2).reshape(-1, 1)
    X_test = np.hstack((Z_test**2, np.ones((2, 1))))
    pred, pdf = model.predict(Z_test, design_matrix=X_test, with_pdf=True)
    def f(u, v):
        return pdf([u, v])
    a = (pred[0]-5.0, pred[0]+5.0)
    b = (pred[1]-5.0, pred[1]+5.0)
    p_int, _ = nquad(f, (a, b), opts=dict(limit=100))
    assert np.isclose(p_int, 1.0)

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
