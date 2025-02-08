import numpy as np
import pytest
from pytest import approx
from bbai.glm import Lasso, RandomRegressionDataset, LassoKKT 

def test_1x1():
    X = np.array([[123]])
    y = np.array([-0.5])

    lambda0 = np.abs(X[0, 0] * y[0])

    # if lambda > lambda0, the regressor is zero
    model = Lasso(lambda0, fit_intercept=False)
    model.fit(X, y)
    assert model.coef_ == np.array([0.0])

    # if lambda == 0, we get the least squares solution
    model = Lasso(0.0, fit_intercept=False)
    model.fit(X, y)
    assert model.coef_[0] == approx(y[0] / X[0, 0])

    # KKT conditions are satisfied
    lda = lambda0 / 2
    model = Lasso(lda, fit_intercept=False)
    model.fit(X, y)
    h = np.dot(X.T, y - np.dot(X, model.coef_))
    assert h[0] == approx(lda * np.sign(model.coef_[0]))

    # we can fit the beta path
    model = Lasso(0.0, fit_intercept=False, fit_beta_path=True)
    model.fit(X, y)
    assert model.beta_path_ == approx(np.array([[0.0, lambda0], [y[0] / X[0, 0], 0.0]]))

def test_2x1():
    X = np.array([[11.3], [-22.4]])
    y = np.array([-0.5, 1.321])

    lambda0 = np.abs(np.dot(X.T, y)[0])
    beta_ols = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))[0]

    # if lambda > lambda0, the regressor is zero
    model = Lasso(lambda0, fit_intercept=False)
    model.fit(X, y)
    assert model.coef_ == np.array([0.0])

    # if lambda == 0, we get the least squares solution
    model = Lasso(0.0, fit_intercept=False)
    model.fit(X, y)
    assert model.coef_[0] == approx(beta_ols)

    # KKT conditions are satisfied
    lda = lambda0 / 2
    model = Lasso(lda, fit_intercept=False)
    model.fit(X, y)
    h = np.dot(X.T, y - np.dot(X, model.coef_))
    assert h[0] == approx(lda * np.sign(model.coef_[0]))

    # we can fit the beta path
    model = Lasso(0.0, fit_intercept=False, fit_beta_path=True)
    model.fit(X, y)
    assert model.beta_path_ == approx(np.array([[0.0, lambda0], [beta_ols, 0.0]]))
    
def test_random():
    np.random.seed(3)

    # we can fit random data sets
    for _ in range(10):
        ds = RandomRegressionDataset(n=10, p=5)
        lda = np.random.uniform() * ds.lambda_max
        model = Lasso(lda, fit_intercept=False)
        model.fit(ds.X, ds.y)
        kkt = LassoKKT(ds.X, ds.y, lda, model.beta_, with_intercept=False)
        assert kkt.within(1.0e-6)
        xp = np.random.uniform(size=5)
        expected = np.dot(xp, model.coef_)
        assert (model.predict(xp) == expected).all()

    # we can fit random data sets with intercept
    for _ in range(10):
        ds = RandomRegressionDataset(n=10, p=5, with_intercept=True)
        lda = np.random.uniform() * ds.lambda_max
        model = Lasso(lda, fit_intercept=True)
        model.fit(ds.X, ds.y)
        kkt = LassoKKT(ds.X, ds.y, lda, model.beta_, with_intercept=True)
        assert kkt.within(1.0e-6)
        xp = np.random.uniform(size=5)
        expected = model.intercept_ + np.dot(xp, model.coef_)
        assert (model.predict(xp) == expected).all()

    # we can fit data sets where (num_regressors) > (num_data)
    for _ in range(10):
        ds = RandomRegressionDataset(n=5, p=6, with_intercept=True)
        lda = np.random.uniform() * ds.lambda_max
        model = Lasso(lda, fit_intercept=True)
        model.fit(ds.X, ds.y)
        kkt = LassoKKT(ds.X, ds.y, lda, model.beta_, with_intercept=True)
        assert kkt.within(1.0e-6)

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
