import numpy as np
import pytest
from pytest import approx
from bbai.glm import LassoAlo, Lasso, RandomRegressionDataset, LassoGridCv, \
                     evaluate_alo_cost_slow, evaluate_lo_cost_slow, LassoKKT

def exercise_dataset(ds):
    fit_intercept = ds.with_intercept

    def alo_slow(X, y, lda):
        n, p = X.shape
        m = Lasso(lambda_=lda, fit_intercept=fit_intercept)
        m.fit(X, y)
        kkt = LassoKKT(X, y, lda, m.beta_, with_intercept=fit_intercept)
        assert kkt.within(1.0e-6, 1.0e-6)
        active = [j for j in range(p) if m.coef_[j] != 0]
        if len(active) == len(y):
            return np.nan
        s = np.sign(m.coef_[active])
        if fit_intercept:
            X = np.hstack((np.ones((n, 1)), X))
            active = [0] + list(1 + np.array(active))
            s = [0] + list(s)
        return evaluate_alo_cost_slow(X, y, active, s, lda)

    model = LassoAlo(fit_intercept=fit_intercept)
    model.fit(ds.X, ds.y)

    # check KKT conditions at lambda_opt
    kkt = LassoKKT(ds.X, ds.y, model.lambda_, model.beta_, with_intercept=fit_intercept)
    assert kkt.within(1.0e-6, 1.0e-6)

    # check that lambda_opt gives the right CV
    cost_opt = model.loo_mse_ * len(ds.y)
    assert cost_opt == pytest.approx(alo_slow(ds.X, ds.y, model.lambda_))

    # check the cost function across a grid
    grid = LassoGridCv(alo_slow, ds.X, ds.y, ds.lambda_max + .5)
    for lda, cv in zip(grid.lambdas, grid.cvs):
        expected = model.loo_squared_error_.evaluate_lambda(lda)
        if np.isnan(cv):
            continue
        assert cv == pytest.approx(model.loo_squared_error_.evaluate_lambda(lda))

    # check that no grid cv gives a better value than lambda opt
    assert cost_opt <= grid.cv_min + 1.0e-6

def test_random():
    np.random.seed(0)

    # test small data sets
    for _ in range(10):
        ds = RandomRegressionDataset(n=2, p=1, with_intercept=False)
        exercise_dataset(ds)

    # test larger data sets
    for _ in range(10):
        ds = RandomRegressionDataset(n=10, p=5, with_intercept=False)
        exercise_dataset(ds)

    # test data sets with intercept
    for _ in range(10):
        ds = RandomRegressionDataset(n=10, p=5, with_intercept=True)
        exercise_dataset(ds)

    # test data sets where (num_regressors) > (num_data)
    for _ in range(10):
        ds = RandomRegressionDataset(n=5, p=6)
        exercise_dataset(ds)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
