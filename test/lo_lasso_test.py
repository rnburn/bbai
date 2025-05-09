import numpy as np
import pytest
from pytest import approx
from bbai.glm import Lasso, RandomRegressionDataset, LassoGridCv, \
                     evaluate_lo_errors_slow, evaluate_lo_cost_slow, LassoKKT

def exercise_dataset_lambda(ds):
    fit_intercept = ds.with_intercept

    def lo_slow(X, y, lda):
        model = Lasso(lda, fit_intercept=fit_intercept)
        return evaluate_lo_cost_slow(X, y, model)

    model = Lasso(fit_intercept=fit_intercept)
    model.fit(ds.X, ds.y)

    # check KKT conditions at lambda_opt
    kkt = LassoKKT(ds.X, ds.y, model.lambda_, model.beta_, with_intercept=fit_intercept)
    kkt.within(1.0e-6)

    # check that lambda_opt gives the right CV
    cost_opt = model.loo_mse_ * len(ds.y)
    assert cost_opt == pytest.approx(lo_slow(ds.X, ds.y, model.lambda_))

    # check the cost function across a grid
    grid = LassoGridCv(lo_slow, ds.X, ds.y, ds.lambda_max)
    for lda, cv in zip(grid.lambdas, grid.cvs):
        assert cv == pytest.approx(model.loo_squared_error_.evaluate_lambda(lda))

    # check that no grid cv gives a better value than lambda opt
    assert cost_opt <= grid.cv_min + 1.0e-6

def exercise_dataset_t(ds):
    fit_intercept = ds.with_intercept

    def lo_slow(X, y, t):
        model = Lasso(t=t, fit_intercept=fit_intercept)
        return evaluate_lo_cost_slow(X, y, model)

    model = Lasso(fit_intercept=fit_intercept, loo_errors=True, loo_mode='t')
    model.fit(ds.X, ds.y)

    # check KKT conditions at lambda_opt
    kkt = LassoKKT(ds.X, ds.y, model.lambda_, model.beta_, with_intercept=fit_intercept)
    kkt.within(1.0e-6)

    # check that lambda_opt gives the right CV
    cost_opt = model.loo_mse_ * len(ds.y)
    assert cost_opt == pytest.approx(lo_slow(ds.X, ds.y, model.t_))

    # check the cost function across a grid
    grid = LassoGridCv(lo_slow, ds.X, ds.y, model.t_max_)
    for t, cv in zip(grid.lambdas, grid.cvs):
        assert cv == pytest.approx(model.loo_squared_error_.evaluate_t(t))

    # we can evaluate individual errors
    errs = model.loo_errors_.evaluate_t(model.t_)
    m = Lasso(t=model.t_, fit_intercept=fit_intercept)
    errs_p = evaluate_lo_errors_slow(ds.X, ds.y, m)
    for e, ep in zip(errs, errs_p):
        assert e == pytest.approx(ep)

    # check that no grid cv gives a better value than lambda opt
    assert cost_opt <= grid.cv_min + 1.0e-6

def exercise_dataset(ds):
    exercise_dataset_lambda(ds)
    exercise_dataset_t(ds)

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

