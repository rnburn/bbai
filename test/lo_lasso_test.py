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

def exercise_early_exit(ds, loo_mode, threshold):
    fit_intercept = ds.with_intercept
    m_full = Lasso(loo_mode=loo_mode, fit_intercept=fit_intercept)
    m_early = Lasso(loo_mode=loo_mode, early_exit_threshold=threshold, fit_intercept=fit_intercept)

    m_full.fit(ds.X, ds.y)
    m_early.fit(ds.X, ds.y)
    assert m_early.loo_mse_ >= m_full.loo_mse_
    assert m_early.s_ <= m_full.s_

    if loo_mode == 't':
        t = m_early.t_ / 1.2
        assert m_early.loo_squared_error_.evaluate_t(t) == m_full.loo_squared_error_.evaluate_t(t)
    else:
        lda = m_early.lambda_*1.1
        assert m_early.loo_squared_error_.evaluate_lambda(lda) == m_full.loo_squared_error_.evaluate_lambda(lda)

    segments = m_early.loo_squared_error_.segments_
    x, a0, a1, a2 = segments[:, -1]
    if not x < np.inf:
        return

    mse = a0 + a1 * x + a2*x**2
    mse /= len(ds.y)
    t = (mse - m_early.loo_mse_) / m_early.loo_mse_
    assert t > threshold

def exercise_dataset(ds):
    exercise_dataset_lambda(ds)
    exercise_dataset_t(ds)
    for cutoff in [0.0, 0.01, 0.05]:
        exercise_early_exit(ds, 't', cutoff)
        exercise_early_exit(ds, 'lambda', cutoff)

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

