import pytest
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as BaselineLogisticRegression
from bbai.glm import LogisticRegression

@pytest.fixture
def iris1_dataset():
    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    return X, y

@pytest.fixture
def iris2_dataset():
    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    return X, y

def test_aloocv_optimization(iris1_dataset):
    X, y = iris1_dataset
    model = LogisticRegression(tolerance=1.0e-6)
    model.fit(X, y)
    assert model.C_ == pytest.approx(43.70957582241037)

def test_m1_case(iris1_dataset):
    X, y = iris1_dataset
    model = LogisticRegression(tolerance=1.0e-6, active_classes='m1')
    model.fit(X, y)
    pred = model.predict_proba(X)
    assert pred.shape == (len(y), 3)

def test_against_baseline(iris2_dataset):
    X, y = iris2_dataset
    C = 0.5

    model = LogisticRegression(tolerance=1.0e-6, C=C, fit_intercept=False)
    model.fit(X, y)

    # sklearn regularizes the bias, so compare with fit_intercept=False
    model_p = BaselineLogisticRegression(C=C, tol=1.0e-6, fit_intercept=False)
    model_p.fit(X, y)

    np.testing.assert_array_almost_equal(model.coef_, model_p.coef_, decimal=3)
    np.testing.assert_array_almost_equal(model.intercept_, model_p.intercept_, decimal=3)

    pred = model.predict_proba(X)
    pred_p = model_p.predict_proba(X)
    np.testing.assert_array_almost_equal(pred, pred_p, decimal=3)

    pred = model.predict(X)
    pred_p = model_p.predict(X)
    np.testing.assert_equal(pred, pred_p)

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
