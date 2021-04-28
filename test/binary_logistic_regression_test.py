import pytest
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression as BaselineLogisticRegression
from bbai.glm import LogisticRegression

@pytest.fixture
def breast_cancer_dataset():
    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    return X, y

def test_aloocv_optimization(breast_cancer_dataset):
    X, y = breast_cancer_dataset
    model = LogisticRegression(tolerance=1.0e-6)
    model.fit(X, y)
    assert model.C_ == pytest.approx(0.6655139682151202)

def test_against_baseline(breast_cancer_dataset):
    X, y = breast_cancer_dataset
    C = 0.5
    model = LogisticRegression(tolerance=1.0e-6, C=C)
    model.fit(X, y)

    model_p = BaselineLogisticRegression(C=C, tol=1.0e-6)
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
