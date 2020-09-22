import unittest
import numpy as np
import peak_engines
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import scipy
from scipy.special import expit

def fit_logistic_regression(X, y, C):
    model = LogisticRegression(solver='lbfgs', C=C)
    model.fit(X, y)
    return np.array(list(model.coef_[0]) + list(model.intercept_))

def compute_hessian(p_vector, X, alpha):
    n, k = X.shape
    a_vector = np.sqrt((1 - p_vector)*p_vector)
    R = scipy.linalg.qr(a_vector.reshape((n, 1))*X, mode='r')[0]
    H = np.dot(R.T, R)
    for i in range(k-1):
        H[i, i] += alpha
    return H

def compute_aloocv(X, y, C):
    alpha = 1.0 / C
    w = fit_logistic_regression(X, y, C)
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    n = X.shape[0]
    y = 2*y - 1
    u_vector = np.dot(X, w)
    p_vector = expit(u_vector*y)
    H = compute_hessian(p_vector, X, alpha)
    L = np.linalg.cholesky(H)
    T = scipy.linalg.solve_triangular(L, X.T, lower=True)
    h_vector = np.array([np.dot(ti, ti) for pi, ti in zip(p_vector, T.T)])
    loo_u_vector = u_vector - y * (1 - p_vector)*h_vector / (1 - p_vector*(1 - p_vector)*h_vector)
    loo_likelihoods = expit(y*loo_u_vector)
    return sum(np.log(loo_likelihoods))

class TestLogisticRegressionModel(unittest.TestCase):
    def test_we_optimize_the_aloocv(self):
        X, y = load_breast_cancer(return_X_y=True)
        X = StandardScaler().fit_transform(X)
        model = peak_engines.LogisticRegressionModel()
        model.fit(X, y)
        self.assertTrue(model.within_tolerance_)
        C_opt = model.C_[0]
        delta = 0.001
        f_low = compute_aloocv(X, y, C_opt-delta)
        f_opt = compute_aloocv(X, y, C_opt)
        f_high = compute_aloocv(X, y, C_opt+delta)
        self.assertLess(f_low, f_opt)
        self.assertGreater(f_opt, f_high)

if __name__ == "__main__":
    unittest.main()
