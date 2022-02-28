import pytest
import numpy as np
from bbai.glm import LogisticRegressionMAP
np.random.seed(0)

###########################################
# Generate Dataset
###########################################
def generate_correlation_matrix(p, param):
    res = np.zeros(shape=(p, p))
    for s in range(p):
        for t in range(0, s+1):
            corr = param
            if s == t:
                corr = 1.0
            res[s, t] = corr
            res[t, s] = corr
    return res

def generate_design_matrix(n, K):
    mean = np.zeros(K.shape[0])
    return np.random.multivariate_normal(mean, K, size=n)

def generate_weights(p):
    return np.random.normal(size=p)

def generate_data_set(n, K):
    p = K.shape[0]
    X = generate_design_matrix(n, K)
    w = generate_weights(p)

    u = np.dot(X, w)

    p = 1 / (1 + np.exp(-u))

    y = []
    for i in range(n):
        y.append(np.random.binomial(1, p[i]))
    y = np.array(y)

    return X, y, w

###########################################
# Compute Jeffrey Prior
###########################################
def compute_a_matrix(X, u):
    p_vector = 1 / (1 + np.exp(u))
    return np.diag(p_vector * (1 - p_vector))

def compute_fisher_information_matrix(X, u):
    A = compute_a_matrix(X, u)
    return np.dot(X.T, np.dot(A, X))

def compute_log_prior(X, u):
    FIM = compute_fisher_information_matrix(X, u)
    return 0.5 * np.log(np.linalg.det(FIM))

###########################################
# Compute Posterior
###########################################
def compute_posterior(X, y, w):
    u = np.dot(X, w)
    log_prior = compute_log_prior(X, u)
    y = 2 * y - 1
    cost = np.sum(np.log(1 + np.exp(-y * u)))
    return cost - log_prior

###########################################
# Tests
###########################################
@pytest.fixture
def dataset1():
    n = 20
    p = 3
    K = generate_correlation_matrix(p, 0.5)
    X, y, _ = generate_data_set(n, K)
    return X, y

def test_map_optimization1(dataset1):
    X, y = dataset1
    model = LogisticRegressionMAP(fit_intercept=False)
    model.fit(X, y)
    assert model.intercept_[0] == 0
    w = model.coef_[0]
    f_opt = compute_posterior(X, y, w)
    delta = 1.0e-3
    for j in range(len(w)):
        for sign in [-1, 1]:
            w_p = np.array(w)
            w_p[j] += sign * delta
            f_delta = compute_posterior(X, y, w_p) - f_opt
            assert f_delta > 0.0

def test_map_optimization1_with_intercept(dataset1):
    X, y = dataset1
    n, p = X.shape
    model = LogisticRegressionMAP()
    model.fit(X, y)
    assert model.intercept_[0] != 0
    w = np.array(list(model.coef_[0]) + [model.intercept_[0]])
    X = np.hstack((X, np.ones((n, 1))))
    f_opt = compute_posterior(X, y, w)
    delta = 1.0e-3
    for j in range(len(w)):
        for sign in [-1, 1]:
            w_p = np.array(w)
            w_p[j] += sign * delta
            f_delta = compute_posterior(X, y, w_p) - f_opt
            assert f_delta > 0.0

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
