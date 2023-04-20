# Use equations from
#
#    Ren, Sun, and He. Objective Bayesian analysis for a spatial model with nugget effects
#    https://www.sciencedirect.com/science/article/abs/pii/S037837581200081X
#
# to validate our Gaussian Process model. The focus here is on correctness and simplicity
# rather than efficiency and numerical stability so some of the equations are evaluated in a
# naive manner.
import numpy as np
from collections import namedtuple
import scipy.linalg
from scipy.stats import invgamma

class RbfCovarianceFunction:
    def compute_covariance_matrix(self, location_matrix, theta):
        n, d = location_matrix.shape
        res = np.zeros((n, n))
        for s in range(n):
            for t in range(0, s+1):
                l1 = location_matrix[s, :]
                l2 = location_matrix[t, :]
                d2 = np.linalg.norm(l1 - l2)**2
                k = np.exp(-0.5 * d2 / theta**2)
                res[s, t] = k
                res[t, s] = k
        return res

    def compute_covariance_derivative_matrix(self, location_matrix, theta):
        n, d = location_matrix.shape
        res = np.zeros((n, n))
        for s in range(n):
            for t in range(0, s+1):
                l1 = location_matrix[s, :]
                l2 = location_matrix[t, :]
                d2 = np.linalg.norm(l1 - l2)**2
                dk = d2 * np.exp(-0.5*d2 / theta**2) / theta**3
                res[s, t] = dk
                res[t, s] = dk
        return res

Workspace = namedtuple(
        'Workspace',
        [
            'G',
            'L',
            'R',
            'Rg',
        ],
)

# Equation (2) from reference
def compute_g_matrix(location_matrix, cvf, theta, eta):
    G = cvf.compute_covariance_matrix(location_matrix, theta)
    n = G.shape[0]
    for i in range(n):
        G[i, i] += eta
    return G

# Equation (8) from reference
def compute_rg_matrix(X, G):
    _, p = X.shape

    L = np.linalg.cholesky(G)
    G_inv, _ = scipy.linalg.lapack.dpotri(L, lower=1)
    G_inv = G_inv + np.triu(G_inv.T, k=1)

    if p == 0:
        return G_inv, L, np.zeros((0, 0))

    U = scipy.linalg.solve_triangular(L, X, lower=True)

    _, R = scipy.linalg.qr(U)
    R = R[:p, :p]

    U = scipy.linalg.solve_triangular(L, U, lower=True, trans='T')
    U = scipy.linalg.solve_triangular(R, U.T, lower=False, trans='T')


    res = G_inv - np.dot(U.T, U)

    return res, L, R

# Equation (21) from reference
def compute_sigma_matrix(Rg, dK, p):
    n = dK.shape[0]

    res = np.zeros((3, 3))

    t1_matrix = np.dot(Rg, dK)
    t1_trace = np.sum(np.diag(t1_matrix))

    t2_matrix = np.dot(Rg, t1_matrix)
    t2_trace = np.sum(np.diag(t2_matrix))

    t3_matrix = np.dot(Rg, Rg)
    t3_trace = np.sum(np.diag(t3_matrix))

    # 0, 0
    res[0, 0] = np.trace(np.dot(t1_matrix, t1_matrix))

    # 0, 1
    res[0, 1] = t2_trace
    res[1, 0] = res[0, 1]

    # 0, 2
    res[0, 2] = t1_trace
    res[2, 0] = res[0, 2]

    # 1, 1
    res[1, 1] = t3_trace

    # 1, 2
    res[1, 2] = np.sum(np.diag(Rg))
    res[2, 1] = res[1, 2]

    # 2, 2
    res[2, 2] = n - p

    return res

def compute_workspace(X, location_matrix, cvf, theta, eta):
    G = compute_g_matrix(location_matrix, cvf, theta, eta)
    Rg, L, R = compute_rg_matrix(X, G)
    return Workspace(
            G = G,
            L = L,
            R = R,
            Rg = Rg,
    )

# Equation (23) from reference
def compute_prior(ws, location_matrix, cvfn, theta):
    p = ws.R.shape[0]
    dK = cvfn.compute_covariance_derivative_matrix(location_matrix, theta)
    sigma_matrix = compute_sigma_matrix(ws.Rg, dK, p)
    _, logdet = np.linalg.slogdet(sigma_matrix)
    return 0.5 * logdet


# Equation (20) from reference
def compute_marginal_likelihood(ws, y):
    n = len(y)
    p = ws.R.shape[0]
    term1 = -np.sum(np.log(np.diag(ws.L)))
    term2 = -np.sum(np.log(np.abs(np.diag(ws.R))))

    S2 = np.dot(y, np.dot(ws.Rg, y))
    term3 = - (n - p) * np.log(S2) / 2.0

    return term1 + term2 + term3

def compute_posterior(X, location_matrix, y, cvfn, theta, eta):
    ws = compute_workspace(X, location_matrix, cvfn, theta, eta)

    term1 = compute_marginal_likelihood(ws, y)
    term2 = compute_prior(ws, location_matrix, cvfn, theta)

    return term1 + term2

def compute_prediction_mean(
        X_train, location_matrix_train, y_train, X_test, location_matrix_test,
        cvfn, theta, eta):
    num_train = len(y_train)
    X = np.vstack((X_train, X_test))
    location_matrix = np.vstack((location_matrix_train, location_matrix_test))
    G = compute_g_matrix(location_matrix, cvfn, theta, eta)
    
    R, _, _ = compute_rg_matrix(X, G)
    R12 = R[:num_train, num_train:]
    R22 = R[num_train:, num_train:]

    return -np.dot(np.linalg.inv(R22), np.dot(R12.T, y_train))

def compute_unormalized_pdf(
        X_train, location_matrix_train, y_train, X_test, location_matrix_test,
        cvfn, theta, eta):
    num_train = len(y_train)
    num_regressors = X_train.shape[1]
    X = np.vstack((X_train, X_test))
    location_matrix = np.vstack((location_matrix_train, location_matrix_test))
    G = compute_g_matrix(location_matrix, cvfn, theta, eta)
    
    R, _, _ = compute_rg_matrix(X, G)
    R11 = R[:num_train, :num_train]
    R12 = R[:num_train, num_train:]
    R22 = R[num_train:, num_train:]

    y_mean = -np.dot(np.linalg.inv(R22), np.dot(R12.T, y_train))

    b = np.dot(y_train, np.dot(R11, y_train)) - np.dot(y_mean, np.dot(R22, y_mean))

    v = (num_train - X_train.shape[1] + len(y_mean)) / 2.0
    def f(y):
        t = y - y_mean
        u = np.dot(t, np.dot(R22, t))
        return (u + b)**(-v)
    return f

def compute_sigma2_dist(X, location_matrix, y, cvfn, theta, eta):
    ws = compute_workspace(X, location_matrix, cvfn, theta, eta)
    S2 = np.dot(y, np.dot(ws.Rg, y))
    a = (len(location_matrix) - X.shape[1]) / 2.0
    scale = S2 / 2
    return invgamma(a, scale=scale)

