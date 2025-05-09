import numpy as np

def evaluate_lo_errors_slow(X, y, m):
    """Given a model, evaluate leave-one-out cross validation using a brute force approach.

    Intended for testing."""
    n = len(y)
    res = np.zeros(n)
    for i in range(n):
        ix = [ip for ip in range(n) if ip != i]
        Xm = X[ix, :]
        ym = y[ix]
        m.fit(Xm, ym)
        pred = m.predict(X[i])
        res[i] += y[i] - pred
    return res

def evaluate_lo_cost_slow(X, y, m):
    """Given a model, evaluate leave-one-out cross validation using a brute force approach.

    Intended for testing."""
    errs = evaluate_lo_errors_slow(X, y, m)
    return np.sum(errs**2)

class LassoGridCv:
    """Construct a grid of leave-one-out cross validation values. Used for testing."""
    def __init__(self, f, X, y, lambda_max, n=10):
        cvs = []
        self.lambdas = np.linspace(0, lambda_max, n)
        for lda in self.lambdas:
            cv = f(X, y, lda)
            cvs.append(cv)
        self.cvs = cvs
        self.cv_min = np.min(cvs)

class LassoKKT:
    """Verify Karus-Kuhn-Tucker conditions for a Lasso solution."""
    def __init__(self, X, y, lda, beta, with_intercept=False):
        if with_intercept:
            X = np.hstack((np.ones((len(y), 1)), X))
        y_tilde = y - np.dot(X, beta)
        self.with_intercept_ = with_intercept
        self.kkt_ = np.dot(X.T, y_tilde)
        self.beta_ = beta
        self.lambda_ = lda

    def __str__(self):
        return str(self.kkt_)

    def within(self, tol1, tol2=0.0):
        beta = self.beta_
        kkt = self.kkt_

        # intercept
        if self.with_intercept_:
            b0 = beta[0]
            cond = kkt[0]
            if np.abs(b0) != 0.0:
                cond /= b0
            if np.abs(cond) > tol1:
                return False
            beta = beta[1:]
            kkt = kkt[1:]
            
        # regressors
        for j, bj in enumerate(beta):
            cond = kkt[j]
            if bj == 0.0:
                if np.abs(cond) > self.lambda_ + tol2:
                    return False
                continue
            cond *= np.sign(bj)
            if self.lambda_ == 0.0:
                if np.abs(cond) > np.abs(bj) * tol1:
                    return False
            else:
                if np.abs(cond - self.lambda_) > tol1 * self.lambda_:
                    return False
        return True
