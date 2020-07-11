import unittest
import math
import scipy.integrate
from sklearn.linear_model import LinearRegression
import numpy as np
import peak_engines
np.random.seed(0)

def vectorizable(f):
    def f_(x):
        if hasattr(x, '__iter__'):
            return np.array([f(xi) for xi in x])
        return f(x)
    return f_

def make_warper(parameters, y):
    y_mean = np.mean(y)
    y_std = np.std(y)
    def f(t):
        t = (t - y_mean) / (y_std * np.sqrt(len(y)))
        result = t
        for i in range(0, len(parameters), 3):
            a, b, c = parameters[i:(i+3)]
            result += a**2*math.tanh(b**2*(t + c))
        return result
    mean = np.mean([f(yi) for yi in y])
    @vectorizable
    def f_(t):
        return f(t) - mean
    @vectorizable
    def fp(t):
        t = (t - y_mean) / (y_std * np.sqrt(len(y)))
        result = 1.0
        for i in range(0, len(parameters), 3):
            a, b, c = parameters[i:(i+3)]
            u = math.tanh(b**2*(t + c))
            result += a**2*b**2*(1 - u**2)
        result /= y_std * np.sqrt(len(y))
        return result
    return f_, fp

def compute_log_likelihood_proxy(X, y, phi):
    f, fp = make_warper(phi, y)
    z = f(y)
    model = LinearRegression(fit_intercept=False)
    model.fit(X, z)
    z_pred = model.predict(X)
    rss = sum((z-z_pred)**2)
    return -len(y)/2.0*np.log(rss) + sum(np.log(fp(y)))


def verify_log_likelihood_opt(X, y, phi):
    delta_x = 1.0e-3
    for i, phi_i in enumerate(phi):
        def f(x):
            phi_copy = np.array(phi)
            phi_copy[i] = x
            return compute_log_likelihood_proxy(X, y, phi_copy)
        f0 = f(phi_i)
        for x in [phi_i - delta_x, phi_i + delta_x]:
            delta_f = f(x) - f0
            relative_delta_f = delta_f / delta_x
            if relative_delta_f > 0 and np.abs(relative_delta_f) > 1.0e-3:
                return False
    return True

def make_random_problem(n, k):
    X = np.random.random_sample((n,k))
    for j in range(X.shape[1]):
        col = X[:, j]
        X[:, j] = (col - np.mean(col)) / np.std(col)
    beta = np.array([1.0, 0.0])
    y = np.dot(X, beta) + np.random.normal(size=n)
    return X, y

class TestWarpedLinearRegressionModel(unittest.TestCase):
    def test_fit_warped_linear_regression_model(self):
        num_features = 2
        num_data = 15
        X, y = make_random_problem(num_data, num_features)
        model = peak_engines.WarpedLinearRegressionModel()
        model.fit(X, y)
        self.assertTrue(model.within_tolerance_)
        self.assertTrue(verify_log_likelihood_opt(X, y, model.warper_.parameters_))

        self.assertEqual(len(model.regressors_), num_features)

        y_pred = model.predict(X)
        self.assertEqual(len(y_pred), num_data)

    def test_warper(self):
        num_features = 2
        num_data = 15
        X, y = make_random_problem(num_data, num_features)
        model = peak_engines.WarpedLinearRegressionModel()
        model.fit(X, y)
        self.assertTrue(model.within_tolerance_)

        warper = model.warper_
        z = warper.compute_latent(y)
        y_prime = warper.invert(z)
        self.assertTrue(np.allclose(y_prime, y))

    def test_prediction_pdf(self):
        num_features = 2
        num_data = 15
        X, y = make_random_problem(num_data, num_features)
        model = peak_engines.WarpedLinearRegressionModel()
        model.fit(X, y)

        X_test = np.random.random_sample((1,num_features))
        logpdf = model.predict_logpdf(X_test)

        def f(y):
            return np.exp(logpdf([y]))
        quad_result = scipy.integrate.quad(f, -100, 100)
        self.assertAlmostEqual(quad_result[0], 1.0)

    def test_invalid_params(self):
        self.assertRaises(Exception, peak_engines.WarpedLinearRegressionModel, num_steps=0)
        self.assertRaises(Exception, peak_engines.WarpedLinearRegressionModel, num_steps=-1)
        self.assertRaises(Exception, peak_engines.WarpedLinearRegressionModel, tolerance=-1)
        self.assertRaises(Exception, peak_engines.WarpedLinearRegressionModel, tolerance=0)

    def test_invalid_fit(self):
        num_features = 2
        num_data = 15
        X, y = make_random_problem(num_data, num_features)
        model = peak_engines.WarpedLinearRegressionModel()
        self.assertRaises(Exception, model.fit, None, y)
        self.assertRaises(Exception, model.fit, X, None)
        self.assertRaises(Exception, model.fit, 3.0, 5.0)
        self.assertRaises(Exception, model.fit, [3.0], [5.0])
        self.assertRaises(Exception, model.fit, [[3.0]], [5.0])
        self.assertRaises(Exception, model.fit, [[3.0], [7.0]], [5.0, None])

    def test_optimization_fail(self):
        num_features = 2
        num_data = 15
        X, y = make_random_problem(num_data, num_features)
        y[0] = np.nan
        model = peak_engines.WarpedLinearRegressionModel()
        self.assertRaises(Exception, model.fit, X, y)


if __name__ == "__main__":
    unittest.main()
