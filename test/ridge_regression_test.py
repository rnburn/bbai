import unittest
import numpy as np
from collections import defaultdict
import peak_engines

np.random.seed(1)

def make_random_problem(n, beta):
    k = len(beta)
    X = np.random.random_sample((n,k))
    for j in range(X.shape[1]):
        col = X[:,j]
        X[:,j] = col / np.std(col)
    y = np.dot(X, beta) + np.random.normal(scale=0.50, size=n)
    return X, y

def compute_loocv(X, y, alpha, fit_intercept=False):
    # Compute the leave-one-out cross-validation using a brute force approach
    if not hasattr(alpha, '__iter__'):
        alpha = [alpha] * X.shape[1]
    if fit_intercept:
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        alpha = np.array(list(alpha) + [0])
    Gamma = np.diag(alpha)
    result = 0
    for i in range(len(y)):
        x_i = X[i,:]
        y_i = y[i]
        X_mi = np.delete(X, i, 0)
        y_mi = np.delete(y, i)
        A_mi = np.dot(np.conj(X_mi.T), X_mi) + np.dot(Gamma.T, Gamma)
        b_hat_mi = np.dot(np.linalg.inv(A_mi), np.dot(np.conj(X_mi.T), y_mi))
        y_hat_i = np.dot(x_i, b_hat_mi)
        result += np.abs(y_i - y_hat_i)**2
    return result / len(y)

def compute_gcv(X, y, alpha, fit_intercept=False):
    # Compute the generalized cross-validation as a leave-one-out cross-validation on a rotated
    # dataset.
    #
    # This approach is valid when multiple regularizers are used and is equivalent to this more
    # familiar formula when only a single regularizer is used
    #  1/n*|y - X b_hat|^2/[1/n Tr(I - X A^{-1} X^T)]^2
    # where
    #  A = X^T X + alpha^2 I
    #
    # See 
    # Golub G., Heath M., and Wahba G., Generalized Cross-Validation as a Method for Choosing a Good Ridge Parameter (1979), TECHNOMETRICS, Vol 21, No 2
    n, k = X.shape
    if not hasattr(alpha, '__iter__'):
        alpha = [alpha] * k
    if fit_intercept:
        X = np.hstack((X, np.ones((n, 1))))
        alpha = np.array(list(alpha) + [0])
    U, S, Vt = np.linalg.svd(X)
    W = np.array([[np.exp(2j*np.pi*i*j/n) / np.sqrt(n) for j in range(n)] for i in range(n)])
    Q = np.dot(W, U.T)
    X_prime = np.dot(Q, X)
    y_prime = np.dot(Q, y)
    return compute_loocv(X_prime, y_prime, alpha, False)

def verify_optimum_1_impl(X, y, alpha, cv_computer):
    # Verify that a regularizer optimizes the LOOCV by stepping to neighboring points and checking
    # that we can't find better parameters
    delta_x = 1.0e-5
    def f(x):
        return cv_computer(X, y, x)
    f0 = f(alpha)
    for x in [alpha-delta_x, alpha+delta_x]:
        x = np.abs(x)
        delta_y = f(x) - f0
        relative_delta_y = delta_y / delta_x
        if relative_delta_y < 0 and np.abs(relative_delta_y) > 1.0e-3:
            print(delta_y)
            print(relative_delta_y)
            return False
    return True

def verify_optimum_1(score, X, y, alpha, fit_intercept=False):
    if score == 'loocv':
        return verify_optimum_1_impl(X, y, alpha, 
                    lambda X, y, alpha: compute_loocv(X, y, alpha, fit_intercept))
    return verify_optimum_1_impl(X, y, alpha,
                    lambda X, y, alpha: compute_gcv(X, y, alpha, fit_intercept))

def verify_optimum_p_impl(X, y, alpha, cv_computer):
    # Verify that a vector of regularizers optimizes the LOOCV by stepping to neighboring points and
    # checking that we can't find better parameters
    delta_x = 1.0e-5
    for i, alpha_i in enumerate(alpha):
        def f(x):
            alpha_copy = np.array(alpha)
            alpha_copy[i] = x
            return cv_computer(X, y, alpha_copy)
        f0 = f(alpha_i)
        for x in [alpha_i - delta_x, alpha_i + delta_x]:
            x = np.abs(x)
            delta_y = f(x) - f0
            relative_delta_y = delta_y / delta_x
            if relative_delta_y < 0 and np.abs(relative_delta_y) > 1.0e-2:
                print(delta_y)
                print(relative_delta_y)
                return False
    return True

def verify_optimum_p(score, X, y, alpha, fit_intercept=False):
    if score == 'loocv':
        return verify_optimum_p_impl(X, y, alpha, 
                    lambda X, y, alpha: compute_loocv(X, y, alpha, fit_intercept))
    return verify_optimum_p_impl(X, y, alpha, 
                    lambda X, y, alpha: compute_gcv(X, y, alpha, fit_intercept))

def verify_optimum_grouped(score, X, y, alpha, groupings, fit_intercept=False):
    cv_computer = compute_loocv
    if score == 'gcv':
        cv_computer = compute_gcv
    grouped_alpha = np.zeros(np.max(groupings)+1)
    group_map = defaultdict(list)
    for i, g in enumerate(groupings):
        group_map[g].append(i)
        grouped_alpha[g] = alpha[i]
    def cv_computer_prime(X, y, alpha):
        alpha_prime = np.zeros(X.shape[1])
        for group_index, indexes in group_map.items():
            for index in indexes:
                alpha_prime[index] = alpha[group_index]
        # print("alpha'=", alpha_prime)
        return cv_computer(X, y, alpha_prime, fit_intercept)
    return verify_optimum_p_impl(X, y, grouped_alpha, cv_computer_prime)


class TestRidgeRegressionModel(unittest.TestCase):
    def test_fit_random_problem(self):
        for score in ['loocv', 'gcv']:
            for fit_intercept in [False, True]:
                beta = [1.0, 2]
                num_data = 15
                X, y = make_random_problem(num_data, beta)
                model = peak_engines.RidgeRegressionModel(
                                fit_intercept=fit_intercept, normalize=False, score=score)
                model.fit(X, y)
                self.assertTrue(model.within_tolerance_)

                self.assertEqual(len(model.regularization_), len(beta))

                self.assertTrue(verify_optimum_1(score, X, y, model.regularization_[0], fit_intercept))

                y_pred = model.predict(X)
                self.assertEqual(len(y_pred), num_data)

    def test_fit_random_problem_multiple_regularizers(self):
        for score in ['loocv', 'gcv']:
            for fit_intercept in [False, True]:
                beta = [1.0, 2.]
                num_data = 15
                X, y = make_random_problem(num_data, beta)
                model = peak_engines.RidgeRegressionModel(
                                fit_intercept=fit_intercept, normalize=False, score=score, 
                                grouping_mode='none')
                model.fit(X, y)
                self.assertTrue(model.within_tolerance_)

                self.assertEqual(len(model.regularization_), len(beta))

                self.assertTrue(verify_optimum_p(score, X, y, model.regularization_, fit_intercept))

                y_pred = model.predict(X)
                self.assertEqual(len(y_pred), num_data)

    def test_fit_random_problem_grouped_regularizers(self):
        for score in ['loocv', 'gcv']:
            for fit_intercept in [False, True]:
                beta = [1.0, 0., 0.5]
                groupings = [0, 1, 0]
                num_data = 15
                X, y = make_random_problem(num_data, beta)
                model = peak_engines.RidgeRegressionModel(
                                fit_intercept=fit_intercept, normalize=False, score=score, 
                                grouper=lambda X, y: groupings)
                model.fit(X, y)
                self.assertTrue(model.within_tolerance_)

                self.assertEqual(len(model.regularization_), len(beta))

                self.assertTrue(verify_optimum_grouped(score, X, y, model.regularization_, groupings, fit_intercept))

                y_pred = model.predict(X)
                self.assertEqual(len(y_pred), num_data)

    def test_group_regressors(self):
        for fit_intercept in [False, True]:
            beta = [1.0, 0., 1.0]
            num_data = 20
            X, y = make_random_problem(num_data, beta)
            model = peak_engines.RidgeRegressionModel(
                            fit_intercept=fit_intercept, normalize=False, score='loocv', num_groups=2)
            model.fit(X, y)
            self.assertTrue(model.within_tolerance_)
            regularization = model.regularization_
            self.assertEqual(regularization[0], regularization[2])
            self.assertNotEqual(regularization[0], regularization[1])

    def test_group_regressors2(self):
        for fit_intercept in [False, True]:
            beta = [4.0, 0.0, 5.0, 0.1, 100.0]
            num_data = 100
            X, y = make_random_problem(num_data, beta)
            model = peak_engines.RidgeRegressionModel(
                            fit_intercept=fit_intercept, normalize=False, score='loocv', num_groups=3)
            model.fit(X, y)
            self.assertTrue(model.within_tolerance_)
            regularization = model.regularization_
            self.assertEqual(regularization[0], regularization[2])
            self.assertEqual(regularization[1], regularization[3])
            self.assertNotEqual(regularization[0], regularization[1])
            self.assertNotEqual(regularization[0], regularization[4])
            self.assertNotEqual(regularization[1], regularization[4])

    def test_invalid_grouper(self):
        beta = [4.0, 0.0, 5.0, 0.1, 100.0]
        num_data = 100
        X, y = make_random_problem(num_data, beta)
        model = peak_engines.RidgeRegressionModel(grouper=lambda X, y: None)
        self.assertRaises(Exception, model.fit, X, y)

        model = peak_engines.RidgeRegressionModel(grouper=lambda X, y: [1, 2, 3, 4, 5])
        self.assertRaises(Exception, model.fit, X, y)

        model = peak_engines.RidgeRegressionModel(grouper=lambda X, y: [0, 1, 2, 3, 4, 5])
        self.assertRaises(Exception, model.fit, X, y)

if __name__ == "__main__":
    unittest.main()
