# Fit a regularized Bayesian Linear Regression model to a data set and compare the
# model to Ordinary Least Squares (OLS) and a ridge regression model fit so as to
# minimize Leave-one-out Cross-validation.
#
#   1. Generate the data set.
#
#   2. Fit OLS, Bayesian ridge regression, and ridge regression models.
#
#   3. Compare the prediction errors (measured in error variance) of the different models.
#
#   4. Compare the amount of shrinkage of Bayesian ridge regression and LOOCV ridge regression.
#
#   5. Compare the predicted noise variance and weight variance of OLS and Bayesian ridge 
#      regression.

####################################################################################################
# Part 1: Generate data set
####################################################################################################
import numpy as np
np.random.seed(0)

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

def generate_data_set(n, K, signal_noise_ratio):
    p = K.shape[0]
    X = generate_design_matrix(n, K)
    w = generate_weights(p)
    signal_var = np.dot(w, np.dot(K, w))
    w *= np.sqrt(signal_noise_ratio / signal_var)
    y = np.dot(X, w) + np.random.normal(size=n)
    return X, y, w

p = 10
n = 20
signal_noise_ratio = 1.0
K = generate_correlation_matrix(p, 0.5)
X, y, w_true = generate_data_set(n, K, signal_noise_ratio)


####################################################################################################
# Part 2: Fit models
####################################################################################################
# OLS
from sklearn.linear_model import LinearRegression
model_ols = LinearRegression(fit_intercept=False)
model_ols.fit(X, y)

# Bayesian Linear Regression
from bbai.glm import BayesianRidgeRegression
model_bay = BayesianRidgeRegression(fit_intercept=False)
model_bay.fit(X, y)

# Ridge Regression (fit to optimize LOOCV error)
from bbai.glm import RidgeRegression
model_rr = RidgeRegression(fit_intercept=False)
model_rr.fit(X, y)


####################################################################################################
# Part 3: Measure and compare the prediction errors for each model
####################################################################################################
def compute_prediction_error_variance(K, w_true, w):
    delta = w - w_true
    noise_variance = 1.0
    return noise_variance + np.dot(delta, np.dot(K, delta))

err_variance_true = compute_prediction_error_variance(K, w_true, w_true)
err_variance_ols = compute_prediction_error_variance(K, w_true, model_ols.coef_)
err_variance_bay = compute_prediction_error_variance(K, w_true, model_bay.weight_mean_vector_)
err_variance_rr = compute_prediction_error_variance(K, w_true, model_rr.coef_)

print("===== prediction error variance")
print("err_variance_true =", err_variance_true)
print("err_variance_ols =", err_variance_ols)
print("err_variance_bay =", err_variance_bay)
print("err_variance_rr =", err_variance_rr)

# Prints:
#   err_variance_true = 1.0
#   err_variance_ols = 2.2741141849936
#   err_variance_bay = 1.2253596030087812
#   err_variance_rr = 1.3346410286197081

####################################################################################################
# Part 4: Compare the amount of regularization, measured by shrinkage, between ridge regression
#         and the expected weight variance of Bayesian ridge regression
####################################################################################################
def print_shrinkage_comparison_table(w_ols, w_bay, w_rr):
    p = len(w_ols)
    print("coef\tbay\t\t\trr")
    for j in range(p):
        shrink_bay = np.abs(w_bay[j] / w_ols[j])
        shrink_rr = np.abs(w_rr[j] / w_ols[j])
        print(j, "\t", shrink_bay, "\t", shrink_rr)
    w_ols_norm = np.linalg.norm(w_ols)
    w_bay_norm = np.linalg.norm(w_bay)
    w_rr_norm = np.linalg.norm(w_rr)
    shrink_bay = w_bay_norm / w_ols_norm 
    shrink_rr = w_rr_norm / w_ols_norm
    print("total\t", shrink_bay, "\t", shrink_rr)

####################################################################################################
# Part 5: Compare OLS expected noise and weight variances to the expected noise and weight
#         variances from Bayesian ridge regression
####################################################################################################
err_ols = y - np.dot(X, model_ols.coef_)
s_ols = np.dot(err_ols, err_ols) / (n - p)
print("===== noise variance comparison")
print("noise_variance_ols =", s_ols)
print("noise_variance_bay =", model_bay.noise_variance_mean_)

# Prints
#    noise_variance_ols = 0.4113088518293715
#    noise_variance_bay = 0.646895178795739

print("===== weight variance comparison")
w_covariance_ols = s_ols * np.linalg.inv(np.dot(X.T, X))
print("coef\tOLS\t\t\tbay")
for j in range(p):
    print(j, "\t", w_covariance_ols[j, j], "\t", model_bay.weight_covariance_matrix_[j, j])

# Prints
#   coef	OLS	                 bay
#   0 	 0.08077309122198699 	 0.07028402634979392
#   1 	 0.11934260399465739 	 0.10372106176189841
#   2 	 0.07565674651068977 	 0.09961127194993058
#   3 	 0.05984623288735467 	 0.07126271235803656
#   4 	 0.06790137334901475 	 0.07153243474011801
#   5 	 0.04090207206067381 	 0.06168627188610626
#   6 	 0.06252983263310996 	 0.06966266955483112
#   7 	 0.2007543244631264 	 0.15580956937954868
#   8 	 0.06512850980887401 	 0.06067162578733245
#   9 	 0.03501167166038174 	 0.03794523021332901
