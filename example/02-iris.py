# Fit a multinomial logistic regression model for on the Iris data set
#
#   1. Preprocess the data set
#
#   3. Plot approximate leave-one-out cross-validation error across a 
#      a range of C values and compare it to leave-one-out cross-validation
#
#   2. Fit logistic regression model with hyperparameters set to minimize
#      the approximate leave-one-out cross-validation error.
#
#   4. Analyze the approximate leave-one-out errors for each data point
#      and identify outliers.

####################################################################################################
# Part 1: Load and preprocess data set
####################################################################################################
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

X, y = load_iris(return_X_y=True)
X = StandardScaler().fit_transform(X)
X = np.hstack((X, np.ones((X.shape[0], 1))))

####################################################################################################
# Part 2: Plot approximate leave-one-out cross-validation error and
#         leave-one-out cross-validation error
####################################################################################################
import bbai.glm
from sklearn.model_selection import LeaveOneOut
import matplotlib.pyplot as plt

def compute_aloocv(C):
    model = bbai.glm.LogisticRegression(C=C, fit_intercept=False)
    model.fit(X, y)
    return model.aloocv_

def compute_loocv(C):
    ll_sum = 0
    for train_indexes, test_indexes in LeaveOneOut().split(X):
        X_train = X[train_indexes]
        y_train = y[train_indexes]
        X_test = X[test_indexes]
        y_test = y[test_indexes]
        model = bbai.glm.LogisticRegression(C=C, fit_intercept=False)
        model.fit(X_train, y_train)
        pred = model.predict_log_proba(X_test)
        ll_sum += pred[0][y_test[0]]
    return -ll_sum / len(y)

Cs = np.logspace(0.5, 2.5, num=100)
aloocvs = [compute_aloocv(C) for C in Cs]
loocvs = [compute_loocv(C) for C in Cs]

plt.plot(Cs, aloocvs, label='ALOOCV')
plt.plot(Cs, loocvs, label='LOOCV')
plt.xlabel('C')
plt.xscale('log')
plt.ylabel('Cross-Validation Error')
plt.legend()
plt.savefig('iris_cv.svg')

####################################################################################################
# Part 2: Find the value of C that optimizes ALOOCV
####################################################################################################
model = bbai.glm.LogisticRegression(fit_intercept=False)
model.fit(X, y)
print("C_opt = ", model.C_)

####################################################################################################
# Part 4: Display the leave-one-out errors for each data point
####################################################################################################
def compute_loocvs(C):
    cvs = []
    for train_indexes, test_indexes in LeaveOneOut().split(X):
        X_train = X[train_indexes]
        y_train = y[train_indexes]
        X_test = X[test_indexes]
        y_test = y[test_indexes]
        model = bbai.glm.LogisticRegression(C=C, fit_intercept=False)
        model.fit(X_train, y_train)
        pred = model.predict_log_proba(X_test)
        cvs.append(-pred[0][y_test[0]])
    return cvs

model = bbai.glm.LogisticRegression(fit_intercept=False)
model.fit(X, y)

n = len(y)
aloocvs = model.aloocvs_
loocvs = compute_loocvs(model.C_)
indexes = list(range(n))
indexes = sorted(indexes, key=lambda i: -aloocvs[i])
aloocvs = [aloocvs[i] for i in indexes]
loocvs = [loocvs[i] for i in indexes]
ix = list(range(n))
plt.plot(ix, aloocvs, marker='x', label='ALOO', linestyle='None')
plt.plot(ix, loocvs, marker='+', label='LOO', linestyle='None')
plt.ylabel('Leave-one-out Error')
plt.xlabel('Data Point')
plt.legend()
plt.savefig('iris_loo.svg')
