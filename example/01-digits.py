# Fit a multinomial logistic regression model for digit classification.
#
#   1. Preprocess the data set
#
#   2. Fit logistic regression model with hyperparameters set to minimize
#      the approximate leave-one-out cross-validation error.
#
#   3. Plot approximate leave-one-out cross-validation error across a 
#      a range of C values. (To verify it was minimized).
#
#   4. Analyze the approximate leave-one-out errors for each data point
#      and identify outliers.

####################################################################################################
# Part 1: Load and preprocess data set
####################################################################################################
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)
X = StandardScaler().fit_transform(X)

####################################################################################################
# Part 2: Fit logistic regression model with hyperparameters set so as to
#         optimize the performance on the approximate leave-one-out cross-validation
####################################################################################################
import bbai.glm

model = bbai.glm.LogisticRegression()
model.fit(X, y)

print("C_opt = ", model.C_)
print("ALO_opt = " , model.aloocv_)

# Prints:
#   C_opt = 1.0605577384508411
#   ALO_opt =  0.09934658483240914

####################################################################################################
# Part 3: Verify C optimizes approximate leave-one-out cross-validation
#         by plotting out ALO across a range of C values
####################################################################################################
import plotext as plt

def compute_alo(C):
    model_p = bbai.glm.LogisticRegression(C = C)
    model_p.fit(X, y)
    return model_p.aloocv_

Cs = np.arange(0.3, 3.0, 0.025)
alos = []
for C in Cs:
    alo = compute_alo(C)
    alos.append(alo)

plt.plotsize(80, 30)
plt.plot(Cs, alos)
plt.show()

#   Plots the approximate leave-one-out error (negative log-likelihood) across a range
#   of C values.
#
#      ┌─────────────────────────────────────────────────────────────────────────┐
# 0.111┤▌                                                                        │
#      │▌                                                                        │
#      │▌                                                                        │
#      │▐                                                                        │
# 0.109┤▐                                                                        │
#      │▐                                                                        │
#      │ ▌                                                                       │
#      │ ▚                                                                       │
# 0.107┤ ▐                                                                       │
#      │  ▌                                                                      │
#      │  ▌                                                                      │
#      │  ▐                                                                     ▄│
#      │  ▐                                                                  ▗▞▀ │
# 0.105┤   ▌                                                              ▗▄▀▘   │
#      │   ▝▖                                                           ▄▀▘      │
#      │    ▌                                                        ▄▞▀         │
#      │    ▚                                                     ▗▄▀            │
#      │    ▝▖                                                 ▄▞▀▘              │
# 0.103┤     ▚                                              ▗▞▀                  │
#      │      ▌                                          ▗▄▀▘                    │
#      │      ▝▖                                       ▄▀▘                       │
#      │       ▝▖                                  ▗▄▀▀                          │
# 0.101┤        ▚                               ▗▄▀▘                             │
#      │         ▀▖                          ▗▞▀▘                                │
#      │          ▝▄                     ▄▄▀▀▘                                   │
#      │            ▀▄               ▄▄▀▀                                        │
# 0.099┤              ▀▀▄▄▄▄▄▄▄▄▄▄▞▀▀                                            │
#      └┬─────────────────┬─────────────────┬─────────────────┬─────────────────┬┘
#      0.3               1.0               1.6               2.3               3.0

####################################################################################################
# Part 4: Display the approximate leave-one-out errors for each data point
####################################################################################################

alos = list(enumerate(model.aloocvs_))
alos = sorted(alos, key=lambda t: -t[1])
plt.clf()
plt.plotsize(80, 30)
plt.scatter([entry[1] for entry in alos])
plt.show()

#    Plots the approximate leave-one-out error (negative log-likelihood) for each data
#    point, sorted by largest error to smallest error.
# 
#    ┌───────────────────────────────────────────────────────────────────────────┐
# 5.4┤▌                                                                          │
#    │                                                                           │
#    │▖                                                                          │
#    │                                                                           │
# 4.5┤▌                                                                          │
#    │▖                                                                          │
#    │▘                                                                          │
#    │                                                                           │
# 3.6┤▚                                                                          │
#    │▗                                                                          │
#    │                                                                           │
#    │                                                                           │
#    │▝                                                                          │
# 2.7┤                                                                           │
#    │▝                                                                          │
#    │▝                                                                          │
#    │▝                                                                          │
# 1.8┤▐                                                                          │
#    │▐▖                                                                         │
#    │ ▌                                                                         │
#    │ ▙                                                                         │
#    │ ▐                                                                         │
# 0.9┤ ▝▖                                                                        │
#    │  ▜▖                                                                       │
#    │   ▙▖                                                                      │
#    │    ▜▄▖                                                                    │
# 0.0┤      ▀▀▜▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄│
#    └┬─────────────────┬──────────────────┬──────────────────┬─────────────────┬┘
#     0                449                898                1347             1796


for index, alo in alos[:10]:
    print(index, alo)

# Prints:
#
#   1264 5.367985195590403
#   792 5.196592418915968
#   1658 4.843286893193743
#   1729 4.5162436057106685
#   1553 4.400927550145405
#   77 4.2193647328582635
#   5 4.217654528037082
#   37 4.184274377533332
#   1727 4.12449347895783
#   1551 4.097293647439981
