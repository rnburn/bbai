# Fit a binomial logistic regression model for on the breast cancer data set with
# weights select so as to maximize the posterior distribution with Jeffreys prior.

from sklearn.preprocessing import StandardScaler
import sklearn.linear_model
from sklearn.datasets import load_breast_cancer
import bbai.glm
import numpy as np

# Step 1: Load data and preprocess
data = load_breast_cancer()

X = data['data']
y = data['target']
features = list(data['feature_names'])
features.append('intercept')
n, p = X.shape
X = StandardScaler().fit_transform(X)

# Step 2: Fit Model
model = bbai.glm.LogisticRegressionMAP()
model.fit(X, y)
w_map = list(model.coef_[0])
w_map.append(model.intercept_[0])

# Step 3: Compute standard error measurements based off of the Fisher information matrix
stderr = 1.0 / np.sqrt(np.diag(model.hessian_))

# Step 4: Print out the weights along with their corresponding standard error.
max_len = max([len(s) for s in features])
fmt = '%-' + str(max_len) + 's'
for j in range(p+1):
    print(fmt % features[j], '%f (%f)' % (w_map[j], stderr[j]))

# Prints out
#
# mean radius             18.844369 (0.246536)
# mean texture            0.718237 (0.264921)
# mean perimeter          -19.185223 (0.247434)
# mean area               -0.157821 (0.206162)
# mean smoothness         0.367833 (0.319041)
# mean compactness        3.227443 (0.262862)
# mean concavity          -0.022651 (0.226283)
# mean concave points     -2.040344 (0.308589)
# mean symmetry           -0.325849 (0.245192)
# mean fractal dimension  -0.766704 (0.210683)
# radius error            -10.264099 (0.134592)
# texture error           0.806598 (0.217691)
# perimeter error         5.748345 (0.127539)
# area error              3.829019 (0.111033)
# smoothness error        0.011620 (0.178030)
# compactness error       -0.722398 (0.144933)
# concavity error         0.377161 (0.105596)
# concave points error    -1.083376 (0.158857)
# symmetry error          -0.103406 (0.191188)
# fractal dimension error 1.550249 (0.127858)
# worst radius            -4.162614 (0.292164)
# worst texture           -2.733197 (0.252259)
# worst perimeter         -12.439716 (0.290176)
# worst area              11.414854 (0.237330)
# worst smoothness        -1.011890 (0.274542)
# worst compactness       1.888830 (0.218036)
# worst concavity         -1.708811 (0.213529)
# worst concave points    0.375874 (0.370696)
# worst symmetry          -0.268151 (0.203808)
# worst fractal dimension -1.351309 (0.216937)
# intercept               1.333285 (0.246327)
