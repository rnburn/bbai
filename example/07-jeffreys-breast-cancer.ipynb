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
s_map = np.sqrt(np.diag(model.laplace_covariance_matrix_))

# Step 4: Print out the weights along with their corresponding standard error.
max_len = max([len(s) for s in features])
fmt = '%-' + str(max_len) + 's'
for j in range(p+1):
    print(fmt % features[j], '%f (%f)' % (w_map[j], s_map[j]))

# Prints out
#
#  mean radius             18.844369 (24.768678)
#  mean texture            0.718237 (0.886588)
#  mean perimeter          -19.185223 (25.130313)
#  mean area               -0.157821 (9.747087)
#  mean smoothness         0.367833 (1.084369)
#  mean compactness        3.227443 (3.134865)
#  mean concavity          -0.022651 (4.891771)
#  mean concave points     -2.040344 (3.665972)
#  mean symmetry           -0.325849 (0.796396)
#  mean fractal dimension  -0.766704 (1.451205)
#  radius error            -10.264099 (3.531063)
#  texture error           0.806598 (0.983447)
#  perimeter error         5.748345 (3.739117)
#  area error              3.829019 (3.411303)
#  smoothness error        0.011620 (0.545934)
#  compactness error       -0.722398 (1.558400)
#  concavity error         0.377161 (1.517341)
#  concave points error    -1.083376 (1.338666)
#  symmetry error          -0.103406 (1.088318)
#  fractal dimension error 1.550249 (0.875781)
#  worst radius            -4.162614 (7.486983)
#  worst texture           -2.733197 (1.280611)
#  worst perimeter         -12.439716 (10.074531)
#  worst area              11.414854 (6.282013)
#  worst smoothness        -1.011890 (1.015591)
#  worst compactness       1.888830 (2.405582)
#  worst concavity         -1.708811 (2.196136)
#  worst concave points    0.375874 (2.718201)
#  worst symmetry          -0.268151 (0.999374)
#  worst fractal dimension -1.351309 (1.305911)
#  intercept               1.333285 (0.733033)
