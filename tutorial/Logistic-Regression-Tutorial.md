In this tutorial, we'll show how to build a logistic regression model on the breast cancer dataset
packaged with skearn.

## Prepare the dataset
```python
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
X, y = load_boston(return_X_y=True)
X = StandardScaler().fit_transform(X)
```

### Fit a model
The package **peak-engines** fits logistic regression models with hyperparameters  
chosen so as to optimize the approximate leave-one-out cross-validation.
```python
model = peak_engines.LogisticRegressionModel()
model.fit(X, y)
print('C =', model.C_[0])
```
prints
```
C = 0.1355386645379227
```

### Other regularizers
By default, **peak-engines** uses l2 regularization. But because it finds hyperparameters using
an efficient second-order optimizer, it can handle regularization functions with multiple
hyperparameters. For example, the [bridge penalty](https://amstat.tandfonline.com/doi/abs/10.1080/10618600.1998.10474784) regularizes using a function of the form
```
alpha x Sum_i |w_i|^beta
```
where `beta >= 1` and `alpha` are both hyperparameters. To fit logistic regression for the bridge
penalty, run
```python
model = peak_engines.LogisticRegressionModel(penalty='bridge')
model.fit(X, y)
print(model.hyperparameters_)
```
prints
```
[1.76421091 0.91834575]
```
This corresponds to the regularization function
```
1.76421091^2 Sum_i |w_i|^(1 + 0.91834575^2)
```
