from .._computation._bridge import glmlg1

import numpy as np

class _Prediction:
    def __init__(self, model, xp): 
        self.model_ = model
        self.xp_ = xp

    def pdf(self, p):
        """Probability density function for the prediction posterior in p"""
        return glmlg1.prediction_pdf(self.model_, self.xp_, p)

    def cdf(self, p):
        """Cumulative density function for the prediction posterior in p"""
        return glmlg1.prediction_cdf(self.model_, self.xp_, p)

    def ppf(self, t):
        """Percentage point function for the prediction posterior in p"""
        return glmlg1.prediction_ppf(self.model_, self.xp_, t)

class BayesianLogisticRegression1:
    """Model logistic regression with a single unknown weight variable and the reference prior.

    Note: Here the reference prior works out to be the same as Jeffreys prior.

    Additionally, using the parameters w_min and w_max the prior can be restricted to an
    interval.
    
    Parameters
    ----------
    w_min : default='-∞'
        A minimum weight restriction for the prior.

    w_max : default='∞'
        A maximum weight restriction for the prior.

    Example
    --------
    >>> from bbai.glm import BayesianLogisticRegression1
    >>> 
    >>> x = [-5, 2, 8, 1]
    >>> y = [0, 1, 0, 1]
    >>> model = BayesianLogisticRegression1()
    >>>
    >>> # Fit a posterior distribution for w with the logistic
    >>> # regression reference prior
    >>> model.fit(x, y)
    >>>
    >>> # Print the probability that w < 0.123
    >>> print(model.cdf(0.123))
    """
    def __init__(self, w_min=-np.inf, w_max=np.inf):
        self.w_min_ = w_min
        self.w_max_ = w_max

    def fit(self, x, y):
        yp = np.zeros(len(y))
        for i, yi in enumerate(y):
            assert yi == 0 or yi == 1
            yp[i] = -1.0 if yi == 0 else 1.0
        self.model_ = glmlg1.logistic1_fit(x, yp, self.w_min_, self.w_max_)

    def prior_pdf(self, w):
        """Probability density function for the prior in w"""
        return glmlg1.prior_pdf(self.model_, w)

    def pdf(self, w):
        """Probability density function for the posterior in w"""
        return glmlg1.pdf(self.model_, w)

    def cdf(self, w):
        """Cumulative density function for the posterior in w"""
        return glmlg1.cdf(self.model_, w)

    def ppf(self, t):
        """Percentage point function for the posterior in w"""
        return glmlg1.ppf(self.model_, t)

    def predict(self, xp):
        """Return the posterior distribution for p where p denotes the probability 1 / (1 + exp(-xp * w))"""
        assert xp != 0.0
        return _Prediction(self.model_, xp)
