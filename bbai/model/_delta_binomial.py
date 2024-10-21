from bbai._computation._bridge import mdldb

class DeltaBinomialModel:
    """Model the difference of the probabilities of two binomial distributions.

    The delta of binomial distribution probabilities was studied by Laplace in [1, p. 59], where
    Laplace sought to determine whether London has a higher boys-to-girls birth rate than Paris.

    The model provides a posterior distribution for the difference of two binomial distribution
    probabilities. It can be fit with three different priors:

       * uniform: the original uniform prior used by Laplace
       * Jeffreys: Jeffreys rule prior
       * Reference (recommended): the reference, as determined by Proposition 0.2 of [2].

    References:
      [1]: Laplace, P. (1778). Mémoire sur les probabilités. Translated by Richard J. Pulskamp.
      [2]: Berger, J., J. Bernardo, and D. Sun (2022). Objective bayesian inference and its 
           relationship to frequentism.
    
    Parameters
    ----------
    prior : default='reference'
        The prior to use. Possible values are 'uniform', 'jeffreys', and 'reference'.

    Example
    --------
    >>> from bbai.model import DeltaBinomialModel
    >>> import numpy as np
    >>> 
    >>> a1, b1, a2, b2 = 5, 3, 2, 7
    >>> model = DeltaBinomialModel(prior='reference')
    >>>
    >>> # Fit a posterior distribution with likelihood function
    >>> #     L(theta, x) = (theta + x)^a1 * (1 - theta - x)^b1 * x^a2 (1-x)^b2 
    >>> # where theta represents the difference of the two binomial distribution probabilities
    >>> model.fit(a1, b1, a2, b2)
    >>>
    >>> # Print the probability that theta < 0.123
    >>> print(model.cdf(0.123))
    >>>     # Prints 0.10907436812863071
    """
    UNIFORM = 0
    JEFFREYS = 1
    REFERENCE = 2

    def __init__(self, prior='reference'):
        if prior == 'uniform':
            self.prior_ = DeltaBinomialModel.UNIFORM
        elif prior == 'jeffreys':
            self.prior_ = DeltaBinomialModel.JEFFREYS
        elif prior == 'reference':
            self.prior_ = DeltaBinomialModel.REFERENCE
        else:
            raise RuntimeError("unknown prior {}".format(prior))
        self.model_ = None

    def fit(self, a1, b1, a2, b2):
        """Fit a posterior distribution for the parameter p1 - p2 where p1 and p2
        represent probabilities for binomial distributions with likelihood functions
            L1(p1) = p1^a1 * (1 - p1)^b1
            L2(p2) = p2^a2 * (1 - p2)^b2
        """
        self.model_ = mdldb.delta_binomial_fit(self.prior_, a1, b1, a2, b2)

    def cdf(self, t):
        """Cumulative density function for the posterior in p1-p2"""
        assert -1 <= t and t <= 1
        assert self.model_ is not None
        return mdldb.delta_binomial_cdf(self.model_, t)

    def pdf(self, t):
        """Probability density function for the posterior in p1-p2"""
        assert -1 <= t and t <= 1
        assert self.model_ is not None
        return mdldb.delta_binomial_pdf(self.model_, t)

    def prior_pdf(self, *args):
        """Probability density function for the prior in p1-p2
            model.prior_pdf(theta)
                 # returns the marginal prior PDF 
            model.prior_pdf(theta, x)
                 # returns the prior PDF 
        """
        assert self.model_ is not None
        if len(args) == 1:
            return mdldb.delta_binomial_prior_marginal_pdf(self.model_, args[0])
        elif len(args) == 2:
            return mdldb.delta_binomial_prior_pdf(self.model_, args[0], args[1])
