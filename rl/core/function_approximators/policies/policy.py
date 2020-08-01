# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from abc import abstractmethod
import numpy as np
from rl.core.function_approximators.function_approximator import FunctionApproximator, online_compatible


class Policy(FunctionApproximator):
    """ An abstract interface that represents conditional distribution \pi(y|x).

        A policy is namely a stochastic FunctionApproximator.
    """

    def __init__(self, x_shape, y_shape, name='policy', **kwargs):
        super().__init__(x_shape, y_shape, name=name, **kwargs)

    def predict(self, xs, stochastic=True, **kwargs):
        """ Predict the values on batches of xs. """
        return super().predict(ts_xs, stochastic=stochastic, **kwargs)

    # New methods of Policy
    @online_compatible
    def predict_w_noise(self, xs, stochastic=True, **kwargs):
        """ Return both the prediction and the associated noise as a tuple, for
        future derandomization purposes. """
        raise NotImplementedError

    @online_compatible
    def noise(self, xs, ys):
        """ Return the noises used to generate ys given xs, so that the same ys
        can be computed when callling the `derandomize` method. """
        raise NotImplementedError

    @online_compatible
    def derandomize(self, xs, noises):
        """ The inverse of `noises`. """
        raise NotImplementedError

    @online_compatible
    def logp(self, xs, ys, **kwargs):
        """ Compute the log probabilities on batches of (xs, ys)."""
        return np.log(self.predict(xs, **kwargs)==ys)  # default behavior

    def logp_grad(self, xs, ys, fs, **kwargs):
        """ Compute the \E[ f(x,y) \nabla log p(y|x) ] on batches of (xs, ys)."""

    @online_compatible
    def exp_fun(self, xs, *args, **kwargs):
        """ Compute the conditional expectation in closed-form."""
        raise NotImplementedError

    def exp_grad(self, xs, *args, **kwargs):
        """ Compute the derivative of conditional expectation in closed-form. """
        raise NotImplementedError

    # Some useful functions
    def kl(self, other, xs, reversesd=False, **kwargs):
        """ Computes KL(self||other), where other is another object of the
            same policy class. If reversed is True, return KL(other||self).
        """
        raise NotImplementedError

    def fvp(self, xs, gs, **kwargs):
        """ Computes F(self.pi)*g, where F is the Fisher information matrix and
        g is a np.ndarray in the same shape as self.variable. """
        raise NotImplementedError

