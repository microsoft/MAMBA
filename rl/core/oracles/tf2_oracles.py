# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import tensorflow as tf
from abc import abstractmethod
from rl.core.function_approximators.normalizers import NormalizerStd, Normalizer
from rl.core.oracles.oracle import Oracle
from rl.core.utils.tf2_utils import tf_float, ts_to_array, array_to_ts, var_assign
from rl.core.utils.misc_utils import flatten


class tfOracle(Oracle):
    """ A minimal wrapper of tensorflow functions. """

    def __init__(self, ts_fun, ts_var=None, **kwargs):
        """ If ts_var is None, the input needs to be provided explicitly to ts_fun. Otherwise,
            ts_fun takes no input except for some keyword arguments. """
        self._ts_fun = ts_fun  # a function that returns tf.Tensor(s)
        self._ts_var = ts_var

    def fun(self, x, **kwargs):
        return ts_to_array(self.ts_fun(array_to_ts(x)), **kwargs)

    def grad(self, x, **kwargs):
        return flatten(ts_to_array(self.ts_grad(array_to_ts(x), **kwargs)))

    def ts_fun(self, ts_x, **kwargs):
        if self._ts_var is None:
            return self._ts_fun(ts_x, **kwargs)
        else:
            var_assign(self._ts_var, ts_x)
            return self._ts_fun(**kwargs)

    def ts_grad(self, ts_x, **kwargs):
        """ If x is not provided, the cached value from the previous call of
         `grad` will be returned. """
        ts_var = self._ts_var or ts_x
        with tf.GradientTape() as tape:
            tape.watch(ts_var)
            ts_loss = self.ts_fun(ts_x, **kwargs)
        return tape.gradient(ts_loss, ts_var)

    def update(self, ts_fun=None, ts_var=None, *args, **kwargs):
        if ts_fun is not None:
            self._ts_fun = ts_fun
        if ts_var is not None:
            self._ts_var = ts_var


class tfLikelihoodRatioOracle(tfOracle):
    """
    An Oracle based on the loss function below: if use_log_loss is True

        E_{x} E_{y ~ q | x} [ w * log p(y|x) * f(x, y) ]

    otherwise, it uses

        E_{x} E_{y ~ q | x} [ p(y|x)/q(y|x) * f(x, y) ]

    where p is the variable distribution, q is a constant
    distribution, and f is a scalar function.

    When w = p/q, then the gradients of two loss functions are equivalent.

    The expectation is approximated by unbiased samples from q. To minimize
    the variance of sampled gradients, the implementation of 'grad' is
    based on a normalizer, which can shift, rescale, or clip f.

    """
    def __init__(self, ts_logp_fun, ts_var,
                 nor=None, biased=False,
                 use_log_loss=False, normalized_is=False):
        # ts_var needs to be prvided.
        assert use_log_loss in (True, False, None)
        self._ts_logp_fun = ts_logp_fun
        self._biased = biased
        self._use_log_loss = use_log_loss
        self._normalized_is = normalized_is  # normalized importance sampling
        if nor is None:
            if biased:  # use the current samples
                self._nor = NormalizerStd((1,), unscale=True, clip_thre=None, momentum=0.0)
            else:  # use a moving average
                self._nor = NormalizerStd((1,), unscale=True, clip_thre=None, momentum=None)
        else:
            assert isinstance(nor, Normalizer)
            self._nor = nor
        super().__init__(self.ts_surrogate_loss, ts_var)

    def ts_surrogate_loss(self):
        """ Return the loss function as tf.Tensor and a list of tf.plyeholders
        required to evaluate the loss function. """
        ts_f = self._ts_f
        ts_w_or_logq = self._ts_w_or_logq
        ts_logp = self._ts_logp_fun()  # the function part
        if tf.equal(self._use_log_loss, True):  # ts_w_or_logq is w
            ts_w = ts_w_or_logq
            ts_loss = tf.reduce_sum(ts_w * ts_f * ts_logp)
        elif tf.equal(self._use_log_loss, False): # ts_w_or_logq is logq
            ts_w = tf.exp(ts_logp - ts_w_or_logq)
            ts_loss = tf.reduce_sum(ts_w*ts_f)
        else:  # ts_w_or_logq is logq
            # Another implementation of `self._use_log_loss==False`
            ts_w = tf.stop_gradient(tf.exp(ts_logp - ts_w_or_logq))
            ts_loss = tf.reduce_sum(ts_w * ts_f * ts_logp)
        if self._normalized_is:  # normalized importance sampling
            return ts_loss / tf.reduce_sum(ts_w)
        else: # regular importance sampling
            return ts_loss / tf.cast(ts_f.shape[0], tf_float)

    def update(self, f, w_or_logq, update_nor=True, **kwargs):
        """ Update the function with Monte-Carlo samples.

            f: sampled function values
            w_or_logq: importance weight or the log probability of the sampling distribution
            update_nor: whether to update the normalizer using the current sample
        """
        super().update(**kwargs)
        if self._biased:
            self._nor.update(f)
        f_normalized = self._nor.normalize(f)  # cv
        if self._use_log_loss:  # ts_w_or_logq is w
            assert np.all(w_or_logq >= 0)
        # these are treated as constants
        assert f_normalized.shape==w_or_logq.shape
        self._ts_f = array_to_ts(f_normalized)
        self._ts_w_or_logq = array_to_ts(w_or_logq)
        if not self._biased and update_nor:
            self._nor.update(f)
