# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
from rl.core.function_approximators.normalizers import NormalizerStd, Normalizer
from rl.core.oracles import Oracle


class LikelihoodRatioOracle(Oracle):
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
    def __init__(self, logp_fun, logp_grad,
                 nor=None, biased=False,
                 use_log_loss=False, normalized_is=False):
        """
            logp_fun: variable -> logp
            logp_grad: variable, f -> E[ f \nabla logp]
        """
        self._logp_fun = logp_fun
        self._logp_grad = logp_grad  # sum
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

    def fun(self, x):
        f = self._f
        w_or_logq = self._w_or_logq
        logp = self._logp_fun(x)
        if self._use_log_loss:  # w_or_logq is w
            w = w_or_logq
            loss = np.sum(w *f *logp)
        else: # w_or_logq is logq
            w = np.exp(logp - w_or_logq)
            loss = np.sum(w*f)
        if self._normalized_is:  # normalized importance sampling
            return loss / np.sum(w)
        else: # regular importance sampling
            return loss / f.shape[0]

    def grad(self, x):
        f = self._f
        w_or_logq = self._w_or_logq
        if self._use_log_loss:  # w_or_logq is w
            w = w_or_logq
        else: # w_or_logq is logq
            logp = self._logp_fun(x)
            w = np.exp(logp - w_or_logq)
        wf = w*f
        print('w',  w.min(), w.max(), w.mean())
        print('wf', wf.min(), wf.max(), wf.mean())
        grad = self._logp_grad(x, wf)  # sum
        if self._normalized_is:  # normalized importance sampling
            return grad / np.sum(w)
        else: # regular importance sampling
            return grad / f.shape[0]

    def update(self, f, w_or_logq, update_nor=True):
        """ Update the function with Monte-Carlo samples.

            f: sampled function values
            w_or_logq: importance weight or the log probability of the sampling distribution
            update_nor: whether to update the normalizer using the current sample
        """
        if self._biased:
            self._nor.update(f)
        f_normalized = self._nor.normalize(f)  # cv
        if self._use_log_loss:  # w_or_logq is w
            assert np.all(w_or_logq >= 0)
        # these are treated as constants
        assert f_normalized.shape==w_or_logq.shape
        self._f = f_normalized
        self._w_or_logq = w_or_logq
        if not self._biased and update_nor:
            self._nor.update(f)
