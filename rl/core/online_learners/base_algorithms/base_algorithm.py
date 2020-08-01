# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Licensed under the MIT License.

from abc import ABC, abstractmethod
import numpy as np
from rl.core.online_learners.prox import BregmanDivergence


class BaseAlgorithm(ABC):
    """
        A general descriptor of the online learning algorithms in terms of 4
        operators: shift, project, adapt, and update.  The user also needs to
        implement stepsize, which is an estimate of the effective stepsize
        taken by the algorithm.
    """

    def shift(self, **kwargs):
        # change the regularizer, potentially based on the current decision
        # NOTE this should only be used if necessary
        pass

    @abstractmethod
    def project(self):
        # decode the current decision from the memory
        pass

    @abstractmethod
    def adapt(self, g, w, **kwargs):
        # change the regularization based on g and w
        pass

    @abstractmethod
    def update(self, g, w):
        # update the memory based on g and w
        pass

    @property
    @abstractmethod
    def stepsize(self):
        # the effective scalar stepsize taken wrt Euclidean norm
        # for debugging
        pass


class MirrorDescent(BaseAlgorithm):
    """
    Update rules in the form of
        argmin_x  <g,x> + B_R(x||y)
    where B_R is Bregman divergence specified by a distance generating function R.

    Mirror descent algorithms all use the current decision as the internal
    state, so we can define the same project and update operators. Here we
    additional define 'set', 'proxstep', 'breg', 'breg_grad', 'dualfun' for
    convenience, as they are fundamental functions associated with mirror
    descent.

    The user needs to implement adapt and (shift)
    """

    def __init__(self, x0, prox):
        self._h = np.copy(x0)
        self._breg = BregmanDivergence(prox, self._h)

    def project(self):
        return np.copy(self._h)

    def update(self, g, w):
        self.set(self.proxstep(g * w))

    @property
    def stepsize(self):
        return 1. / self._breg.size

    # additional operators for convenience
    def set(self, x):
        self._h = np.copy(x)
        self._breg.recenter(self._h)

    def proxstep(self, g):
        return self._breg.proxstep(g)

    def bregfun(self, x):
        return self._breg.fun(x)

    def breggrad(self, x):
        return self._breg.grad(x)

    def dualfun(self, x):  # dual function of R
        return self._breg.prox.dualfun(x)

    @abstractmethod
    def adapt(self, g, w, **kwargs):
        # update self._breg
        pass
