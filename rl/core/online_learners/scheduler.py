# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Licensed under the MIT License.

from abc import ABC, abstractmethod
import numpy as np
import math


class Scheduler(ABC):
    # Interface for schedulers.

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @property
    @abstractmethod
    def stepsize(self):
        pass


class PowerScheduler(object):
    """
        A helper class for calculating the stepsize (i.e. the regularization
        constant) for a weighted online learning problem.
    """

    def __init__(self, eta, k=None, c=1e-3, p=0.0, N=200, limit=None):
        # It computes stepsize = \eta / (1+c*sum_w / sqrt{n}) / eta_nor, where
        # eta_nor is a normalization factor so that different choices of p are
        # comparable (p and N are only used in eta normalization).
        self._c = c  # how fast the learning rate decays
        self._k = k if k is not None else 0.5  # 0.5 for CVX, and 0.0 for SCVX
        self._eta = eta  # nominal stepsize
        self._eta_nor = self._compute_eta_nor(p, self._k, self._c, N)  # the constant normalizer of eta
        self._limit = eta if limit is None else limit
        self._w = 1
        self.reset()

    def reset(self):
        self._sum_w = 0
        self._itr = 0

    def update(self, w=1.0):
        self._w = w
        self._sum_w += w
        self._itr += 1

    @property
    def stepsize(self):
        stepsize = self._eta / (1.0 + self._c * self._sum_w / np.sqrt(self._itr + 1e-8)) / self._eta_nor
        if stepsize * self._w > self._limit:
            stepsize = self._limit / self._w
        return stepsize

    @staticmethod
    def _compute_eta_nor(p, k, c, N):
        # Compute the normalization constant for lr, s.t. the area under the
        # scheduling with arbitrary p is equal to that with p = 0.
        nn = np.arange(1, N, 0.01)

        def area_under(_p, _k, _c):
            return np.sum(nn**_p / (1.0 + _c * np.sum(nn**_p) / np.sqrt(nn**_k + 1e-8)))
        return area_under(p, k, c) / area_under(0, 0.5, c)


