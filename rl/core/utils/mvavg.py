# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Licensed under the MIT License.

from abc import ABC, abstractmethod
import numpy as np


class MvAvg(ABC):
    """Online Moving average."""

    @abstractmethod
    def update(self, val, weight=1.0):
        """ Update the moving average. """

    @property
    @abstractmethod
    def val(self):
        """ The value of the current estimate / the moving average."""


class ExpMvAvg(MvAvg):
    """An estimator based on exponential moving average.

    The estimate after N calls is computed as
        val = (1-rate) \sum_{n=1}^N rate^{N-n} x_n / nor_N
    where nor_N is equal to (1-rate) \sum_{n=1}^N rate^{N-n}
    """

    def __init__(self, val, rate, weight=0.):
        self._val = val*weight if val is not None else 0.
        self._nor = weight  # sum of weights
        self.rate = rate

    def update(self, val, weight=1.):
        self._val = self.mvavg(self._val, val*weight, self.rate)
        self._nor = self.mvavg(self._nor, weight, self.rate)

    @property
    def val(self):
        return self._val/np.maximum(1e-8, self._nor)

    @staticmethod
    def mvavg(old, new, rate):
        return rate*old + (1.0-rate)*new


class MomentMvAvg(MvAvg):
    """ An estimator based on momentum.

        The estimate after N calls is computed as
            val = \sum_{n=1}^N rate^{N-n} x_n
        Namely, it is ExpMvAvg but without normalization.
    """
    def __init__(self, val, rate, weight=0.):
        self._val = val*weight if val is not None else 0.
        self.rate = rate

    def update(self, val, weight=1.0):
        self._val = self.mvavg(self._val, val, self.rate)

    @property
    def val(self):
        return self._val

    @staticmethod
    def mvavg(old, new, rate):
        return rate*old + new


class PolMvAvg(MvAvg):
    """ An estimator based on polynomially weighted moving average.

        The estimate after N calls is computed as
            val = \sum_{n=1}^N n^power x_n / nor_N
        where nor_N is equal to \sum_{n=1}^N n^power, and power is a parameter.
    """

    def __init__(self, val, power=0, weight=0.):
        self._val = val*weight if val is not None else 0.
        self._nor = weight
        self.power = power
        self._itr = 1

    def update(self, val, weight=1.0):
        self._val = self.mvavg(self._val, val*weight, self.power)
        self._nor = self.mvavg(self._nor, weight, self.power)
        self._itr += 1

    def mvavg(self, old, new, power):
        return old + new*self._itr**power

    @property
    def val(self):
        return self._val/np.maximum(1e-8, self._nor)
