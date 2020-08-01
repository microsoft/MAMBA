# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Licensed under the MIT License.

from abc import ABC, abstractmethod


class OnlineLearner(ABC):
    """ An abstract interface of iterative algorithms. """

    @abstractmethod
    def update(self, *args, **kwargs):
        """ Update the state given feedback. """

    @property
    @abstractmethod
    def decision(self):
        """ Return the (stochastic) decision. """
