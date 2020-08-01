# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Licensed under the MIT License.

import functools
import copy
from collections import deque
from rl.core import oracles as Or
from rl.oracles.oracle import rlOracle
from rl.experimenter import RO


class AggregatedOracle(rlOracle):
    """Aggregates rollouts."""

    def __init__(self, base_oracle, max_n_rollouts=None, max_n_samples=None, max_n_iterations=5):
        assert isinstance(base_oracle, rlOracle)
        self._base_oracle = copy.deepcopy(base_oracle)
        self._max_n_rollouts = max_n_rollouts
        self._max_n_samples = max_n_samples
        self._max_n_iterations = max_n_iterations
        if max_n_iterations is None:
            self._ro = None
        else:
            self._ro = deque([None for _ in range(max_n_iterations)])

        super().__init__(base_oracle.policy)

    @property
    def ro(self):
        if self._max_n_iterations is None:
            return self._ro
        else:  # queue
            rollouts = [r.rollouts for r in self._ro if r is not None]
            return RO(functools.reduce(lambda x, y: x + y, rollouts))

    def update(self, ro, *args, **kwargs):

        if self._max_n_iterations is None:
            if self._ro is None:
                self._ro = copy.deepcopy(ro)
                self._ro.max_n_rollouts = self._max_n_rollouts
                self._ro.max_n_samples = self._max_n_samples
            else:
                self._ro.append(ro.rollouts)
        else:  # queue
            self._ro.pop()
            self._ro.appendleft(ro)

        self._base_oracle.update(ro=self.ro, *args, **kwargs)

    def compute_loss(self):
        return self._base_oracle.compute_loss()

    def compute_grad(self):
        return self._base_oracle.compute_grad()


def _rlMetaOracleDecorator(cls):
    """A decorator for quickly defining new rlOracles from MetaOracle. """

    assert issubclass(cls, Or.MetaOracle) and not issubclass(cls, rlOracle)

    class decorated_cls(cls, rlOracle):
        def __init__(self, *args, **kwargs):
            cls.__init__(self, *args, **kwargs)
            # get policy from base oracle
            if hasattr(self, '_base_oracle'):
                assert isinstance(self._base_oracle, rlOracle)
                policy = self._base_oracle.policy
            elif hasattr(self, '_base_oracles'):
                assert len(self._base_oracles) > 0
                assert all([isinstance(bor, rlOracle) for bor in self._base_oracles])
                policy = self._base_oracles[0].policy
                assert all([bor.policy == policy for bor in self._base_oracles])
            else:
                raise TypeError('A MetaOracle should have _base_oracle or _base_oracles as attributes.')
            rlOracle.__init__(self, policy)

        @property
        def ro(self):
            if hasattr(self, '_base_oracle'):
                return self._base_oracle.ro
            elif hasattr(self, '_base_oracles'):
                ros = [ba.ro for ba in self._base_oracles]
                return functools.reduce(lambda a, b: a + b, ros)

        def update(self, ro, *args, **kwargs):
            cls.update(self, ro=ro, *args, **kwargs)

    # to make them look the same as intended
    decorated_cls.__name__ = cls.__name__
    decorated_cls.__qualname__ = cls.__qualname__
    return decorated_cls


@_rlMetaOracleDecorator
class LazyOracle(Or.LazyOracle):
    pass


@_rlMetaOracleDecorator
class DummyOracle(Or.DummyOracle):
    pass


@_rlMetaOracleDecorator
class AdversarialOracle(Or.AdversarialOracle):
    pass
