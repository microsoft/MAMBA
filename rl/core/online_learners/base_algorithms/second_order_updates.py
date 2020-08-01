# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Licensed under the MIT License.

import numpy as np
from abc import abstractmethod
from rl.core.online_learners.base_algorithms import MirrorDescent
from rl.core.utils.mvavg import ExpMvAvg
from rl.core.online_learners.prox import MvpQuad


class SecondOrderUpdate(MirrorDescent):
    """
    The decision is computed
        h_new = h - stepsize * H^{-1} g
    where H is provided as a matrix-vector-product function
    and stepsize is given by the user-defined function _compute_stepsize.
    """

    def __init__(self, x0, scheduler, mvp=None,
                 eps=1e-5, cg_iters=20, verbose=False, use_cache=True, **kwargs):
        self._scheduler = scheduler
        prox = MvpQuad(mvp=mvp, scale=1.0 / self._scheduler.stepsize,
                       eps=eps, cg_iters=cg_iters,
                       verbose=verbose, use_cache=use_cache)
        super().__init__(x0, prox)

    def reset(self):
        self._scheduler.reset()
        self._breg.update(scale=1.0 / self._scheduler.stepsize)

    def shift(self, mvp=None):
        self._breg.update(mvp=mvp)  # update the distance generating function

    def adapt(self, g, w, mvp=None, **kwargs):
        self._breg.update(mvp=mvp)  # update the distance generating function
        self._scheduler.update(w)
        stepsize = self._compute_stepsize(g, w, **kwargs)
        self._breg.update(scale=1.0 / stepsize)

    def dualnorm2(self, g):  # 0.5 g H^{-1} g
        return np.abs(self._breg.prox.dualfun(g) * self._breg.size)

    def primalmap(self, g):  # H^{-1} g
        return self._breg.prox.primalmap(g) * self._breg.size

    @abstractmethod
    def _compute_stepsize(self, g, w, **kwargs):
        pass


class AdaptiveSecondOrderUpdate(SecondOrderUpdate):
    # use exponential moving average for adaptive stepsize

    def __init__(self, x0, scheduler, beta2=0.999, **kwargs):
        super().__init__(x0, scheduler, **kwargs)
        self._v = ExpMvAvg(0.0, beta2)

    def _compute_stepsize(self, g, w):
        self._v.update(self.dualnorm2(g))
        return self._scheduler.stepsize / (np.sqrt(self._v.val) + 1e-8)


class RobustAdaptiveSecondOrderUpdate(AdaptiveSecondOrderUpdate):
    def __init__(self, x0, scheduler, max_dist=None, ls_iters=10, ls_decay=0.5, **kwargs):
        AdaptiveSecondOrderUpdate.__init__(self, x0, scheduler, **kwargs)
        assert max_dist is not None  # needs to be provided
        self._max_dist = max_dist
        self._ls_iters = ls_iters
        self._ls_decay = ls_decay

    def _compute_stepsize(self, g, w, dist_fun=None):
        stepsize = AdaptiveSecondOrderUpdate._compute_stepsize(self, g, w)
        if dist_fun is None:  # just use the initial heuristic stepsize
            return stepsize
        # Back-tracking line search so KL divergence is within the limit
        direction = self.primalmap(g)
        for _ in range(self._ls_iters):
            h = self._h - direction * stepsize  # new point
            distance = dist_fun(h)
            if distance > self._max_dist:
                print("violated distance constraint. shrinking step.")
            else:
                print("Stepsize OK!")
                break
            stepsize *= self._ls_decay
        else:
            print("couldn't compute a good step")
            stepsize = 0.0

        return stepsize


class TrustRegionSecondOrderUpdate(SecondOrderUpdate):
    # use trust-region line search for adaptive stepsize

    def __init__(self, x0, scheduler, ls_iters=10, ls_decay=0.5, **kwargs):
        super().__init__(x0, scheduler, **kwargs)
        self._ls_iters = ls_iters
        self._ls_decay = ls_decay

    def _compute_stepsize(self, g, w, dist_fun=None, loss_fun=None):
        """
        Find a stepsize gamma such that h_new =  h - gamma H^{-1}g satisfies
            1) dist_fun(h_new) <= trust_region
            2) loss_fun(h_new) <= loss_fun(h)
        """
        assert loss_fun is not None or dist_fun is not None
        if loss_fun is None:
            def loss_fun(h): return 0.
        if dist_fun is None:  # just use the initial heuristic stepsize
            def dist_fun(h): return 0.

        # Update the size of trust_region.
        trust_region = w * self._scheduler.stepsize

        # Compute the initial stepsize (by assuming dist_fun is 0.5 dx^t H dx,
        # where dx = gamma * H^{-1} g ).
        stepsize = np.sqrt(trust_region / self.dualnorm2(g))
        stepsize = min(1.0, stepsize)  # make sure stepsize is less than 1.0

        # Back-tracking line search
        direction = self.primalmap(g)
        loss_before = loss_fun(self._h)
        expected_improve = g.dot(direction) * stepsize
        for _ in range(self._ls_iters):
            h = self._h - direction * stepsize  # new point
            loss_after, distance = loss_fun(h), dist_fun(h)
            assert np.isfinite([loss_after, distance]).all()
            improve = loss_before - loss_after
            print("Expected improve: %.3f Actual: %.3f" % (expected_improve, improve))
            if distance > trust_region:
                print("violated distance constraint. shrinking step.")
            elif improve < 0:
                print("surrogate didn't improve. shrinking step.")
            else:
                print("Stepsize OK!")
                break
            stepsize *= self._ls_decay
        else:
            print("couldn't compute a good step")
            stepsize = 0.0

        return stepsize
