# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Licensed under the MIT License.

import numpy as np
from abc import ABC, abstractmethod
from rl.core.online_learners.utils import cg


class Prox(ABC):
    """An interface of strictly convex functions R."""

    def proxstep(self, g):
        # argmin <g,x> + R(x)
        return self.primalmap(-g)

    def dualfun(self, u):
        # sup_x <u,x> - R(x)
        x = self.primalmap(u)
        legendre = (u * x).sum() - self.fun(x)
        return legendre

    def grad(self, x):
        return self.dualmap(x)  # for convenience

    @abstractmethod
    def primalmap(self, u):
        # map the dual variable to the primal variable \nabla R*(u)
        pass

    @abstractmethod
    def dualmap(self, x):
        # map the primal variable to the dual variable \nabla R(x)
        pass

    @abstractmethod
    def fun(self, x):
        # return the function value R(x)
        pass

    @property
    @abstractmethod
    def size(self):
        # effective magnitude or size of R wrt Euclidean norm
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        # update the function definition
        pass

    @abstractmethod
    def __eq__(self, other):
        # check if two instances belong to the same family of functions that
        # are closed under addition and positive multiplication
        pass


class Quad(Prox):
    """
    An common interface for quadratic functions in the form of
        0.5 x' * M x
    The user needs to implement 'primalmap', 'dualmap', and 'update'
    """

    def fun(self, x):
        # 0.5 x' M x
        return 0.5 * (x * self.dualmap(x)).sum()

    def __eq__(self, other):
        return isinstance(other, Quad)


class DiagQuad(Quad):
    # 0.5 x' D x, where D is diagonal and provided as a 1d array or a scalar.

    def __init__(self, D=1.0):
        self._D = D

    def primalmap(self, u):
        return u / self._D

    def dualmap(self, x):
        return self._D * x

    def update(self, D):
        self._D = D

    @property
    def size(self):
        return np.sqrt(np.mean(self._D))


class MatQuad(Quad):
    # 1/2 x' M x, where M is a dense matrix

    def __init__(self, M=None, eps=1e-8):
        self._M = M
        self._eps = 1e-8

    def primalmap(self, u):
        eps = np.eye(u.size) * self._eps
        v = u if self._M is None else np.linalg.solve(self._M + eps, u)
        assert u.shape == v.shape
        return v

    def dualmap(self, x):
        y = x if self._M is None else np.matmul(self._M, x)
        assert x.shape == y.shape
        return y

    def update(self, M):
        self._M = M

    @property
    def size(self):
        return np.sqrt(np.mean(np.diagonal(self._M)))


class MvpQuad(Quad):
    # scale/2 x^\t M x, where M is given as a matrix-vector-product function 'mvp'
    # and 'scale' is a positive scalar multiplier.

    def __init__(self, mvp=None, scale=1.0, eps=1e-8, cg_iters=20, verbose=False, use_cache=False):
        self.scale = scale
        self._mvp = lambda x: x if mvp is None else mvp
        self._eps = 1e-8
        self._cg_iters = cg_iters
        self._verbose = verbose
        self._cache = None  # for efficiency
        self._use_cache = use_cache

    def primalmap(self, u):
        # try to reuse cache information for efficiency
        # NOTE this could cause potential bugs due to user modifying _mvp
        # outside, without calling update.
        if self._cache is not None and self._use_cache:
            ratio = u / self._cache['u']
            if np.allclose(ratio, ratio[0]):  # use the cached result
                print('--use cached primalmap for efficiency')
                return self._cache['x_'] * ratio[0] / self.scale

        # compute from scratch
        if np.isclose(np.linalg.norm(u), 0.0):
            x_ = x = np.zeros_like(u)
        else:  # cg has an NaN problem when grads_flat is 0
            def mvp(g): return self._mvp(g) + self._eps * g
            x_ = cg(mvp, u, cg_iters=self._cg_iters, verbose=self._verbose)
            x = x_ / self.scale  # more stable
        assert x.shape == u.shape
        assert np.isfinite(x).all()

        # update the cache
        self._cache = {'u': np.copy(u), 'x_': x_}

        return x

    def dualmap(self, x):
        if np.allclose(x, 0.):
            return 0. * x  # for robustness
        u = self._mvp(x) * self.scale
        assert u.shape == x.shape
        return u

    def update(self, mvp=None, scale=None):
        # update mvp
        if mvp is not None:
            self._mvp = mvp
            self._cache = None
        if scale is not None:
            self.scale = scale

    @property
    def size(self):
        return self.scale


class BregmanDivergence(Prox):
    # Bregman divergence derived from a strictly convex function R.
    # It is also a strictly convex function (and therefore a Prox object).

    def __init__(self, prox, center=0.0):
        self.prox = prox
        self.recenter(center)

    def proxstep(self, g):
        if isinstance(self.prox, Quad):  # more stable
            return self._y - self.prox.primalmap(g)
        else:
            return super().proxstep(g)

    def primalmap(self, u):
        # find x such that u = \nabla R(x) - v
        return self.prox.primalmap(u + self._v)

    def dualmap(self, x):
        # \nabla_x B(x || y) = \nabla R(x) - \nabla R(y)
        return self.prox.dualmap(x) - self._v

    def fun(self, x):
        # B(x || y) = R(x) - R(y) - <\nabla R(y), x-y>
        return self.prox.fun(x) - self._R - np.sum(self._v * (x - self._y))

    @property
    def size(self):
        return self.prox.size

    def update(self, *args, **kwargs):
        # update the distance generating function R
        self.prox.update(*args, **kwargs)
        self.recenter(self._y)

    def recenter(self, center):
        # update the center
        self._y = center  # center of the Bregman divergence
        self._v = self.prox.dualmap(self._y)
        self._R = self.prox.fun(self._y)

    def __eq__(self, other):
        if isinstance(other, BregmanDivergence):
            return self.prox == other._prox
        return self.prox == other
