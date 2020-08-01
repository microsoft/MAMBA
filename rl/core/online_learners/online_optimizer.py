# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Licensed under the MIT License.

from abc import ABC, abstractmethod
import numpy as np
import copy
from rl.core.online_learners.online_learner import OnlineLearner
from rl.core.online_learners.base_algorithms import MirrorDescent, Adam, BaseAlgorithm
from rl.core.online_learners.scheduler import PowerScheduler
from rl.core.utils.misc_utils import cprint


class OnlineOptimizer(OnlineLearner):
    """ An easy-to-use interface of BaseAlgorithm for solving weighted online
        learning problems with full information or first-order feedbacks.

        The weight is n^p by default.
    """
    def __init__(self, base_alg, p=0.0, **kwargs):
        assert isinstance(base_alg, BaseAlgorithm)
        self._base_alg = base_alg  # a BaseAlgorithm object
        self._itr = 0  # starts with 0
        self._p = p  # the rate of the weight

    def reset(self):
        self._itr = 0
        self._base_alg.reset()

    @property
    def w(self):  # weighting for the loss sequence
        return self.get_w(self._itr)

    def get_w(self, itr):
        return itr**self._p

    @property
    def x(self):  # alias of decision
        return self._base_alg.project()

    @x.setter
    def x(self, val):
        assert isinstance(self._base_alg, MirrorDescent)
        self._base_alg.set(val)

    @property
    def stepsize(self):  # effective stepsize taken (for debugging)
        return self.w * self._base_alg.stepsize

    @abstractmethod
    def update(self, *args, **kwargs):
        # self._itr += 1  # starts a new round
        # update the decision with g wrt w
        pass

    @property
    def decision(self):
        return self.x


class BasicOnlineOptimizer(OnlineOptimizer):
    """ A online optimizer for adversarial linear problems. """

    def update(self, g, **kwargs):
        self._itr += 1  # starts a new round
        self._base_alg.adapt(g, self.w, **kwargs)
        self._base_alg.update(g, self.w)


class Piccolo(OnlineOptimizer):
    """ A reduction-based online optimizer for predictable linear problems. """

    def __init__(self, base_alg, p=0.0):
        super().__init__(base_alg, p)
        self._g_hat = None  # direction used in the prediction step

    @property
    def has_predicted(self):
        return self._g_hat is not None

    @property
    def g_hat(self):
        return np.zeros(self.x.shape) if self._g_hat is None else np.copy(self._g_hat)

    def clear_g_hat(self):
        self._g_hat = None

    def update(self, g, mode, **kwargs):
        # **kwargs are for the adapt operator of the base algorithm
        assert mode in ['predict', 'correct']
        if mode == 'predict':
            # a slight modification so that the size of hat_g is considered at
            # least once in the regularization
            kwargs['adapt'] = (not self.has_predicted) or kwargs.get('adapt', False)
            direction = self._predict(g, **kwargs)
            self._g_hat = direction
        elif mode == 'correct':
            self._itr += 1  # starts a new round
            direction = self._correct(g, **kwargs)

        assert direction is not None  # double check
        return direction

    def _correct(self, g, **kwargs):
        e = g if self._g_hat is None else g - self._g_hat
        self._base_alg.adapt(e, self.w, **kwargs)
        self._base_alg.update(e, self.w)
        return e

    def _predict(self, g_hat, adapt=False, **kwargs):  # NOTE this can be overloaded
        if adapt:
            self._base_alg.adapt(g_hat, self.w_next, **kwargs)
        else:
            self._base_alg.shift(**kwargs)
        self._base_alg.update(g_hat, self.w_next)
        return g_hat

    @property
    def w_next(self):
        return self.get_w(self._itr + 1)


class PiccoloOpt(Piccolo):
    """ An alternate version that (approxiately) solves an optimization problem
        in the prediction step.
    """

    def __init__(self, base_alg, p=0.0, n_steps=20):
        assert isinstance(base_alg, MirrorDescent)
        assert n_steps >= 0, str(n_steps)
        self._n_steps = n_steps
        super().__init__(base_alg, p)

    def _correct(self, g, grad_hat=None, loss_hat=None, callback=None,
                 warm_start=True, get_projection=None, **kwargs):
        return super()._correct(g, **kwargs)  # catch some unused key words

    def _predict(self, g_hat, adapt=False, grad_hat=None, loss_hat=None, callback=None,
                 warm_start=True, get_projection=None, **kwargs):
        # NOTE Assume callback is called after modifying the variable
        assert grad_hat is not None
        assert loss_hat is not None

        # grad_hat and loss_hat are functions
        # callback is used after every inner-loop update
        if adapt:
            self._base_alg.adapt(g_hat, self.w_next, **kwargs)
        else:
            self._base_alg.shift(**kwargs)

        # below finds the effective g_hat
        ###########################
        def fun(x): return self.w_next * loss_hat(x) + self._base_alg.bregfun(x)

        def jac(x): return self.w_next * grad_hat(x) + self._base_alg.breggrad(x)
        if callback is None:
            def callback(x): return None
        # warm start
        x0_hat = self._base_alg.proxstep(g_hat * self.w_next)
        x0 = x0_hat if warm_start else self.x
        if get_projection is None:
            def projection(x): return x
        else:
            projection = get_projection(x0)
        # solve for g_hat
        assert np.isclose(0., self._base_alg.bregfun(self.x))
        print('initial loss_hat {}, bregfun {}'.format(loss_hat(x0), self._base_alg.bregfun(x0)))
        new_x = self._solve(x0, fun, jac, callback)
        new_x = projection(new_x)
        if np.linalg.norm(new_x - x0) > 5*self._n_steps*np.linalg.norm(x0_hat - self.x):
            new_x = x0
            callback(new_x)  # need to reset grad_hat properly (almsot)
            cprint('VI problem diverges; reset to the initial condition')
        print('loss_hat {}, bregfun {}'.format(loss_hat(new_x), self._base_alg.bregfun(new_x)))
        ############################
        # redo the usual update rule
        g_hat = grad_hat(new_x)
        self._base_alg.update(g_hat, self.w_next)
        self._base_alg.set(new_x)  # force it to be consistent
        return g_hat

    @abstractmethod
    def _solve(self, x0, fun, jac, callback):
        #
        # Args:
        #   x0: initial condition
        #   fun: a function that takes the variable and returns the function value
        #   jac: a function that takes the variable and returns the gradient
        #   callback: a function that takes the variable and s called after every inner optimization step
        #
        # Return:
        #   x_opt: an approximate optimal solution that minimizes fun
        pass


class PiccoloOptBasic(PiccoloOpt):
    """ A version that uses a BasicOnlineOptimizer object to solve the inner
        optimization problem.
    """

    def __init__(self, base_alg, p=0.0, n_steps=20, method=None):
        super().__init__(base_alg, p=p, n_steps=n_steps)
        self.method = method  # it can be changed online

    def _solve(self, x0, fun, jac, callback):
        if self.method is None:
            balg = copy.deepcopy(self._base_alg)
            opt = BasicOnlineOptimizer(balg)
            opt.x = x0

        elif isinstance(self.method, Piccolo):
            opt = self.method
            opt.x = x0
            opt.clear_g_hat()
            opt.reset()

        elif type(self.method) is dict:
            _scheduler = PowerScheduler(**self.method)
            balg = Adam(x0, _scheduler)
            opt = BasicOnlineOptimizer(balg)

        else:
            raise ValueError('Unsupported method.')

        # just run the BasicOnlineOptimizer for n_steps iterations
        print('Running BasicOnlineOptimizer for {} iterations'.format(self._n_steps))
        x = x0
        callback(x)
        for i in range(self._n_steps):
            g = jac(x)
            kwargs = {}
            if isinstance(opt, Piccolo):
                kwargs['mode'] = 'correct'
                if hasattr(opt, 'ro'):
                    kwargs['ro'] = opt.ro

            opt.update(g, **kwargs)
            print('  inner stepsize {}'.format(opt.stepsize))

            x = opt.x
            callback(x)

        return x
