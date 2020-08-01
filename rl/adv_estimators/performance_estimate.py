# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
from scipy import linalg as la


class PerformanceEstimate:
    """ A helper class for computing \lambda-weighted advantage/Q estimates.

        Given a function v_t, the advantage estimate is based on

            A_t = \gamma^t (1-\lambda)  \sum_{k=0}^\infty \lambda^k  A_{k,t}

        where
            A_{k,t} = c_t - v_t + \delta * V_{k,t+1}
            V_{k,t} = w_t * c_t
                      + \delta * w_t * w_{t+1} * c_{t+1} + ...
                      + \delta^{k-1} * w_t * ... * w_{t+k-1} * c_{t+k-1}
                      + \delta^k     * w_t * ... * w_{t+k-1} * v_{t+k}
            c is the instantaneous cost,
            w is the importance weight
            v is the baseline
            \gamma in [0, 1] is the original discount factor in the problem
            \lambda in [0,1] defines the \lambda-mixing of a family of estimates
            \delta in [0, \gamma] is the additional discount factor for variance reduction

        It also provides an estimate of the Q function, which is simply given as

            Q_t = A_t + v_t

        One can show this estimates is a \lambda-weighted version of different
        truncated Monte-Carlo Q-estimates. This Q function estimator is biased,
        when \delta != \gamma or \lambda != 1. But the advantage function is
        always biased, unless v_t is exactly the value function (though such
        bias does not matter when computing policy gradients).


        The feature of the estimator is determined by the following criteria:

        1) \delta==\gamma or \delta<\gamma:
            whether to use the same discount factor as \gamma in the problem's
            definition for estimating value function. Using smaller delta
            simplifes the estimation problem but introduces additional bias.

        2) \lambda==1 or  \lambda <1:
            whether to use Monte-Carlo rollouts or a \lambda-weighted estimate,
            which has larger bias but smaller vairance.

        3) w==1, or w==p(\pi*)/p(\pi):
            whether to use importance sampling to estimate the advantage function
            with respect to some other policy \pi* using the samples from the
            exploration policy \pi. The use of non-identity w can let A_{V, k}
            to estimate the advantage function with respect to \pi* even when the
            rollouts are collected by \pi.

        =========================================================================================
        Some examples (and their imposed constraints):

        1) Actor-Critic Family (\delta==\gamma) (w=1)
            a) \lambda==1, unbiased Monte-Carlo rollout with costs reshaped by some
                           arbitrary function V
            b) \lambda==0, basic Actor-Critic when V is the current value estimate
            c) \labmda in (0,1), lambda-weighted Actor-Critic, when V is the current
                                 value estimate.

        2) GAE Family (\delta<\gamma) (w=1)
            a) \gamma==1, (\delta, \lambda)-GAE estimator for undiscounted problems
                when V is the current value estimate (Schulmann et al., 2016)

            b) \gamma in (0,1], (\delta, \lambda)-GAE for \gamma-discounted
                problems, when V is the current value estimate

        3) PDE (Performance Difference Estimate) Family (w = p(\pi') / p(\pi) ):
            PDE builds an estimate of E_{d_\pi} (\nabla E_{\pi}) [ A_{\pi'} ]
            where A_{\pi'} is the (dis)advantage function wrt \pi', in which
            V is the value estimate of some arbitrary policy \pi', \lambda in [0,
            1] and \delta in [0, \gamma] are bias-variance.
    """

    def __init__(self, gamma, lambd=0., delta=None):
        delta = np.min([gamma, 0.9999]) if delta is None else np.min([delta, gamma])
        self.gamma = np.clip(gamma, 0., 1.)
        self.delta = np.clip(delta, 0., 1.)
        self.lambd = np.clip(lambd, 0., 1.)

    def dynamic_program(self, a, b, c, d, w):
        """ Compute the expression below recursibely from the end

              val_t = d^t ( a_t + \sum_{k=1}^infty c^k w_{t+1} ... w_{t+k} b_{t+k} )
                    = d^t (a_t + e_t )

            in which e_t is computed recursively from the end as

              e_t = \sum_{k=1}^infty c^k w_{t+1} ... w_{t+k} b_{t+k} )
                  = c w_{t+1} b_{t+1} + \sum_{k=2}^infty c^k w_{t+1} ... w_{t+k} b_{t+k} )
                  = c w_{t+1} b_{t+1} + c w_{t+1} \sum_{k=1}^infty c^k w_{t+1+1} ... w_{t+1+k} b_{t+1+k} )
                  = c w_{t+1} b_{t+1} + c w_{t+1} e_{t+1}
                  = c w_{t+1}(b_{t+1} + e_{t+1})

            where the boundary condition of e is zero.
        """
        assert a.shape==b.shape, 'Lengths of the two sequences do not match.'
        horizon = len(a)
        if type(w) is not np.ndarray:
            w = np.full_like(a, w)  # try to make it one
        e = np.zeros_like(a)  # last val is 0
        cw = c * w
        for i in reversed(range(1, len(e))):
            e[i-1] = cw[i] * (b[i] + e[i])
        val = (d**np.arange(horizon)) * (a + e)
        return val

    @staticmethod
    def shift_l(v, padding=0.):
        return np.append(v[1:], padding) if np.array(v).size > 1 else v

    def reshape_cost(self, c, V, done, w=1.0):
        v, v_next = V[:-1], V[1:]
        if done:
            v_next[-1] = c[-1]
        return w*(c[:-1] + self.delta*v_next) - v

    def adv(self, c, V, done, w=1., lambd=None, gamma=None):
        """ Compute A_t = \gamma^t (1-\lambda)  \sum_{k=0}^\infty \lambda^k  A_{k,t}.

            In implementationn, A_t is computed as

                A_t = x_t + (\lambda*\delta) * X_{t+1} + Y_t

            where x_t =     c_t - v_t + \delta *    v_{t+1}
                  X_t = (w*c)_t - v_t + \delta * (wv)_{t+1}
                  Y_t = \sum_{k=2}^\infty (\lambda*\delta)^k * w_{t+1} * ... * w_{t+k-1} X_{t+k}

            and the boundary condition is given by zero.

            `c`, `V` are of the same length, including the value at the
            terminal state.  If done is True, the last value of `c` is treated
            as the terminal cost.  Otherwise, the last value of V is used.  w
            can be int, float, or np.ndarray with length equal to len(c)-1.

        """
        assert c.shape == V.shape
        assert type(done) is bool
        if isinstance(w, np.ndarray):
            assert (len(w)==len(c)-1) and  len(w.shape)<=1
        lambd = self.lambd if lambd is None else lambd
        gamma = self.gamma if gamma is None else gamma

        delta_lambd = self.delta*lambd
        x = self.reshape_cost(c, V, done, w=1.0)
        X = self.reshape_cost(c, V, done, w=w)
        Y = self.shift_l(X) * delta_lambd  # this always pads 0
        a = x + Y
        b = Y
        c = delta_lambd
        d = gamma
        return self.dynamic_program(a, b, c, d, w)

    def qfn(self, c, V, done, w=1., lambd=None, gamma=None):
        return V[:-1] + self.adv(c, V, done, w, lambd, gamma)



class SimplePerformanceEstimate(PerformanceEstimate):
    """ A simplified version of PerformanceEstimate, without considering importance sampling.

        Given a function v_t, the advantage estimate is based on

            A_t = \gamma^t (1-\lambda) \sum_{k=0}^\infty \lambda^k  A_{k,t}

        where
            A_{k,t} = c_t - v_t + \delta * V_{k,t+1}
            V_{k,t} = c_t + ... + \delta^{k-1} * c_{t+k-1} + \delta^k * v_{t+k}
    """

    def adv(self, c, V, done, w=1., lambd=None, gamma=None):
        """
            A_t = TD_t + decay TD_{t+1} + decay^2 TD_{t+2} + ...

            where decay = lambd*gamma, and TD_t = c_t + delta*v_{t+1} - v_t
        """

        assert c.shape == V.shape
        assert type(done) is bool
        if isinstance(w, np.ndarray):
            assert (len(w)==len(c)-1) and  len(w.shape)<=1
        lambd = self.lambd if lambd is None else lambd
        gamma = self.gamma if gamma is None else gamma

        delta_lambd = self.delta*lambd
        td = self.reshape_cost(c, V, done, w=1.0)
        decay = delta_lambd ** np.arange(len(td))
        decay = np.triu(la.circulant(decay).T, k=0)
        adv = np.ravel(np.matmul(decay, td[:,None]))
        return (gamma**np.arange(len(adv))) * adv
