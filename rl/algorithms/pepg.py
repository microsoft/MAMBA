# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import functools, copy
import numpy as np
from rl.algorithms.algorithm import Algorithm, PolicyAgent
from rl.algorithms.utils import get_learner
from rl.adv_estimators.advantage_estimator import ValueBasedAE
from rl.oracles.rl_oracles import ValuedBasedParameterExploringPolicyGradient
from rl import online_learners as ol
from rl.policies import Policy
from rl.core.utils.misc_utils import timed
from rl.core.utils import logz


class ParameterExploringPolicyGradient(Algorithm):
    """ Policy gradient based on parameter exploration. """

    # The codes share the same structure as PolicyGradient, so it has advanced
    # features like various adv estimates.

    def __init__(self, distribution,
                 policy, vfn,
                 optimizer='adam',
                 lr=1e-3, c=1e-3, max_kl=0.1,
                 horizon=None, gamma=1.0, delta=None, lambd=0.99,
                 max_n_batches=2,
                 n_warm_up_itrs=None,
                 n_pretrain_itrs=1):

        """
            `distribution` is where the variable of optimization is hosted and
            `policy` is used as a template (which hosts other non-variable
            parameters like moving average, if any). `distribution` is a Policy
            instance, whose `x_shape` is (0,) and `y_shape` is in the same
            shape as `policy.variable`.  `distribution` also needs to implement
            `mean` method so we can derandomize the algorithm to construct the
            final policy.
        """

        assert isinstance(distribution, Policy)
        assert isinstance(policy, Policy)
        assert distribution.x_shape==(0,)
        assert distribution.y_shape==policy.variable.shape
        self.distribution = distribution
        self.policy = policy  # just a template
        self.policy = self.get_policy()
        self.vfn = vfn

        # Create online learner.
        scheduler = ol.scheduler.PowerScheduler(lr, c=c)
        self.learner = get_learner(optimizer, distribution, scheduler, max_kl) # dist.
        self._optimizer = optimizer

        # Create oracle.
        self.ae = ValueBasedAE(policy, vfn, gamma=gamma, delta=delta, lambd=lambd,
                               horizon=horizon, use_is=None, max_n_batches=max_n_batches)
        self.oracle = ValuedBasedParameterExploringPolicyGradient(self.distribution, self.ae) # dist.

        # Misc.
        self._n_pretrain_itrs = n_pretrain_itrs
        if n_warm_up_itrs is None:
            n_warm_up_itrs = float('Inf')
        self._n_warm_up_itrs =n_warm_up_itrs
        self._itr = 0

    def get_policy(self):
        # we use `mean` to derandomize
        self.policy.variable = self.distribution.mean(np.empty((0,)))
        return self.policy

    def agent(self, mode):
        if mode=='behavior':
            return PEPolicyAgent(self.policy, self.distribution)
        elif mode=='target':
            return PolicyAgent(self.get_policy())

    def pretrain(self, gen_ro):
        with timed('Pretraining'):
            for _ in range(self._n_pretrain_itrs):
                ros, _ = gen_ro(self.agent('behavior'))
                ro = self.merge(ros)
                self.oracle.update(ro, self.distribution)  # dist.
                self.policy.update(xs=ro['obs_short'])

    def update(self, ros, agents):
        # Aggregate data
        ro = self.merge(ros)

        # Update input normalizer for whitening
        if self._itr < self._n_warm_up_itrs:
            self.policy.update(xs=ro['obs_short'])

        # Below we update `distribution` where the variables are hosted.
        with timed('Update oracle'):
            _, ev0, ev1 = self.oracle.update(ro, self.distribution)  # dist.

        with timed('Compute policy gradient'):
            g = self.oracle.grad(self.distribution.variable)  # dist.

        with timed('Policy update'):
            if isinstance(self.learner, ol.FisherOnlineOptimizer):
                if self._optimizer=='trpo_wl':  # use also the loss function
                    self.learner.update(g, ro=ro, policy=self.distribution, loss_fun=self.oracle.fun)  # dist.
                else:
                    self.learner.update(g, ro=ro, policy=self.distribution)  # dist.
            else:
                self.learner.update(g)
            self.distribution.variable = self.learner.x  # dist.

        # log
        logz.log_tabular('stepsize', self.learner.stepsize)
        if hasattr(self.distribution,'lstd'):
            logz.log_tabular('std', np.mean(np.exp(self.distribution.lstd)))
        logz.log_tabular('g_norm', np.linalg.norm(g))
        logz.log_tabular('ExplainVarianceBefore(AE)', ev0)
        logz.log_tabular('ExplainVarianceAfter(AE)', ev1)

        self._itr +=1

    @staticmethod
    def merge(ros):
        """ Merge a list of Dataset instances. """
        return functools.reduce(lambda x,y: x+y, ros)


class PEPolicyAgent(PolicyAgent):

    def __init__(self, policy, distribution):
        self.policy = policy
        self.distribution = distribution

    def _sample(self):
        return self.distribution(np.empty((0,)))

    def pi(self, ob, t, done):
        if t==0:
            self.policy.variable = self._sample()
        return self.policy(ob)

    def callback(self, ro):
        ro.pol_var = self.policy.variable
