# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import functools
import numpy as np
from rl.algorithms.algorithm import Algorithm, PolicyAgent
from rl.algorithms.utils import get_learner
from rl.adv_estimators.advantage_estimator import ValueBasedAE
from rl.oracles.rl_oracles import ValueBasedPolicyGradient
from rl import online_learners as ol
from rl.policies import Policy
from rl.core.utils.misc_utils import timed
from rl.core.utils import logz


class PolicyGradient(Algorithm):
    """ Basic policy gradient method. """

    def __init__(self, policy, vfn,
                 optimizer='adam',
                 lr=1e-3, c=1e-3, max_kl=0.1,
                 horizon=None, gamma=1.0, delta=None, lambd=0.99,
                 max_n_batches=2,
                 n_warm_up_itrs=None,
                 n_pretrain_itrs=1):

        assert isinstance(policy, Policy)
        self.vfn = vfn
        self.policy = policy

        # Create online learner.
        scheduler = ol.scheduler.PowerScheduler(lr)
        self.learner = get_learner(optimizer, policy, scheduler, max_kl)
        self._optimizer = optimizer

        # Create oracle.
        self.ae = ValueBasedAE(policy, vfn, gamma=gamma, delta=delta, lambd=lambd,
                               horizon=horizon, use_is='one', max_n_batches=max_n_batches)
        self.oracle = ValueBasedPolicyGradient(policy, self.ae)

        # Misc.
        self._n_pretrain_itrs = n_pretrain_itrs
        if n_warm_up_itrs is None:
            n_warm_up_itrs = float('Inf')
        self._n_warm_up_itrs =n_warm_up_itrs
        self._itr = 0

    def get_policy(self):
        return self.policy

    def agent(self, mode):
        return PolicyAgent(self.policy)

    def pretrain(self, gen_ro):
        with timed('Pretraining'):
            for _ in range(self._n_pretrain_itrs):
                ros, _ = gen_ro(self.agent('behavior'))
                ro = self.merge(ros)
                self.oracle.update(ro, self.policy)
                self.policy.update(xs=ro['obs_short'])

    def update(self, ros, agents):
        # Aggregate data
        ro = self.merge(ros)

        # Update input normalizer for whitening
        if self._itr < self._n_warm_up_itrs:
            self.policy.update(xs=ro['obs_short'])

        with timed('Update oracle'):
            _, ev0, ev1 = self.oracle.update(ro, self.policy)

        with timed('Compute policy gradient'):
            g = self.oracle.grad(self.policy.variable)

        with timed('Policy update'):
            if isinstance(self.learner, ol.FisherOnlineOptimizer):
                if self._optimizer=='trpo_wl':  # use also the loss function
                    self.learner.update(g, ro=ro, policy=self.policy, loss_fun=self.oracle.fun)
                else:
                    self.learner.update(g, ro=ro, policy=self.policy)
            else:
                self.learner.update(g)
            self.policy.variable = self.learner.x

        # log
        logz.log_tabular('stepsize', self.learner.stepsize)
        if hasattr(self.policy,'lstd'):
            logz.log_tabular('std', np.mean(np.exp(self.policy.lstd)))
        logz.log_tabular('g_norm', np.linalg.norm(g))
        logz.log_tabular('ExplainVarianceBefore(AE)', ev0)
        logz.log_tabular('ExplainVarianceAfter(AE)', ev1)

        self._itr +=1

    @staticmethod
    def merge(ros):
        """ Merge a list of Dataset instances. """
        return functools.reduce(lambda x,y: x+y, ros)

