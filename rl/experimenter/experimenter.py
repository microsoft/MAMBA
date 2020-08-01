# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import functools, copy
import time, os
import numpy as np
from rl.algorithms import Algorithm
from rl.experimenter.mdps import MDP
from rl.core.utils.misc_utils import safe_assign, timed, set_randomseed
from rl.core.utils import logz


class Experimenter:

    def __init__(self, alg, mdp, ro_kwargs, mdp_test=None, ro_kwargs_test=None):
        """
            ro_kwargs is a dict with keys, 'min_n_samples', 'max_n_rollouts'

            mdp and ro_kwargs and be a MDP (or dict) or a list of MDPs (or dicts)
        """
        self.alg = safe_assign(alg, Algorithm)
        self.mdp = safe_assign(mdp, MDP)
        self.ro_kwargs = ro_kwargs
        self.mdp_test = mdp_test
        self.ro_kwargs_test = ro_kwargs_test
        self._n_samples = 0  # number of data points seen
        self._n_rollouts = 0
        self.best_policy = copy.deepcopy(self.alg.get_policy())
        self.best_performance = -float('Inf')

    def gen_ro(self, agent, mdp=None, ro_kwargs=None, initialize=False,
               prefix='', to_log=False, eval_mode=False):
        """ Run the agent in the mdp and return rollout statistics as a Dataset
            and the agent that collects it.

            mpds, ro_kwargs can be either a single instance or a list.
        """
        ro_kwargs = ro_kwargs or self.ro_kwargs
        mdp = mdp or self.mdp

        # Make mdp, ro_kwargs as lists
        if not isinstance(mdp, list):
            mdp = [mdp]
        if not isinstance(ro_kwargs, list):
            ro_kwargs = [ro_kwargs]
        if len(mdp)>1 and len(ro_kwargs)==1:
            ro_kwargs*=len(mdp)
        assert len(mdp)==len(ro_kwargs)

        # Run the agent and log statistics
        ros_all, agents_all = [], []
        avg_performance = 0.
        for i, (m, kw) in enumerate(zip(mdp, ro_kwargs)):
            if initialize:  # so deterministic behaviors can be realized.
                m.initialize()
            ros, agents = m.run(agent, **kw)
            ros_all.extend(ros)
            agents_all.extend(agents)

            # Log
            ro = functools.reduce(lambda x,y: x+y, ros)
            def scale_back_rws(rws): return [r/self.mdp.rw_scale for r in rws]
            if not eval_mode:
                self._n_rollouts += len(ro)
                self._n_samples += ro.n_samples
            if to_log:
                if len(mdp)>1:
                    prefix = 'MDP'+str(i)+'_'
                # current ro
                gamma = m.gamma
                sum_of_rewards = [ ((gamma**np.arange(len(r.rws)))*r.rws).sum() for r in ro]
                if gamma<1.:
                    avg_of_rewards = [ (1-gamma)*sr for sr, r in zip(sum_of_rewards, ro)]
                else:
                    avg_of_rewards = [ sr/len(r) for sr, r in zip(sum_of_rewards, ro)]

                # scale back the reward in logging
                sum_of_rewards = scale_back_rws(sum_of_rewards)
                avg_of_rewards = scale_back_rws(avg_of_rewards)

                # compute the statistics
                performance = np.mean(sum_of_rewards)
                performance_avg = np.mean(avg_of_rewards)
                rollout_lens = [len(rollout) for rollout in ro]
                n_samples = sum(rollout_lens)
                logz.log_tabular(prefix + "NumSamples", n_samples)
                logz.log_tabular(prefix + "NumberOfRollouts", len(ro))
                logz.log_tabular(prefix + "MeanAvgOfRewards", performance_avg)
                logz.log_tabular(prefix + "MeanSumOfRewards", performance)
                logz.log_tabular(prefix + "StdSumOfRewards", np.std(sum_of_rewards))
                logz.log_tabular(prefix + "MaxSumOfRewards", np.max(sum_of_rewards))
                logz.log_tabular(prefix + "MinSumOfRewards", np.min(sum_of_rewards))
                logz.log_tabular(prefix + "MeanRolloutLens", np.mean(rollout_lens))
                logz.log_tabular(prefix + "StdRolloutLens", np.std(rollout_lens))

                avg_performance+=performance/len(mdp)

        if to_log:  # total
            if avg_performance >= self.best_performance:
                self.best_policy = copy.deepcopy(self.alg.policy)
                self.best_performance = avg_performance
            logz.log_tabular(prefix + 'TotalNumberOfSamples', self._n_samples)
            logz.log_tabular(prefix + 'TotalNumberOfRollouts', self._n_rollouts)
            logz.log_tabular(prefix + 'BestSumOfRewards', self.best_performance)

        return ros_all, agents_all

    def run(self, n_itrs, pretrain=True, seed=None,
            save_freq=None, eval_freq=None, final_eval=False, final_save=True):

        eval_policy = eval_freq is not None
        save_policy = save_freq is not None

        if seed is not None:
            set_randomseed(seed)
            self.mdp.env.seed(seed)

        start_time = time.time()
        if pretrain:
            self.alg.pretrain(functools.partial(self.gen_ro, to_log=False))

        # Main loop
        for itr in range(n_itrs):
            logz.log_tabular("Time", time.time() - start_time)
            logz.log_tabular("Iteration", itr)

            if eval_policy:
                if itr % eval_freq == 0:
                    self._eval_policy()

            with timed('Generate env rollouts'):
                ros, agents = self.gen_ro(self.alg.agent('behavior'), to_log=not eval_policy)
            self.alg.update(ros, agents)

            if save_policy:
                if itr % save_freq == 0:
                    self._save_policy(self.alg.policy, itr)
            # dump log
            logz.dump_tabular()

        # Save the final policy.
        if final_eval:
            logz.log_tabular("Time", time.time() - start_time)
            logz.log_tabular("Iteration", itr+1)
            self._eval_policy()
            logz.dump_tabular()

        if final_save:
            self._save_policy(self.alg.policy, n_itrs)
            self._save_policy(self.best_policy, 'best')

    def _eval_policy(self):
        with timed('Evaluate policy performance'):
            self.gen_ro(self.alg.agent('target'),
                        mdp=self.mdp_test,
                        ro_kwargs=self.ro_kwargs_test,
                        initialize=True,
                        to_log=True,
                        eval_mode=True)

    def _save_policy(self, policy, suffix):
        path = os.path.join(logz.LOG.output_dir,'saved_policies')
        name = policy.name+'_'+str(suffix)
        policy.save(path, name=name)
