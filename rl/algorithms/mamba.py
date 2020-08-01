# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
import numpy as np
from scipy.special import softmax
from functools import partial, reduce
from rl.algorithms import PolicyGradient
from rl.algorithms.algorithm import PolicyAgent, Agent
from rl.algorithms import utils as au
from rl import online_learners as ol
from rl.core.datasets import Dataset
from rl.core.utils.misc_utils import timed, zipsame
from rl.core.utils import logz
from rl.core.utils.mvavg import PolMvAvg
from rl.core.function_approximators import online_compatible
from rl.core.function_approximators.supervised_learners import SupervisedLearner

class Exploration:
    """ Maximization """
    def decision(self, xs, explore=False, **kwargs):
        """ Return the chosen arm. """
        pass


class EpsilonGreedy(Exploration):

    def __init__(self, x_shape, eps):
        self.x_shape = x_shape # for online
        self.eps = eps

    @online_compatible
    def decision(self, xs, models=None, explore=False, **kwargs):
        assert models is not None
        # epsilon-greedy choice (batch mode)
        N = len(xs)
        K = len(models)
        vals = [m(xs, **kwargs) for m in models]
        vals = np.concatenate(vals, axis=1)
        k_star = np.argmax(vals, axis=1)  # greedy action
        if explore:
            # The greedy choice is selected with probability (1-\eps)+eps/K.
            # Other choices are selected with probability eps/K.

            # Choose uniform actions with probability epsilon
            k_star_0 = k_star.copy()
            ind = np.where(np.random.rand(N)<=self.eps)
            k_star[ind] = np.random.randint(0,K,size=(len(ind),))
            # Compute the probability of the selected action
            prob = np.ones(k_star.shape)*(1-self.eps+self.eps/K)
            prob[ind] = self.eps/K
            prob[k_star[ind]==k_star_0[ind]] = 1-self.eps+self.eps/K
            return k_star, prob
        else:
            val_star = np.amax(vals, axis=1).reshape([-1,1])
            return k_star, val_star


class Uniform(Exploration):

    def __init__(self, x_shape, strategy='max'):
        self.x_shape = x_shape
        self.strategy = strategy
        assert strategy in ['max', 'uniform', 'mean'] or isinstance(strategy,float)

    @online_compatible
    def decision(self, xs, models=None, explore=False, **kwargs):
        assert models is not None
        # epsilon-greedy choice
        N = len(xs)
        K = len(models)
        vals = [m(xs, **kwargs) for m in models]
        vals = np.concatenate(vals, axis=1)
        k_star = np.random.randint(0,K,(N,))  # uniform random
        if explore:
            prob = np.ones(k_star.shape)/K
            return k_star, prob
        else:
            if self.strategy=='mean':  # return the average
                val_star = np.mean(vals, axis=1).reshape([-1,1])
            elif self.strategy=='max':  # return the max
                val_star = np.max(vals, axis=1).reshape([-1,1])
            elif self.strategy=='uniform':  # randomly return one
                val_star = vals.flatten()[k_star+np.arange(N)*K].reshape([-1,1])
            elif isinstance(self.strategy, float):  # softmax
                beta = self.strategy
                val_star = np.sum(softmax(vals*beta,axis=1)*vals, axis=1).reshape([-1,1])
            else:
                raise ValueError('Unknown strategy {}'.format(self.strategy))
            return k_star, val_star


class AggregatedValueFunction(SupervisedLearner):
    """ Statewise maximum over a set of value functions.

        It uses a contextual bandit algoithm to help learn the best expert to
        follow at a visited state
    """
    def __init__(self, vfns, eps=0.5, strategy='max', name='agg_vfn'):
        assert all([v.x_shape==vfns[0].x_shape for v in vfns])
        assert all([v.y_shape==vfns[0].y_shape for v in vfns])
        assert all([isinstance(v, SupervisedLearner) for v in vfns])
        super().__init__(vfns[0].x_shape, vfns[0].y_shape, name=name)

        self.explorer = Uniform(self.x_shape, strategy)
        # assert strategy in ['max', 'uniform', 'mean']
        # # TODO unified this part
        # if strategy=='uniform':
        #     self.explorer = Uniform(self.x_shape, deterministic=False)
        # elif strategy=='mean':
        #     self.explorer = Uniform(self.x_shape, deterministic=True)
        # elif strategy=='max':
        #     self.explorer = EpsilonGreedy(self.x_shape, eps=eps)
        # else:
        #     raise ValueError('Unknown strategy')
        self.vfns = vfns

    def decision(self, xs, explore=False, **kwargs):
        return self.explorer.decision(xs, models=self.vfns, explore=explore, **kwargs)

    def as_funcapp(self):
        new = super().as_funcapp()
        new.vfns = [vfn.as_funcapp() for vfn in new.vfns]
        return new

    def predict(self, xs, **kwargs):
        _, v_star = self.decision(xs, explore=False, **kwargs)
        return v_star

    def update(self, *args, k=None, **kwargs):  # overload
        # As vfns are already SupervisedLearners, we don't need to aggregate
        # data here.
        assert not k is None
        return self.vfns[k].update(*args, **kwargs)

    def update_funcapp(self, *args, k=None, **kwargs):
        pass

    @property
    def variable(self):
        return np.concatenate([v.variable for v in self.vfns])

    # TODO: this definition is note symmetric
    @variable.setter
    def variable(self, vals):
        [setattr(v,'variable',val) for v, val in zip(self.vfns, vals)]


class Mamba(PolicyGradient):
    """ Use max_k V^k as the value function. It overwrites the behavior policy. """

    def __init__(self, policy, vfn,
                 experts=None,
                 eps=0.5,  # for epsilon greedy
                 strategy='max',  # max, mean, or uniform
                 max_n_batches_experts=1000,  # for the experts
                 policy_as_expert=True,
                 mix_unroll_kwargs=None,  # extra kwargs for ExpertsAgent
                 **kwargs):

        if experts is None:
            experts = []
            policy_as_expert=True
            print('No expert is available. Use policy gradient.')

        # Define max over value functions
        vfns = [copy.deepcopy(vfn) for _ in range(len(experts))]
        if policy_as_expert:
            experts += [policy]
            vfns += [vfn]
            print('Using {} experts, including its own policy'.format(len(vfns)))
        else:
            print('Using {} experts'.format(len(vfns)))
        self.experts = experts
        vfn_agg = AggregatedValueFunction(vfns, eps=eps, strategy=strategy)  # max over values
        self.policy_as_expert = policy_as_expert

        # The main update is policy gradient but with vfn_agg as vfn.
        # The algorithm here is basically GAE with a general baseline function.
        # Therefore, we reuse most features from PolicyGradient but change the
        # update rule of the value estimate, because it's now the max over
        # experts' values.
        super().__init__(policy, vfn_agg, **kwargs)
        self.ae.update = None  # The update wrt vfn_agg should not be called

        # Create aes for manually updating value functions
        create_ae = partial(type(self.ae), gamma=self.ae.gamma,
                    delta=self.ae.delta, lambd=self.ae.lambd, use_is=self.ae.use_is)
        aes = []
        for i, (e,v) in enumerate(zip(experts, vfns)):
            if policy_as_expert and i==(len(experts)-1):  # policy's value
                aes.append(create_ae(e, v, max_n_batches=self.ae.max_n_batches))
            else:
                aes.append(create_ae(e, v, max_n_batches=max_n_batches_experts))
        self.aes = aes  # of the experts

        # For rollout
        self._avg_n_steps = PolMvAvg(1,weight=1)
        self.mix_unroll_kwargs = mix_unroll_kwargs or {}

    def pretrain(self, gen_ro):
        with timed('Pretraining'):
            for _ in range(self._n_pretrain_itrs):
                for k, expert in enumerate(self.experts):
                    ros, _ = gen_ro(PolicyAgent(expert))
                    ro = self.merge(ros)
                    _, ev0, ev1 = self.aes[k].update(ro)
                    print('pretrain explained variances', ev0, ev1)
                    self.policy.update(ro['obs_short'])

    def update(self, ros, agents):  # agents are behavior policies
        # Aggregate data
        data = [a.split(ro, self.policy_as_expert) for ro,a in zip(ros, agents)]
        ro_exps = [d[0] for d in data]
        ro_exps = list(map(list, zip(*ro_exps)))  # transpose, s.t. len(ro_exps)==len(self.experts)
        ro_exps = [self.merge(ros) for ros in ro_exps]
        ro_pol = [d[1] for d in data]
        ro_pol = self.merge(ro_pol)

        # Update input normalizer for whitening
        if self._itr < self._n_warm_up_itrs:
            ro = self.merge(ros)
            self.policy.update(xs=ro['obs_short'])

        with timed('Update oracle'):
            # Update the value function of the experts
            EV0, EV1 = [], []
            for k, ro_exp in enumerate(ro_exps):
                if len(ro_exp)>0:
                    _, ev0, ev1 = self.aes[k].update(ro_exp)
                    EV0.append(ev0)
                    EV1.append(ev1)
            # Update oracle
            self.oracle.update(ro_pol, update_vfn=False, policy=self.policy)

            # Update the value function the learner (after oracle update so it unbiased)
            if self.policy_as_expert:
                _, ev0, ev1 = self.aes[-1].update(ro_pol)

            # For adaptive sampling
            self._avg_n_steps.update(np.mean([len(r) for r in ro_pol]))

        with timed('Compute gradient'):
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

        # Log
        logz.log_tabular('stepsize', self.learner.stepsize)
        logz.log_tabular('std', np.mean(np.exp(2.*self.policy.lstd)))
        logz.log_tabular('g_norm', np.linalg.norm(g))
        if self.policy_as_expert:
            logz.log_tabular('ExplainVarianceBefore(AE)', ev0)
            logz.log_tabular('ExplainVarianceAfter(AE)', ev1)
        logz.log_tabular('MeanExplainVarianceBefore(AE)', np.mean(EV0))
        logz.log_tabular('MeanExplainVarianceAfter(AE)', np.mean(EV1))
        logz.log_tabular('NumberOfExpertRollouts', np.sum([len(ro) for ro in ro_exps]))
        logz.log_tabular('NumberOfLearnerRollouts', len(ro_pol))

        # Reset
        self._itr+=1

    def agent(self, mode):
        if mode=='target':
            return PolicyAgent(self.policy)
        elif mode=='behavior':
            return ExpertsAgent(self.policy,
                                self.experts,
                                self.vfn.as_funcapp(),
                                horizon=self.ae.horizon,
                                gamma=self.ae.gamma,
                                avg_n_steps=self._avg_n_steps.val,
                                **self.mix_unroll_kwargs)

class ExpertsAgent(Agent):
    """ An roll-in and roll-out agent useful for imitation learning.

        It alternates between two phases
          1) roll-in learner and roll-out expert for updating value function
          2) execute fully the learner for computing gradients

        NOTE The below implementation assume that `callback` is called "ONLY ONCE" at
        the end of each rollout.
    """

    def __init__(self,
                 policy,  # the learner policy
                 experts,  # a list of expert policies
                 vfn_agg,  # aggregation of the experts' value functions
                 horizon,  # problem horizon
                 gamma,   # discount factor
                 avg_n_steps,  # average number of steps the policy can survive
                 sampling_rule='geometric', # define how random switching time is generated
                 cyclic_rate=2, # the rate of forward training, relative to the number of iterations
                 ro_by_n_samples=False, # 'sample' or 'rollout'
                 ):

        self.policy = policy
        self.experts = experts
        self.vfn = vfn_agg

        # For defining swtiching time
        assert horizon<float('Inf') or gamma<1.
        self._setup = {'gamma':gamma, 'horizon':horizon}

        assert sampling_rule in ['geometric','cyclic','uniform']
        self._sampling_rule = sampling_rule
        self._cyclic_rate = cyclic_rate
        self._avg_n_steps = avg_n_steps  # number of steps the policy can survive
        self._ro_by_n_samples = ro_by_n_samples

        # For controlling data collection
        self._locked = False  #  free to call `pi`
        self._ro_with_policy = True  # in the phase of learner rollout
        self._n_samples_ro_pol = 0
        self._n_samples_ro_mix = 0

        # For splitting data
        self._n_ro = 0  # number of rollouts so far
        self._t_switch = []  # switching time
        self._scale = []  # extra scale (importance weight) due to sampling/switching
        self._k_star = []  # indices of the selected expert
        self._ind_ro_pol = []  # indices of learner rolluots
        self._ind_ro_mix = []  # indices of mixed rollouts

    def pi(self, ob, t, done):
        if t==0:  # make sure `callback` has been called
            assert not self._locked
            self._locked=True

        if self._ro_with_policy:  # just run the learner
            return self.policy(ob)
        else:  # roll-in policy and roll-out expert
            if t==0:  # sample t_switch in [1, horizon)   # NOTE why not starting from 0??
                if self._sampling_rule=='cyclic':
                    t_switch, scale = au.cyclic_t(self._cyclic_rate, **self._setup)
                elif self._sampling_rule=='geometric':
                    t_switch, scale = au.geometric_t(self._avg_n_steps, **self._setup)
                else:  # sampling according to the discount factor
                    t_switch, scale = au.natural_t(**self._setup)
                self._t_switch.append(t_switch)
                self._scale.append(scale)
                self._k_star.append(None)

            if t<self._t_switch[-1]:  # roll-in
                return self.policy(ob)
            else:
                if t==self._t_switch[-1]: # select the expert to rollout
                    k_star, prob = self.vfn.decision(ob, explore=True)
                    self._k_star[-1] = k_star
                    # rescale the importance based on the exploration probability
                    self._scale[-1] *= 1/len(self.experts)/prob
                k_star = self._k_star[-1]
                return self.experts[k_star](ob)

    def logp(self, obs, acs):
        # Does not change the state of the agent
        assert len(obs)==len(acs)
        if self._ro_with_policy or self._unfinished_mix(len(acs)):
            return  self.policy.logp(obs, acs)
        else:
            t_switch = self._t_switch[-1]
            k_star = self._k_star[-1]
            logp0 = self.policy.logp(obs[:t_switch], acs[:t_switch])
            logp1 = self.experts[k_star].logp(obs[t_switch:], acs[t_switch:])
            return np.concatenate([logp0, logp1])

    def _unfinished_mix(self, ro_len):
        """ ro_len = len(acs) """
        # the mixing did not really happen
        return not self._ro_with_policy and ro_len-1<self._t_switch[-1]

    def callback(self, rollout):
        """ Determine collection strategy for the next rollout. """
        assert len(self._t_switch)==len(self._scale)==len(self._k_star)
        # Log rollout statistics
        if self._ro_with_policy or self._unfinished_mix(len(rollout)):
            if self._unfinished_mix(len(rollout)):
                del self._k_star[-1]
                del self._t_switch[-1]
                del self._scale[-1]
            self._ind_ro_pol.append(self._n_ro)
            self._n_samples_ro_pol+=len(rollout)
            if self._ro_by_n_samples:
                self._ro_with_policy = self._n_samples_ro_pol<self._n_samples_ro_mix
            else:
                self._ro_with_policy = False
        else:
            self._ind_ro_mix.append(self._n_ro)
            self._n_samples_ro_mix+=len(rollout)
            self._ro_with_policy = True

        # unlock so `pi` can be called again
        self._locked =False
        self._n_ro+=1

    def split(self, ro, policy_as_expert):
        # Split ro into two phases
        rollouts = ro.to_list()
        ro_mix = [rollouts[i] for i in self._ind_ro_mix]
        ro_pol = [rollouts[i] for i in self._ind_ro_pol]
        assert (len(ro_mix)+len(ro_pol))==len(rollouts)
        ro_exps = [ [] for _ in range(len(self.experts))]
        for r, t, s, k in zipsame(ro_mix, self._t_switch, self._scale, self._k_star):
            assert len(r)>=t  # because t >= 1
            if not policy_as_expert or k<len(self.experts)-1:
                # we assume the last expert is the learner
                r = r[t:]
            r.weight = 1.0
            ro_exps[k].append(r)
        if policy_as_expert:
            ro_pol += ro_exps[-1]
            del ro_exps[-1]
        ro_exps = [Dataset(ro_exp) for ro_exp in ro_exps]
        ro_pol = Dataset(ro_pol)

        return ro_exps, ro_pol
