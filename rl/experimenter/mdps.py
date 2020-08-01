# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import time, copy
import numpy as np
from functools import partial
from rl.core.datasets import Dataset
from rl.core.utils.mp_utils import Worker, JobRunner



def linear_t_state(t, horizon):
    return t/horizon

def rw_scaling(rw, ob, ac, scale):
    return rw*scale



class MDP:
    """ A wrapper for gym env. """
    def __init__(self, env, gamma=1.0, horizon=None, use_time_info=True,
                 v_end=None, rw_scale=1.0, n_processes=1, min_ro_per_process=1,
                 max_run_calls=None):
        self.env = env  # a gym-like env
        self.gamma = gamma
        horizon = float('Inf') if horizon is None else horizon
        self.horizon = horizon
        self.use_time_info = use_time_info
        self.rw_scale = rw_scale

        # configs for rollouts
        t_state = partial(linear_t_state, horizon=self.horizon) if use_time_info else None
        rw_shaping = partial(rw_scaling, scale=self.rw_scale)
        self._gen_ro = partial(self.generate_rollout,
                               env=self.env,
                               v_end=v_end,
                               rw_shaping= rw_shaping,
                               t_state=t_state,
                               max_rollout_len=horizon)
        self._n_processes = n_processes
        self._min_ro_per_process = int(max(1, min_ro_per_process))
        self._max_run_calls=max_run_calls  # for freeing memory

    def initialize(self):
        try:  # try to reset the env
            self.env.initialize()
        except:
            pass

    @property
    def ob_shape(self):
        return self.env.observation_space.shape

    @property
    def ac_shape(self):
        return self.env.action_space.shape

    def run(self, agent, min_n_samples=None, max_n_rollouts=None,
            force_cpu=False, with_animation=False):

        # default keywords
        kwargs = {'min_n_samples':min_n_samples,
                  'max_n_rollouts':max_n_rollouts,
                  'force_cpu':force_cpu,
                  'with_animation':with_animation}

        if self._n_processes>1: # parallel data collection
            if not hasattr(self, '_job_runner'):  # start the process
                workers = [Worker(method=self._gen_ro) for _ in range(self._n_processes)]
                self._job_runner = JobRunner(workers, max_run_calls=self._max_run_calls)
            # determine rollout configs
            N = self._n_processes  # number of jobs
            if max_n_rollouts is not None:
                N = int(np.ceil(max_n_rollouts/self._min_ro_per_process))
                max_n_rollouts = self._min_ro_per_process
            if min_n_samples is not None:
                min_n_samples = int(min_n_samples/N)
            kwargs['min_n_samples'] = min_n_samples
            kwargs['max_n_rollouts'] = max_n_rollouts
            kwargs['min_n_rollouts'] = self._min_ro_per_process
            # start data collection
            job = ((agent,), kwargs)
            res = self._job_runner.run([job]*N)
            ros, agents = [r[0] for r in res], [r[1] for r in res]
        else:
            ro, agent = self._gen_ro(agent, **kwargs)
            ros, agents = [ro], [agent]
        return ros, agents

    @staticmethod
    def generate_rollout(agent, *args, force_cpu=False, **kwargs):  # a wrapper
        if force_cpu:  # requires tensorflow
            import tensorflow as tf
            with tf.device('/device:CPU:0'):
                agent = copy.deepcopy(agent)
                ro = generate_rollout(agent.pi, agent.logp, *args,
                                      callback=agent.callback, **kwargs)
        else:
            ro = generate_rollout(agent.pi, agent.logp, *args,
                                  callback=agent.callback, **kwargs)
        return ro, agent

class Rollout:
    """ A container for storing statistics along a trajectory.

        The length of a rollout is determined by the number actions applied.
        The observations/states/rewards can contain in additional the
        information of the last step (i.e. having one more entry than the
        actions.)

    """
    def __init__(self, obs, acs, rws,  done, logp, weight=1.0):
        """
            `obs`, `acs`, `rws`  are lists of floats
            `done` is bool
            `logp` is a callable function or an nd.array

            `obs`, `rws` can be of length of `acs` or one element longer if they contain the
            terminal observation/reward.
        """
        self.__attrlist = []

        assert len(obs)==len(rws)
        assert (len(obs) == len(acs)+1) or (len(obs)==len(acs))
        self.obs = np.array(obs)
        self.acs = np.array(acs)
        self.rws = np.array(rws)
        self.dns = np.zeros((len(self)+1,))
        self.dns[-1] = float(done)
        if isinstance(logp, np.ndarray):
            assert len(logp)==len(acs)
            self.lps = logp
        else:
            self.lps = logp(self.obs[:len(self)], self.acs)
        self.weight = weight

    @property
    def obs_short(self):
        return self.obs[:len(self),:]

    @property
    def rws_short(self):
        return self.rws[:len(self)]

    @property
    def done(self):
        return bool(self.dns[-1])

    def __len__(self):
        return len(self.acs)

    def __setattr__(self, name, value):
        # Set per-rollout attributes, which are not sliced when self[ind] is called.
        if not name in ('_Rollout__attrlist','obs', 'acs', 'rws', 'lps', 'dns', 'lps'):
            self._Rollout__attrlist.append(name)
        object.__setattr__(self, name, value)

    def __getitem__(self, key):
        assert isinstance(key, slice) or isinstance(key, int)
        obs=self.obs[key]
        acs=self.acs[key]
        rws=self.rws[key]
        logp=self.lps[key]
        done = bool(self.dns[key][-1])
        rollout = Rollout(obs=obs, acs=acs, rws=rws, done=done, logp=logp)
        for name in self._Rollout__attrlist:
            setattr(rollout, name, copy.deepcopy(getattr(self, name)))
        return rollout

def generate_rollout(pi, logp, env,
                     callback=None,
                     v_end=None,
                     t_state=None,
                     rw_shaping=None,
                     min_n_samples=None,
                     max_n_rollouts=None,
                     min_n_rollouts=0,
                     max_rollout_len=None,
                     with_animation=False):

    """ Collect rollouts until we have enough samples or rollouts.

        Each rollout is generated by repeatedly calling the behavior `pi`. At
        the end of the rollout, the statistics (e.g. observations, actions) are
        packaged as a Rollout object and then `logp` is called **once** to save
        the log probability of the behavior policy `pi`.

        All rollouts are COMPLETE in that they never end prematurely, even when
        `min_n_samples` is reached. They end either when `done` is true, or
        `max_rollout_len` is reached, or `pi` returns None.

        Args:
            `pi`: the behavior policy, which takes (observation, time, done)
                  and returns the action or None. If None is returned, the
                  rollout terminates. done, here, is treated as special symbol
                  of state. If `pi` returns None, the rollout will be
                  terminated.

            `logp`: either None or a function that maps (obs, acs) to log
                    probabilities (called at end of each rollout)

            `env`: a gym-like environment

            `v_end`: the terminal value when the episoide ends (a callable
                     function of observation and done)

            `t_state`: a function that maps time to desired features

            `rw_shaping`: a function that maps a reward to the new reward

            `max_rollout_len`: the maximal length of a rollout (i.e. the problem's horizon)

            `min_n_samples`: the minimal number of samples to collect

            `max_n_rollouts`: the maximal number of rollouts,

            `min_n_rollouts`: the minimal number of rollouts,

            `with_animation`: display animiation of the first rollout

    """
    # Configs
    assert (min_n_samples is not None) or (max_n_rollouts is not None)  # so we can stop
    min_n_samples = min_n_samples or float('Inf')
    max_n_rollouts = max_n_rollouts or float('Inf')
    min_n_rollouts = min(min_n_rollouts, max_n_rollouts)
    max_rollout_len = max_rollout_len or float('Inf')
    max_episode_steps = getattr(env, '_max_episode_steps', float('Inf'))
    max_rollout_len = min(max_episode_steps, max_rollout_len)

    if v_end is None:
        def v_end(ob, dn): return 0.

    if rw_shaping is None:
        def rw_shaping(rw, ob, ac): return rw

    def post_process(x, t):
        # Augment observation with time information, if needed.
        return x if t_state is None else np.concatenate([x.flatten(), (t_state(t),)])

    def step(ac, tm):
        ob, rw, dn, info = env.step(ac)  # current reward, next ob and dn
        return post_process(ob, tm), rw, dn, info

    def reset(tm):
        ob = env.reset()
        return post_process(ob, tm)

    # Start trajectory-wise rollouts.
    n_samples = 0
    rollouts = []
    while True:
        animate_this_rollout = len(rollouts)==0 and with_animation
        obs, acs, rws, = [], [], []
        tm = 0  # time step
        dn = False
        ob = reset(tm)
        # each trajectory
        while True:
            if animate_this_rollout:
                env.render()
                time.sleep(0.05)
            ac = pi(ob, tm, dn) # apply action and get to the next state
            if ac is None:
                dn = False  # the learner decides to stop collecting data
                break
            # ob, ac, rw are at tm
            obs.append(ob)
            acs.append(ac)
            ob, rw, dn, _ = step(ac, tm)
            rw = rw_shaping(rw, ob, ac)
            rws.append(rw)
            tm += 1
            if dn or tm >= max_rollout_len:
                break # due to steps limit or entering an absorbing state
        # save the terminal observation/reward
        obs.append(ob)
        rws.append(v_end(ob, dn))  # terminal reward
        # end of one rollout (`logp` is called once)
        rollout = Rollout(obs=obs, acs=acs, rws=rws, done=dn, logp=logp)
        if callback is not None:
            callback(rollout)
        rollouts.append(rollout)
        n_samples += len(rollout)
        if (n_samples >= min_n_samples) or (len(rollouts) >= max_n_rollouts):
            if len(rollouts)>= min_n_rollouts:
                break
    ro = Dataset(rollouts)
    return ro
