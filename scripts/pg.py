# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import tensorflow as tf
import numpy as np
from scripts.utils import parser as ps
from rl import experimenter as Exp
from rl.algorithms import PolicyGradient
from rl.core.function_approximators.policies.tf2_policies import RobustKerasMLPGassian
from rl.core.function_approximators.supervised_learners import SuperRobustKerasMLP

def main(c):

    # Setup logz and save c
    ps.configure_log(c)

    # Create mdp and fix randomness
    mdp = ps.setup_mdp(c['mdp'], c['seed'])

    # Create learnable objects
    ob_shape = mdp.ob_shape
    ac_shape = mdp.ac_shape
    if mdp.use_time_info:
        ob_shape = (np.prod(ob_shape)+1,)

    # Define the learner
    policy = RobustKerasMLPGassian(ob_shape, ac_shape, name='policy',
                                   init_lstd=c['init_lstd'],
                                   units=c['policy_units'])

    vfn = SuperRobustKerasMLP(ob_shape, (1,), name='value function',
                              units=c['value_units'])
    # Create algorithm
    alg = PolicyGradient(policy, vfn,
                         gamma=mdp.gamma, horizon=mdp.horizon,
                         **c['algorithm'])

    # Let's do some experiments!
    exp = Exp.Experimenter(alg, mdp, c['experimenter']['ro_kwargs'])
    exp.run(**c['experimenter']['run_kwargs'])


CONFIG = {
    'top_log_dir': 'log_pg',
    'exp_name': 'cp',
    'seed': 9,
    'mdp': {
        'envid': 'DartCartPole-v1',
        'horizon': 1000,  # the max length of rollouts in training
        'gamma': 1.0,
        'n_processes':1,
    },
    'experimenter': {
        'run_kwargs': {
            'n_itrs': 100,
            'pretrain': True,
            'final_eval': False,
            'save_freq': 5,
        },
        'ro_kwargs': {
            'min_n_samples': 2000,
            'max_n_rollouts': None,
            'force_cpu': False,
        },
    },
    'algorithm': {
        'optimizer':'adam',
        'lr':0.001,
        'max_kl':0.1,
        'delta':None,
        'lambd':0.99,
        'max_n_batches':2,
        'n_warm_up_itrs':None,
        'n_pretrain_itrs':1,
    },
    'policy_units': (64,),
    'value_units': (128,128),
    'init_lstd': -1,
}


if __name__ == '__main__':
    main(CONFIG)
