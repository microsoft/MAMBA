# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os, time, git, gym
import tensorflow as tf
import numpy as np
from rl.experimenter import MDP
from rl.core.utils import logz
from rl.core.utils.misc_utils import set_randomseed


try:
    import pybulletgym
except ImportError:
    pass


def configure_log(config, unique_log_dir=False):
    """ Configure output directory for logging. """

    # parse config to get log_dir
    top_log_dir = config['top_log_dir']
    log_dir = config['exp_name']
    seed = config['seed']

    # create dirs
    os.makedirs(top_log_dir, exist_ok=True)
    if unique_log_dir:
        log_dir += '_' + time.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(top_log_dir, log_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_dir = os.path.join(log_dir, '{}'.format(seed))
    os.makedirs(log_dir, exist_ok=True)

    # Log commit number.
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    config['git_commit_sha'] = sha

    # save config
    logz.configure_output_dir(log_dir)
    logz.save_params(config)

def create_env(envid, seed):
    env = gym.make(envid)
    env.seed(seed)
    return env

def setup_mdp(c, seed):
    """ Set seed and then create an MDP. """
    c = dict(c)
    envid = c['envid']
    env = create_env(envid, seed)
    tf.keras.backend.clear_session()
    # fix randomness
    set_randomseed(seed)
    del c['envid']
    mdp = MDP(env, **c)
    return mdp
