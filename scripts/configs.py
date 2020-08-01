# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy

# This file contains certain default configs that can used to compose new
# configs by overwriting certain fields in CONFIG.

def def_traj_config(c):
    c = copy.deepcopy(c)
    c['experimenter']['ro_kwargs']['max_n_rollouts'] = \
        c['experimenter']['ro_kwargs']['min_n_samples']/c['mdp']['horizon']
    c['experimenter']['ro_kwargs']['min_n_samples'] = None
    return c

config_cp = {
    'exp_name': 'cp',
    'mdp': {
        'envid': 'DartCartPole-v1',
        'horizon': 1000,
        'gamma': 1.0
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 100},
        'ro_kwargs': {'min_n_samples': 2000},
    },
}

config_cp_traj = def_traj_config(config_cp)

config_dip = {
    'exp_name': 'dip',
    'mdp': {
        'envid': 'DartDoubleInvertedPendulumEnv-v1',
        'horizon': 1000,
        'gamma': 1.0,
        'rw_scale': 0.01,
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 200},
        'ro_kwargs': {'min_n_samples': 2000},
    },
}

config_dip_traj = def_traj_config(config_dip)

config_hopper = {
    'exp_name': 'hopper',
    'mdp': {
        'envid': 'DartHopper-v1',
        'horizon': 1000,
        'gamma': 1.0,
        'rw_scale': 0.01,
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 200},
        'ro_kwargs': {'min_n_samples': 16000},
    },
}

config_hopper_traj = def_traj_config(config_hopper)

config_reacher = {
    'exp_name': 'reacher',
    'mdp': {
        'envid': 'DartReacher-v1',
        'horizon': 500,
        'gamma': 1.0,
        'rw_scale': 0.01,
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 500},
        'ro_kwargs': {'min_n_samples': 16000},
    },
}

config_reacher3d_traj = def_traj_config(config_reacher)


config_reacher3d = {
    'exp_name': 'reacher3d',
    'mdp': {
        'envid': 'DartReacher3d-v1',
        'horizon': 500,
        'gamma': 1.0,
        'rw_scale': 0.01,
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 500},
        'ro_kwargs': {'min_n_samples': 16000},
    },
}

config_reacher3d_traj = def_traj_config(config_reacher3d)

config_halfcheetah = {
    'exp_name': 'halfcheetah',
    'mdp': {
        'envid': 'DartHalfCheetah-v1',
        'horizon': 1000,
        'gamma': 1.0,
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 200},
        'ro_kwargs': {'min_n_samples': 16000},
    },
}

config_halfcheetah_traj = def_traj_config(config_halfcheetah)

config_dog = {
    'exp_name': 'dog',
    'mdp': {
        'envid': 'DartDog-v1',
        'horizon': 1000,
        'gamma': 1.0,
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 1000},
        'ro_kwargs': {'min_n_samples': 16000},
    },
}

config_dog_traj = def_traj_config(config_dog)

config_humanwalker = {
    'exp_name': 'humanwalker',
    'mdp': {
        'envid': 'DartHumanWalker-v1',
        'horizon': 300,
        'gamma': 1.0,
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 1000},
        'ro_kwargs': {'min_n_samples': 16000},
    },
}

config_humanwalker_traj = def_traj_config(config_humanwalker)

config_walker2d = {
    'exp_name': 'walker2d',
    'mdp': {
        'envid': 'DartWalker2d-v1',
        'horizon': 1000,
        'gamma': 1.0,
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 500},
        'ro_kwargs': {'min_n_samples': 16000},
        'rw_scale': 0.01,
    },
}

config_walker2d_traj = def_traj_config(config_walker2d)

config_walker3d = {
    'exp_name': 'walker3d',
    'mdp': {
        'envid': 'DartWalker3d-v1',
        'horizon': 1000,
        'gamma': 1.0,
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 1000},
        'ro_kwargs': {'min_n_samples': 16000},
    },
}

config_walker3d_traj = def_traj_config(config_walker3d)

config_snake = {
    'exp_name': 'snake',
    'mdp': {
        'envid': 'DartSnake7Link-v1',
        'horizon': 1000,
        'gamma': 1.0,
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 200},
        'ro_kwargs': {'min_n_samples': 16000},
    },
}

config_sanke_traj = def_traj_config(config_snake)


config_humanoid = {
    'exp_name': 'humanoid',
    'mdp': {
        'envid': 'Humanoid-v2',
        'horizon': 1000,
        'gamma': 1.0,
        'rw_scale': 0.01,
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 1000},
        'ro_kwargs': {'min_n_samples': 50000,
                      'max_n_rollouts': None},
    },
}

config_humanoid_traj = def_traj_config(config_humanoid)


config_bhumanoid = {
    'exp_name': 'bhumanoid',
    'mdp': {
        'envid': 'HumanoidPyBulletEnv-v0',
        'horizon': 1000,
        'gamma': 1.0,
        'rw_scale': 0.01,
    },
    'experimenter': {
        'run_kwargs': {'n_itrs': 1000},
        'ro_kwargs': {'min_n_samples': 50000,
                      'max_n_rollouts': None},
    },
}

config_bhumanoid_traj = def_traj_config(config_bhumanoid)


