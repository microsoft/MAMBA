# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Licensed under the MIT License.

import gym, os
from rl import envs
from rl.core.function_approximators import supervised_learners as Sup


# Env.
ENVID2MODELENV = {
    'DartCartPole-v1': envs.Cartpole,
    'DartHopper-v1': envs.Hopper,
    'DartSnake7Link-v1': envs.Snake,
    'DartWalker3d-v1': envs.Walker3d,
}


def create_sim_env(env, seed, inaccuracy=None, dyn_configs=None):
    """ Create an EnvWithModel object as a model of env."""
    if dyn_configs:
        # Learnable dynamics.
        st = env.env.get_state() if hasattr(env, 'env') else env.get_state()
        st_dim, ac_dim = len(st), env.action_space.shape[0]
        dyn_cls = getattr(Sup, dyn_configs['dyn_cls'])
        # build_nor = Nor.create_build_nor_from_str(dyn_configs['nor_cls'], dyn_configs['nor_kwargs'])
        dyn = dyn_cls(st_dim + ac_dim, st_dim, **dyn_configs['dyn_kwargs'])
        predict = dyn.predict
    else:
        predict = None

    envid = env.env.spec.id
    sim_env = ENVID2MODELENV[envid](env, predict=predict, model_inacc=inaccuracy, seed=seed)
    return sim_env
