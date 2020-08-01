# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
from rl import online_learners as ol
from rl.online_learners import base_algorithms as balg


def get_learner(optimizer, policy, scheduler, max_kl=None):
    """ Return an first-order optimizer. """
    x0 = policy.variable
    if optimizer=='adam':
        return ol.BasicOnlineOptimizer(balg.Adam(x0, scheduler))
    elif optimizer=='natgrad':
        return ol.FisherOnlineOptimizer(
                    balg.AdaptiveSecondOrderUpdate(x0, scheduler),
                    policy=policy)
    elif optimizer=='rnatgrad':
        return ol.FisherOnlineOptimizer(
                    balg.RobustAdaptiveSecondOrderUpdate(x0, scheduler, max_dist=max_kl),
                    policy=policy)
    elif 'trpo' in optimizer:
        return ol.FisherOnlineOptimizer(
                    balg.TrustRegionSecondOrderUpdate(x0, scheduler),
                    policy=policy)
    else:
        raise NotImplementedError

# Different schemes for sampling the switch time step in RIRO
def natural_t(horizon, gamma):
    # Sampling according the problem's original weighting
    if horizon < float('Inf'):
        p0 = gamma**np.arange(horizon)
        sump0 = np.sum(p0)
        p0 = p0/sump0
        ind = np.random.multinomial(1,p0)
        t_switch = np.where(ind==1)[0][0]
        p = p0[t_switch-1]
    else:
        gamma = min(gamma, 0.999999)
        t_switch = np.random.geometric(p=1-gamma)[0]
        p = gamma**t_switch*(1-gamma)
    prob, scale = compute_prob_and_scale(t_switch, horizon, gamma)
    return t_switch, prob/p

def cyclic_t(rate, horizon, gamma):
    if getattr(cyclic_t, '_itr', None) is None:
        cyclic_t._itr = 0
    assert horizon < float('Inf')
    t_switch = (int(rate*cyclic_t._itr)%horizon)+1  # start from 1
    p = 1./horizon
    cyclic_t._itr +=1
    prob, scale = compute_prob_and_scale(t_switch, horizon, gamma)
    return t_switch, prob/p

def geometric_t(mean, horizon, gamma):
    prob = 1/mean
    t_switch = np.random.geometric(prob)  # starts from 1
    if t_switch>horizon-1:
        t_switch=horizon-1
        p = (1-prob)**t_switch  # tail probability
    else:
        p = (1-prob)**(t_switch-1)*prob
    prob, scale = compute_prob_and_scale(t_switch, horizon, gamma)
    return t_switch, prob/p

def compute_prob_and_scale(t, horizon, gamma):
    """ Treat the weighting in a problem as probability. Compute the
        probability for a time step and the sum of the weights.

        For the objective below,
            \sum_{t=0}^{T-1} \gamma^t c_t
        where T is finite and \gamma in [0,1], or T is infinite and gamma<1.
        It computes
            scale = \sum_{t=0}^{T-1} \gamma^t
            prob = \gamma^t / scale
    """
    assert t<=horizon-1
    if horizon < float('Inf'):
        p0 = gamma**np.arange(horizon)
        sump0 = np.sum(p0)
        prob = p0[t]/sump0
    else:
        sump0 = 1/(1-gamma)
        prob = gamma**t_switch*(1-gamma)
    return prob, sump0
