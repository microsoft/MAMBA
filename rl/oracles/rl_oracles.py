# Copyright (c) 2016 rllab contributors
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import copy
from functools import partial
from rl.adv_estimators.advantage_estimator import ValueBasedAE
from rl.oracles.oracle import rlOracle
from rl.core.oracles import LikelihoodRatioOracle
from rl.core.function_approximators.policies import Policy
from rl.core.datasets import Dataset


class ValueBasedPolicyGradient(rlOracle):
    """ A wrapper of LikelihoodRatioOracle for computing policy gradient of the type

            E_{d_\pi} (\nabla E_{\pi}) [ A_{\pi'} ]

        where \pi' is specified in ae.
    """
    def __init__(self, policy, ae,
                 use_is='one', avg_type='sum',
                 biased=True, use_log_loss=False, normalized_is=False):
        assert isinstance(ae, ValueBasedAE)
        self._ae = ae
        # define the internal oracle
        assert isinstance(policy, Policy)
        self._policy = copy.deepcopy(policy)  # just a template
        self._or = LikelihoodRatioOracle(
                    self._logp_fun, self._logp_grad,
                    biased=biased, # basic mvavg
                    use_log_loss=use_log_loss, normalized_is=normalized_is)
        # some configs for computing gradients
        assert use_is in ['one', 'multi', None]
        self._use_is = use_is  # use importance sampling for polcy gradient
        assert avg_type in ['avg', 'sum']
        self._avg_type = avg_type
        self._scale = None
        self._ro = None

    def _logp_fun(self, x):
        self._policy.variable = x
        return self._policy.logp(self.ro['obs_short'], self.ro['acs'])

    def _logp_grad(self, x, fs):
        self._policy.variable = x
        return self._policy.logp_grad(self.ro['obs_short'], self.ro['acs'], fs)

    @property
    def ro(self):
        return self._ro

    def fun(self, x):
        return self._or.fun(x) * self._scale

    def grad(self, x):
        return self._or.grad(x) * self._scale

    def update(self, ro, policy, update_nor=True, update_vfn=True, **kwargs):
        # Sync policies' parameters.
        self._policy.assign(policy) # NOTE sync BOTH variables and parameters
        # Compute adv.
        self._ro = ro
        advs, _ = self._ae.advs(self.ro, use_is=self._use_is)
        adv = np.concatenate(advs)
        self._scale = 1.0 if self._avg_type=='avg' else len(adv)/len(advs)
        # Update the loss function.
        if self._or._use_log_loss is True:
            #  - E_{ob} E_{ac ~ q | ob} [ w * log p(ac|ob) * adv(ob, ac) ]
            if self._use_is:  # consider importance weight
                w_or_logq = np.concatenate(self._ae.weights(ro, policy=self.policy))
            else:
                w_or_logq = np.ones_like(adv)
        else:  # False or None
            #  - E_{ob} E_{ac ~ q | ob} [ p(ac|ob)/q(ac|ob) * adv(ob, ac) ]
            assert self._use_is in ['one', 'multi']
            w_or_logq = ro['lps']
        # Update the LikelihoodRatioOracle.
        self._or.update(-adv, w_or_logq, update_nor=update_nor) # loss is negative reward
        # Update the value function at the end, so it's unbiased.
        if update_vfn:
            return self.update_vfn(ro, **kwargs)

    def update_vfn(self, ro, **kwargs):
        return self._ae.update(ro, **kwargs)


class ValuedBasedParameterExploringPolicyGradient(ValueBasedPolicyGradient):
    """ A wrapper of LikelihoodRatioOracle for computing policy gradient of the type

           \nabla p(\theta)   E_{d_\pi} E_{\pi} [ \sum r ]

        where p(\theta) is the distribution of policy parameters.
    """

    def __init__(self, distribution, *args, use_is=None, **kwargs):
        use_is = None  # importance sampling is not considered
        super().__init__(distribution, *args, use_is=use_is, **kwargs)
        self._distribution = self._policy  # so it's not confusing
        del self._policy
        assert self._use_is is None
        assert self._avg_type in ['sum']
        del self._avg_type
        self._scale = 1.0
        self.sampled_vars = None

    def _logp_fun(self, x):
        self._distribution.variable = x
        z = np.empty((len(self.sampled_vars),0))
        return self._distribution.logp(z, self.sampled_vars)

    def _logp_grad(self, x, fs):
        self._distribution.variable = x
        z = np.empty((len(self.sampled_vars),0))
        return self._distribution.logp_grad(z, self.sampled_vars, fs)

    def update(self, ro, distribution, update_nor=True, update_vfn=True, **kwargs):
        # NOTE we assume the sampled policies (given in `ro` as an attribute
        # `pol_var`) are i.i.d.  according to `distribution`, while the
        # behvaior policy which is used to collect the data could be different.
        # When `use_is` is "multi", the effects of behavior policy will be
        # corrected.

        # Make sure the sampled policy for a rollout is saved.
        self.sampled_vars = np.array([r.pol_var for r in ro])

        # Sync parameters.
        self._distribution.assign(distribution)
        # Compute adv.
        self._ro = ro
        def compute_adv(r):
            advs, _ = self._ae.advs([r], use_is=self._use_is)
            return advs[0][0] # we only concern the adv at the first time step
        adv = np.array([compute_adv(r) for r in self.ro])

        # Update the loss function.
        # NOTE the weight here is for `distribution`
        if self._or._use_log_loss:
            # - \E_{\theta} log p(\theta) E_{ob ~ p0} E_{ac ~ \pi | ob} [ adv(ob, ac) ]
            w_or_logq = np.ones_like(adv)  # w
        else:  # False or None
            # - \E_{\theta} E_{ob ~ p0} E_{ac ~ \pi | ob} [ adv(ob, ac) ]
            w_or_logq = self._logp_fun(self._distribution.variable)  # log_q

        # Update the LikelihoodRatioOracle.
        self._or.update(-adv, w_or_logq, update_nor=update_nor) # loss is negative reward
        # Update the value function at the end, so it's unbiased.
        if update_vfn:
            return self.update_vfn(ro, **kwargs)


class ValueBasedExpertGradientByRandomTimeSampling(rlOracle):
    """ A wrapper of LikelihoodRatioOracle for computing policy gradient of the type

            E_{d_\pi} (\nabla E_{\pi}) [ A_{\pi'} ]

        where \pi' is specified in ae.
    """
    def __init__(self, policy, ae,
                 use_is='one',
                 biased=True, use_log_loss=False, normalized_is=False):
        assert isinstance(ae, ValueBasedAE)
        self._ae = ae
        # Define the internal oracles
        assert isinstance(policy, Policy)
        self._policy = copy.deepcopy(policy)  # just a template
        def logp_fun(x, ro):
            self._policy.variable = x
            return self._policy.logp(ro['obs_short'], ro['acs'])
        def logp_grad(x, fs, ro):
            self._policy.variable = x
            return self._policy.logp_grad(ro['obs_short'], ro['acs'], fs)

        self._scale_or, self._scale_cv = 0., 0.
        self._ro_or, self._ro_cv = None, None
        # noisy gradient
        self._or = LikelihoodRatioOracle(
                    lambda var: logp_fun(var, self._ro_or),
                    lambda var, fs: logp_grad(var, fs, self._ro_or),
                    biased=biased, # basic mvavg
                    use_log_loss=use_log_loss, normalized_is=normalized_is)
        # another oracle for control variate's bias
        self._cv = LikelihoodRatioOracle(
                    lambda var: logp_fun(var, self._ro_cv),
                    lambda var, fs: logp_grad(var, fs, self._ro_cv),
                    biased=biased, # basic mvavg
                    use_log_loss=use_log_loss, normalized_is=normalized_is)
        # some configs for computing gradients
        assert use_is in ['one'] #, 'multi']
        self._use_is = use_is  # use importance sampling for polcy gradient

    @property
    def ro(self):
        return self._ro_or + self._ro_cv

    def fun(self, x):
        f1 = 0. if self._ro_exp is None else self._or.fun(x)*self._scale_or
        f2 = 0. if self._ro_pol is None else self._cv.fun(x)*self._scale_cv
        return f1+f2

    def grad(self, x):
        g1 = np.zeros_like(x) if self._ro_or is None \
                else self._or.grad(x)*self._scale_or
        g2 = np.zeros_like(x) if self._ro_cv is None \
                else self._cv.grad(x)*self._scale_cv
        print('noisy grad', np.linalg.norm(g1), 'cv grad', np.linalg.norm(g2))
        return g1+g2

    def update(self, ro_exp=None, ro_pol=None, policy=None, update_nor=True, **kwargs):
        """ Need to provide either `ro_exp` or `ro_pol`, and `policy`.

            `ro_exp` is used to compute an unbiased but noisy estimate of

                E_{pi}[\nabla \pi(s,a) \hat{A}_{\pi^*}(s,a)]

            when \hat{A}_{\pi^*} given by `self._or` is unbiased.

            `ro_pol` provides a biased gradient which can be used as a control
            variate (when `ro_exp` is provided) or just to define a biased
            oracle.
        """
        assert (ro_exp is not None) or (ro_pol is not None)
        assert policy is not None

        # Sync policies' parameters.
        self._policy.assign(policy) # NOTE sync BOTH variables and parameters
        # Update the oracles
        n_rollouts = len(ro_exp) if ro_pol is None else len(ro_pol)
        self._ro_or = None
        if ro_exp is not None:
            # compute adv
            if len(ro_exp)>0:
                advs, _ = self._ae.advs(ro_exp, use_is=self._use_is)
                advs = [a[0:1]*r.scale for a, r in zip(advs, ro_exp)]
                adv = np.concatenate(advs)
                if ro_pol is not None:  # compute the control variate
                    advs_cv, _ = self._ae.advs(ro_exp, use_is=self._use_is, lambd=0.)
                    advs_cv = [a[0:1]*r.scale for a,r  in zip(advs_cv, ro_exp)]
                    adv -= np.concatenate(advs_cv)
                logq = np.concatenate([r.lps[0:1] for r in ro_exp])
                # update noisy oracle
                self._scale_or = len(adv)/n_rollouts
                self._or.update(-adv, logq, update_nor=update_nor) # loss is negative reward
                self._ro_or = Dataset([r[0:1] for r in ro_exp])  # for defining logp

        self._ro_cv = None
        if ro_pol is not None:
            # update biased oracle
            advs, _ = self._ae.advs(ro_pol, use_is=self._use_is, lambd=0.)
            adv = np.concatenate(advs)
            self._scale_cv = len(adv)/n_rollouts
            logq = ro_pol['lps']
            self._cv.update(-adv, logq, update_nor=update_nor) # loss is negative reward
            self._ro_cv = ro_pol  # for defining logp

        # Update the value function at the end, so it's unbiased.
        if ro_exp is not None:
            return self._ae.update(ro_exp, **kwargs)
        else:  # when biased gradient is used
            return self._ae.update(ro_pol, **kwargs)


