# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Licensed under the MIT License.

import copy
from rl.oracles.oracle import rlOracle

class SimulationOracle(rlOracle):

    def __init__(self,
                 base_oracle,  # it does not define the graph, rather uses the base_oracle
                 env,  # object like gym env
                 gen_ro,  # takes pi and logp
                 ):

        assert isinstance(base_oracle, rlOracle)
        super().__init__(base_oracle.policy)
        self._base_oracle = copy.deepcopy(base_oracle)
        self._env = env
        self._gen_ro = gen_ro
        self._sim_ro = None  # simulated rollouts

    def __deepcopy__(self, memo, exclude=None):
        exclude = [] if exclude is None else exclude
        exclude += ['_env']
        return super().__deepcopy__(memo=memo, exclude=exclude)

    def compute_loss(self):
        return self._base_oracle.compute_loss()

    def compute_grad(self):
        return self._base_oracle.compute_grad()

    @property
    def ro(self):
        return self._sim_ro

    def update(self, ro=None, update_nor=False, update_ae=False,
               update_pol_nor=False, weight=1.0, to_log=False):
        '''
        update_pol_nor:
            update the nor of the policy BEFORE base oracle update.
        update_nor:
            update the nor in the base oracle.
        update_ae:
            update ae in the base oracle update.
        '''
        # Update base oracle with simulated ro.
        self._sim_ro = self._gen_ro(self.policy.pi, self.policy.logp)
        if update_pol_nor:
            # Update policy nor using sim ro, it should go before base oracle update.
            self.policy.prepare_for_update(self._sim_ro.obs)
        # Update the base oracle.
        self._base_oracle.update(self._sim_ro, update_nor=update_nor, to_log=to_log)
        if update_ae and callable(getattr(self._base_oracle, 'update_ae', None)):
            self._base_oracle.update_ae(self._sim_ro)
