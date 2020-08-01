# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
from scripts.mamba import CONFIG
from scripts import configs as dc
from rl.core.utils.misc_utils import dict_update

config_dip = copy.deepcopy(CONFIG)
config_dip = dict_update(config_dip, dc.config_dip_traj)
config_dip['experimenter']['ro_kwargs']['max_n_rollouts']=8

