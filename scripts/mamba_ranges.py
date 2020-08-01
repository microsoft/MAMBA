# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
from scripts import ranges as R


range_common = [
    [['seed'], [x * 100 for x in range(8)]],
]

# basic baseline

range_pg = [
    [['top_log_dir'], ['log_pg']],
    [['algorithm', 'lambd'], [0.9, 0.98, 1.00]],
    [['use_experts'], [False]],
]
range_pg = R.merge_ranges(range_common, range_pg)


def get_path(seed):
    return 'experts/DartCartPole-v1/'+str(seed)+'/saved_policies'

range_agg_all = [
    [['top_log_dir'], ['log_aggrevated']],
    [['algorithm', 'lambd'], [0.]],
    [['n_experts'], [1]],
    [['algorithm', 'strategy'], ['max']],
    [['algorithm', 'policy_as_expert'], [False]],
    [['expert_path'],[ get_path(seed*100) for seed in range(8)]]
]
range_agg_all = R.merge_ranges(range_common, range_agg_all)

range_aggrevated = [
    [['top_log_dir'], ['log_aggrevated_lambda']],
    [['algorithm', 'lambd'], [0, 0.1, 0.5, 0.9,]],
    [['n_experts'], [1]],
    [['algorithm', 'strategy'], ['max']],
    [['algorithm', 'policy_as_expert'], [False]],
]
range_aggrevated = R.merge_ranges(range_common, range_aggrevated)

# aggregation

range_softmax = [
    [['top_log_dir'], ['log_softmax']],
    [['algorithm', 'lambd'], [0., 0.1, 0.5, 0.9]],
    [['n_experts'], [1, 2, 4, 8]],
    [['algorithm', 'strategy'], [1.0]],
    [['algorithm', 'policy_as_expert'], [False]],
]
range_softmax = R.merge_ranges(range_common, range_softmax)

range_max = [
    [['top_log_dir'], ['log_max']],
    [['algorithm', 'lambd'], [0., 0.1, 0.5, 0.9]],
    [['n_experts'], [1, 2, 4, 8]],
    [['algorithm', 'strategy'], ['max']],
    [['algorithm', 'policy_as_expert'], [False]],
]
range_max = R.merge_ranges(range_common, range_max)

range_uniform = [
    [['top_log_dir'], ['log_uniform']],
    [['algorithm', 'lambd'], [0., 0.1, 0.5, 0.9]],
    [['n_experts'], [1, 2, 4, 8]],
    [['algorithm', 'strategy'], ['uniform']],
    [['algorithm', 'policy_as_expert'], [False]],
]
range_uniform = R.merge_ranges(range_common, range_uniform)

range_mean = [
    [['top_log_dir'], ['log_mean']],
    [['algorithm', 'lambd'], [0., 0.1, 0.5, 0.9]],
    [['n_experts'], [1, 2, 4, 8]],
    [['algorithm', 'strategy'], ['mean']],
    [['algorithm', 'policy_as_expert'], [False]],
]
range_mean = R.merge_ranges(range_common, range_mean)

# debug

range_debug = [
    [['top_log_dir'], ['log_debug']],
    [['seed'], [x * 100 for x in range(2)]],
    [['algorithm', 'lambd'], [0.9]],
    [['n_experts'], [8]]
]





