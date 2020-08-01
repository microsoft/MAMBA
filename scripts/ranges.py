# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy

def merge_ranges(a, b):
    # merge b into a
    a = copy.deepcopy(a)
    for rb in b:
        for ra in a:
            if all([sa==sb for sa, sb in zip(rb[0], ra[0])]):
                ra[1][:] = list(set(ra[1]+rb[1]))
                break
        a.append(rb)
    return a

range_common = [
    [['seed'], [x * 100 for x in range(4)]],
]




