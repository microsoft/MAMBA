# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Licensed under the MIT License.

import numpy as np
import time

from core.utils.misc_utils import flatten, unflatten

def test_misc_utils():

    vs = [np.random.random((10,)), np.random.random((4,3))]
    v = flatten(vs)
    vs_u = unflatten(v, template=vs)
    assert all([np.all(np.isclose(v1-v2, 0.0)) for v1, v2 in zip(vs,vs_u)])
    vs_u = unflatten(v, shapes=[vv.shape for vv in vs])
    assert all([np.all(np.isclose(v1-v2, 0.0)) for v1, v2 in zip(vs,vs_u)])


if __name__=='__main__':

    t = time.time()
    for _ in range(1000):
        test_misc_utils()
    print(time.time()-t)
