# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import numpy as np
from rl.core.utils.misc_utils import zipsame

tf_float = tf.float32
tf_int = tf.int32

def none_to_zero(tensor, var):
    if type(tensor) is list:
        tensor = [tf.zeros_like(v) if t is None else t for v, t in zip(var, tensor)]
    else:
        tensor =  tf.zeros_like(var) if tensor is None else tensor
    return tensor

def identity(ts_x):
    return ts_x + tf.zeros_like(ts_x)

""" Conversion between tf.Tensor(s) and np.ndarray(s) """
def ts_to_array(ts_x):
    # convert tf.Tensor(s) to np.ndarray(s)
    if isinstance(ts_x, list) or isinstance(ts_x, tuple):
        return [ts_xx.numpy() if ts_xx is not None else None for ts_xx in ts_x]
    else:
        return ts_x.numpy()

def array_to_ts(x):
    # convert np.ndarray(s) to tf.Tensor(s)
    if isinstance(x, list) or isinstance(x, tuple):
        return [tf.constant(xx, dtype=tf_float) for xx in x]
    else:
        return tf.constant(x, dtype=tf_float)

def var_assign(ts_var, x):
    # convert np.ndarray(s) to tf.Tensor(s)
    if type(ts_var) is list:
        return [vv.assign(xx) for vv, xx in zipsame(ts_var, x)]
    else:
        return ts_var(x)
