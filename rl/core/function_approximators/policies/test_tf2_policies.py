# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import copy
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import functools
from rl.core.function_approximators import policies as POL
from rl.core.function_approximators.policies.tf2_policies import _RobustKerasMLPPolicy, _RobustKerasPolicy

def assert_array(a,b):
    assert np.all(np.isclose(a-b,0.0, atol=1e-6))

def test_copy(cls):
    x_shape = (10,2,3)
    y_shape = (3,)
    fun = cls(x_shape, y_shape)
    new_fun = copy.deepcopy(fun)
    new_fun.variable = fun.variable
    new_fun.variable = fun.variable+1
    assert all([np.all(np.isclose(v1-v2,1.0)) for v1, v2 in zip(new_fun.variable,fun.variable)])

def test_predict(cls):
    x_shape = (10,2,3)
    y_shape = (3,)
    fun = cls(x_shape, y_shape)
    x = np.random.random(fun.x_shape)
    fun(x)
    xs = np.random.random([10,]+list(fun.x_shape))
    fun.predict(xs)

def test_save_and_restore(cls):
    x_shape = (10,2,3)
    y_shape = (1,)

    xs = np.random.random([100]+list(x_shape))
    ys = np.random.random([100]+list(y_shape))
    fun1 = cls(x_shape, y_shape)
    fun1.update(xs)

    xs = np.random.random([100]+list(x_shape))
    ys = np.random.random([100]+list(y_shape))
    fun2 = cls(x_shape, y_shape)
    fun2.update(xs)

    import tempfile
    with tempfile.TemporaryDirectory() as path:
        fun1.save(path)
        fun2.restore(path)
        assert all([np.all(np.isclose(v1-v2,0.0)) for v1, v2 in zip(fun1.variable,fun2.variable)])
        assert_array(fun1.variable, fun2.variable)
        if hasattr(fun1,'_x_nor'):
            assert_array(fun1._x_nor._bias, fun2._x_nor._bias)


def build_kmodel1(x_shape, y_shape):
    # function approximator based on tf.keras.Model
    kmodel = tf.keras.Sequential()
    # Adds a densely-connected layer with 64 units to the kmodel:
    kmodel.add(layers.Dense(64, activation='relu'))
    # Add another:
    kmodel.add(layers.Dense(64, activation='relu'))
    # Add a softmax layer with 10 output units:
    kmodel.add(layers.Dense(y_shape[0]))
    return kmodel

def build_kmodel2(x_shape, y_shape):
    inputs = tf.keras.Input(shape=x_shape)
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    y = layers.Dense(y_shape[0])(x)
    kmodel = tf.keras.Model(inputs=inputs, outputs=y)
    return kmodel



class Tests(unittest.TestCase):

    """
    def test_robust_func_app(self):
        def test(cls):
            test_copy(cls)
            test_predict(cls)
            test_save_and_restore(cls)

            x_shape = (10,2,3)
            y_shape = (3,)
            xs = np.random.random([100]+list(x_shape))
            ys = np.random.random([100]+list(y_shape))

            fun = cls(x_shape, y_shape)
            fun.update(xs, ys)

            assert_array(fun._x_nor._bias, np.mean(xs, axis=0))
            assert_array(fun._y_nor._bias, np.mean(ys, axis=0))
            fun.predict(xs)

        test(functools.partial(_RobustKerasPolicy, build_kmodel=build_kmodel1))
        test(functools.partial(_RobustKerasPolicy, build_kmodel=build_kmodel2))
        test(lambda xsh, ysh: _RobustKerasPolicy(xsh, ysh, build_kmodel=build_kmodel1(xsh, ysh)))
        test(_RobustKerasMLPPolicy)
        test(functools.partial(POL.RobustKerasMLPGassian, init_lstd=1.0))
    """

    def test_gaussian(self):

       x_shape = (10,2,3)
       y_shape = (3,)
       xs = np.random.random([100]+list(x_shape))
       ys = np.random.random([100]+list(y_shape))
       pol = POL.RobustKerasMLPGassian(x_shape, y_shape,  init_lstd=np.random.random(y_shape))
       pol.lstd
       pol.ts_lstd

       ms = pol.mean(xs)
       assert ys.shape==ms.shape
       var = np.exp(pol.lstd*2)
       logp = -np.sum((ms-ys)**2/var/2, axis=1) - np.prod(y_shape)/2 * np.log(2*np.pi) - 0.5 *np.log(np.prod(var))
       logp2 = pol.logp(xs,ys)
       assert_array(logp, logp2)
       kls = pol.kl(pol, xs)
       assert_array(kls, 0)

       pol.fvp(xs, pol.variable)


if __name__ == '__main__':
    unittest.main()
