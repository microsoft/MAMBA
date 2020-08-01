# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import copy
from functools import partial
import numpy as np
import tensorflow as tf
from abc import abstractmethod
from rl.core.function_approximators.function_approximator import FunctionApproximator
from rl.core.function_approximators.normalizers import tfNormalizerMax
from rl.core.utils.tf2_utils import array_to_ts, tf_float, ts_to_array, identity
from rl.core.utils.misc_utils import flatten, unflatten, zipsame
# NOTE ts_* methods are in batch mode
#      python methods return a single nd.array
#      ts_variables is a list of tf.Variables


class tfFuncApp(FunctionApproximator):
    """ A minimal wrapper for tensorflow 2 operators.

        The user needs to define `ts_predict`and `ts_variables`.

        (Everything else should work out of the box, because of tensorflow 2.)
    """
    def __init__(self, x_shape, y_shape, name='tf_func_app', **kwargs):
        self._var_shapes = None  # cache
        super().__init__(x_shape, y_shape, name=name, **kwargs)

    def predict(self, xs, **kwargs):
        return self.ts_predict(array_to_ts(xs), **kwargs).numpy()

    def grad(self, xs, **kwargs):
        """ Derivative with respect to xs. """
        return ts_to_array(self.ts_grad(array_to_ts(xs), **kwargs))

    @property
    def variable(self):
        return self.flatten(self.variables)

    @variable.setter
    def variable(self, val):
        self.variables = self.unflatten(val)

    def assign(self, other, excludes=()):
        ts_vars = [k for k,v in self.__dict__.items() if isinstance(v, tf.Variable)]
        excludes = list(excludes)+ts_vars
        super().assign(other, excludes)
        [getattr(self,k).assign(getattr(other,k)) for k in ts_vars]

    # Users can choose to implement `update`.

    # New methods of tfFuncApp

    def ts_grad(self, ts_xs, **kwargs):
        with tf.GradientTape() as gt:
            gt.watch(ts_xs)
            ts_fun = self.ts_predict(ts_xs, **kwargs)
        return gt.gradient(ts_fun, ts_xs)

    @property
    def variables(self):
        return [var.numpy() for var in self.ts_variables]

    @variables.setter
    def variables(self, vals):  # vals can be a list of nd.array or tf.Tensor
        [var.assign(val) for var, val in zipsame(self.ts_variables, vals)]

    @property
    def var_shapes(self):
        if self._var_shapes is None:
            self._var_shapes = [var.shape.as_list() for var in self.ts_variables]
        return self._var_shapes

    # helps functions
    def flatten(self, vals):
        return flatten(vals)

    def unflatten(self, val):
        return unflatten(val, shapes=self.var_shapes)

    # required implementation
    @property
    @abstractmethod
    def ts_variables(self):
        """ Return a list of tf.Variables """

    @abstractmethod
    def ts_predict(self, ts_xs, **kwargs):
        """ Define the tf operators for predict """


class KerasFuncApp(tfFuncApp):
    """
        A wrapper of tf.keras.Model.

        It is a FunctionApproximator with an additional attribute `kmodel`,
        which is a tf.keras.Model.

        It adds a new method `k_predict` which calls tf.keras.Model.preidct.

        When inheriting this class, users can choose to implement the
        `_build_kmodel` method, for ease of implementation. `build_kmodel` can be
        used to create necessary tf.keras.Layer or tf.Tensor to help defining
        the kmodel. Note all attributes created, if any, should be deepcopy/pickle
        compatible.

        Otherwise, a tf.keras.Model or a method, which shares the same
        signature of `_build_kmodel`, can be passed in __init__ .

    """
    def __init__(self, x_shape, y_shape, name='keras_func_app',
                 build_kmodel=None, **kwargs):
        """ Build an initialized tf.keras.Model as the function approximator

            `build_kmodel` can be  keras.Model or a method that shares the
            signature of `self._build_kmodel`.
        """
        super().__init__(x_shape, y_shape, name=name, **kwargs)
        # decide how to build the kmodel
        if isinstance(build_kmodel, tf.keras.Model):
            self.kmodel = build_kmodel
        else:
            build_kmodel = build_kmodel or self._build_kmodel
            self.kmodel = build_kmodel(self.x_shape, self.y_shape)
        # make sure the model is constructed
        ts_x = tf.zeros([1]+list(self.x_shape))
        self.kmodel(ts_x)

    def _build_kmodel(self, x_shape, y_shape):
        """ Build the default kmodel.

            Users are free to create additional attributes, which are
            tf.keras.Model, tf.keras.Layer, tf.Variable, etc., to help
            construct the overall function approximator. At the end, the
            function should output a tf.keras.Model, which is the overall
            function approximator.
        """
        raise NotImplementedError

    # required methods of tfFuncApp
    def ts_predict(self, ts_xs, **kwargs):
        return self.kmodel(ts_xs)

    @property
    def ts_variables(self):
        return self.kmodel.trainable_variables

    # New methods of KerasFuncApp

    def k_predict(self, xs, **kwargs):
        """ Batch prediction using the keras implementation. """
        return self.kmodel.predict(xs, **kwargs)

    # utilities (tf.keras.Model needs to be serialized)
    def assign(self, other, excludes=()):
        """ Set the parameters as others. """
        super().assign(other, excludes=('kmodel',))
        self.variable = other.variable

    def __copy__(self):
        # need to overwrite; otherwise it calls __getstate__
        new = type(self).__new__(type(self))
        new.__dict__.update(self.__dict__)
        return new

    def __getstate__(self):
        if hasattr(super(), '__getstate__'):
            d = super().__getstate__()
        else:
            d = self.__dict__
        d = dict(d)
        del d['kmodel']
        d['kmodel_config'] = self.kmodel.get_config()
        d['kmodel_weights'] = self.kmodel.get_weights()
        return d

    def __setstate__(self, d):
        d = dict(d)
        weights = d['kmodel_weights']
        config = d['kmodel_config']
        del d['kmodel_weights']
        del d['kmodel_config']
        if hasattr(super(), '__setstate__'):
            super().__setstate__(d)
        else:
            self.__dict__.update(d)
        try:
            self.kmodel = tf.keras.Model.from_config(config)
        except KeyError:
            self.kmodel = tf.keras.Sequential.from_config(config)
        # intialize the weights (keras bug...)
        ts_x = tf.zeros([1]+list(self.x_shape))
        self.kmodel(ts_x)  #self.ts_predict(ts_x)
        self.kmodel.set_weights(weights)


class tfRobustFuncApp(tfFuncApp):
    """ A function approximator with input and output normalizers.

        This class can be viewed as a wrapper in inheritance.  For example, for
        any subclass `A` of `tfFuncApp`, we can create a robust subclass `B` by
        simply defining

            class B(tfRobustFuncApp, A):
                pass
    """

    def __init__(self, x_shape, y_shape, name='tf_robust_func_app',
                 x_nor=None, y_nor=None, **kwargs):
        self._x_nor = x_nor or tfNormalizerMax(x_shape, unscale=False, \
                                    unbias=False, clip_thre=None, rate=0., momentum=None)
        self._y_nor = y_nor or tfNormalizerMax(y_shape, unscale=True, \
                                    unbias=True,  clip_thre=5.0, rate=0., momentum=None)
        # Normalizers are created first, as __init__ might call `predict`.
        super().__init__(x_shape, y_shape, name=name, **kwargs)

    def predict(self, xs, clip_y=True, **kwargs):  # add a new keyword
        return super().predict(xs, clip_y=clip_y, **kwargs)

    def ts_predict(self, ts_xs, clip_y=True, **kwargs):
        # include also input and output normalizeations
        ts_xs = self._x_nor.ts_predict(ts_xs)
        ts_ys = super().ts_predict(ts_xs)
        return self._y_nor.ts_predict(ts_ys) if clip_y else ts_ys

    def update(self, xs=None, ys=None, *args, **kwargs):
        print('Update normalizers of {}'.format(self.name))
        if xs is not None:
            self._x_nor.update(xs)
        if ys is not None:
            self._y_nor.update(ys)
        return super().update(xs=xs, ys=ys, *args, **kwargs)


class RobustKerasFuncApp(tfRobustFuncApp, KerasFuncApp):

    def __init__(self, x_shape, y_shape, name='robust_k_func_app', **kwargs):
        super().__init__(x_shape, y_shape, name=name, **kwargs)

    def k_predict(self, xs, clip_y=True, **kwargs):  # take care of this new method
        xs = self._x_nor(xs)
        ys = super().k_predict(xs)
        return self._y_nor(ys) if clip_y else ys


# Some examples


def _keras_mlp(x_shape, y_shape,
                units=(),
                activation='tanh',
                hidden_layer_init_scale=1.0,
                output_layer_init_scale=1.0,
                init_distribution='uniform'):

        initializer = partial(tf.keras.initializers.VarianceScaling,
                      mode='fan_avg', distribution=init_distribution)

        ts_in = tf.keras.Input(x_shape)
        ts_xs = tf.keras.layers.Reshape((np.prod(x_shape),))(ts_in)
        # build the hidden layers
        for unit in units:
            init = initializer(scale=hidden_layer_init_scale**2)
            ts_xs = tf.keras.layers.Dense(unit, activation=activation,
                        kernel_initializer=init)(ts_xs)
        # build the last linear layer
        init = initializer(scale=output_layer_init_scale**2)
        ts_ys = tf.keras.layers.Dense(np.prod(y_shape), activation='linear',
                    kernel_initializer=init)(ts_xs)
        ts_out = tf.keras.layers.Reshape(y_shape)(ts_ys)
        kmodel = tf.keras.Model(ts_in, ts_out)
        return kmodel



class KerasMLP(KerasFuncApp):
    """ Basic MLP using tf.keras.layers """

    def __init__(self, x_shape, y_shape, name='robust_k_mlp', units=(),
                 activation='tanh', **kwargs):
        self.units, self.activation = units, activation
        super().__init__(x_shape, y_shape, name=name, **kwargs)

    def _build_kmodel(self, x_shape, y_shape):
        return _keras_mlp(x_shape, y_shape, units=self.units, activation=self.activation)

class RobustKerasMLP(RobustKerasFuncApp):
    """ Basic MLP using tf.keras.layers """

    def __init__(self, x_shape, y_shape, name='robust_k_mlp', units=(),
                 activation='tanh', **kwargs):
        self.units, self.activation = units, activation
        super().__init__(x_shape, y_shape, name=name, **kwargs)

    def _build_kmodel(self, x_shape, y_shape):
        return _keras_mlp(x_shape, y_shape, units=self.units, activation=self.activation)


class tfMLP(tfFuncApp):
    """ Basic MLP using basic tensorflow operations. """

    def __init__(self, x_shape, y_shape, name='robust_k_mlp', units=(),
                 activation='tanh', **kwargs):
        assert(activation in ['tanh', 'relu', 'sigmoid'])
        self.units, self.activation = units, activation
        self.activation_fn = getattr(tf.nn, activation)
        self.ts_w, self.ts_b = [], []
        dims = [np.prod(x_shape)]+list(units)+[np.prod(y_shape)]
        for i in range(1,len(dims)):
            dim_in = dims[i-1]
            dim_out = dims[i]
            std = np.sqrt(2.0/(dim_in+dim_out))
            self.ts_w.append(tf.Variable(np.random.normal(scale=std,
                             size=(dim_in, dim_out)),dtype=tf_float))
            self.ts_b.append(tf.Variable(np.random.normal(scale=std,
                             size=(dim_out,)),dtype=tf_float))

        super().__init__(x_shape, y_shape, **kwargs)

    @property
    def ts_variables(self):
        return self.ts_w+self.ts_b

    #@tf.function
    def ts_predict(self, ts_xs):
        # the last layer is linear
        ts_xs = tf.reshape(ts_xs,(-1,np.prod(self.x_shape)))
        for i in range(len(self.ts_w)-1):
            ts_w = self.ts_w[i]
            ts_b = self.ts_b[i]
            ts_xs = self.activation_fn(tf.tensordot(ts_xs, ts_w, axes=1)+ts_b)
        ts_w = self.ts_w[-1]
        ts_b = self.ts_b[-1]
        ts_ys = tf.tensordot(ts_xs, ts_w, axes=1)+ts_b
        ts_ys = tf.reshape(ts_ys, np.array([-1]+list(self.y_shape)))
        return ts_ys

class tfRobustMLP(tfRobustFuncApp, tfMLP):
    pass


class tfConstant(tfFuncApp):
    """ A constant function. """
    def __init__(self, x_shape, y_shape, name='tf_constant', **kwargs):
        assert x_shape==(0,)  # make sure the user know what's going on
        self._ts_val = tf.Variable(tf.random.normal(y_shape))
        super().__init__((0,), y_shape, name=name, **kwargs)

    @property
    def ts_variables(self):
        return [self._ts_val]

    def ts_predict(self, ts_xs, **kwargs):
        ts_ones = tf.ones([ts_xs.shape[0]]+[1]*len(self.y_shape))
        return ts_ones * self._ts_val


class tfIdentity(tfFuncApp):
    """ Just an identity map """
    def __init__(self, x_shape, y_shape=None, name='tf_identity', **kwargs):
        assert y_shape is None or x_shape==y_shape
        super().__init__(x_shape, x_shape, name=name, **kwargs)

    @property
    def ts_variables(self):
        return []

    def ts_predict(self, ts_xs, **kwargs):
        return identity(ts_xs)  # prevent wrong chain-rule


class tfGradFun(tfFuncApp):
    """ Gradient of some scalar base function. """
    def __init__(self, x_shape, y_shape, name='tf_grad', base_fun=None, **kwargs):
        assert isinstance(base_fun, tfFuncApp)
        assert sum(base_fun.y_shape)==1
        assert base_fun.x_shape==x_shape
        self.base_fun = base_fun
        super().__init__(x_shape, x_shape, name=name, **kwargs)

    @property
    def ts_variables(self):
        return self.base_fun.ts_variables

    def ts_predict(self, ts_xs, **kwargs):
        return self.base_fun.ts_grad(ts_xs, **kwargs)
