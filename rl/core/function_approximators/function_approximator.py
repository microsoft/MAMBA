# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from abc import abstractmethod
from functools import wraps
import os, pickle, copy
from collections import Iterable
from rl.core.oracles.oracle import Oracle



def online_compatible(f):
    def to_batch(x):  # add an extra dimension
        return  [xx[None,:] for xx in x] if isinstance(x, list) or isinstance(x, tuple)\
                else x[None,:]
    @wraps(f)
    def decorated_f(self, x, *args, **kwargs):
        single = getattr(x,'shape', None)==self.x_shape
        if single:  # single instance
            x = to_batch(x)
            args =[to_batch(a) for a in args]
            y = f(self, x, *args, **kwargs)
            y = [yy[0] for yy in y] if isinstance(y, list) or isinstance(y, tuple)\
                else y[0]  # remove the extra dimension
        else:
            y = f(self, x, *args, **kwargs)
        return y
    return decorated_f

# TODO
def predict_in_batches(fun):
    """ for wrapping a predit method of FunctionApproximator objects """
    @wraps(fun)
    def wrapper(self, x):
        return minibatch_utils.apply_in_batches(lambda _x: fun(self, _x),
                                                x, self._batch_size_for_prediction, [self.y_dim])
    return wrapper


class FunctionApproximator(Oracle):
    """ An abstract interface of function approximators.

        Generally a function approximator has
            1) "variables" that are amenable to gradient-based updates,
            2) "parameters" that works as the hyper-parameters.

        This is realized by adding `variables` property to `Oracle`. In
        addition, here we require function calls to be compatible with both
        single-instance and batch queries.

        We also provide basic `assign`, `save`, `restore` functions, based on
        deepcopy and pickle, which should work for nominal python objects. But
        they might need be overloaded when more complex objects are used (e.g.,
        tf.keras.Model) as attributes.

        The user needs to implement the following
            `predict`, `variables` (getter and setter), and `update` (optional)

        In addition, the class should be copy.deepcopy compatible.
    """
    def __init__(self, x_shape, y_shape, name='func_app', **kwargs):
        self.name = name
        self.x_shape = x_shape  # a nd.array or a list of nd.arrays
        self.y_shape = y_shape  # a nd.array or a list of nd.arrays

    def fun(self, x, **kwargs):  # alias
        return self(x, **kwargs)

    # Users can choose to implement `grad`.

    def update(self, *args, **kwargs):
        """ Perform update the parameters.

            This can include updating internal normalizers, etc.
            Return a report, if any.
        """
        # callable, but does nothing by default

    # New methods of FunctionApproximator
    @abstractmethod
    def predict(self, xs, **kwargs):
        """ Predict the values on batches of xs. """

    @online_compatible
    def __call__(self, xs, **kwargs):
        return self.predict(xs, **kwargs)

    @property
    @abstractmethod
    def variable(self):
        """ Return the variable as a np.ndarray. """

    @variable.setter
    @abstractmethod
    def variable(self, val):
        """ Set the variable as val, which is a np.ndarray in the same format as self.variable. """

    # utilities
    def save(self, path, name=None):
        """ Save the instance in path. """
        if not os.path.exists(path):
            os.makedirs(path)
        name = name or self.name
        path = os.path.join(path, name)
        with open(path, 'wb') as pickle_file:
            pickle.dump(self, pickle_file)

    def restore(self, path, name=None):
        """ restore the saved instance in path. """
        name = name or self.name
        path = os.path.join(path, name)
        with open(path, 'rb') as pickle_file:
            saved = pickle.load(pickle_file)
        self.__dict__.update(saved.__dict__)
