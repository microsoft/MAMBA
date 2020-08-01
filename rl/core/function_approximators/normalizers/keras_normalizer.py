# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import numpy as np
import copy
from abc import ABC, abstractmethod

from rl.core.function_approximators.normalizers.normalizer import Normalizer, NormalizerStd, NormalizerMax
from rl.core.utils.tf2_utils import tf_float

class ClipNormalizer(tf.keras.layers.Layer):
    """ A tf.keras.layers.Layer that mimics the behavior of Normalizer. """
    # NOTE custom layers can be deepcopy and pickle
    def __init__(self, shape, thre_shape):
        super().__init__()
        self._ts_bias = self.add_variable('bias', shape, dtype=tf_float, trainable=False)
        self._ts_scale = self.add_variable('scale', shape, dtype=tf_float, trainable=False)
        self._ts_unscale = self.add_variable('unscale', dtype=tf.bool, trainable=False)
        self._ts_unbias = self.add_variable('unbias', dtype=tf.bool, trainable=False)
        self._ts_initialized = self.add_variable('initialized', dtype=tf.bool, trainable=False)
        self._ts_clip = self.add_variable('clip', dtype=tf.bool, trainable=False)
        self._ts_thre = self.add_variable('thre', thre_shape, dtype=tf_float, trainable=False)

    def build(self, input_shape):
        pass

    def call(self, ts_x):
        if tf.logical_not(self._ts_initialized):
            return ts_x
        # do something
        if tf.logical_not(self._ts_clip):
            if tf.logical_not(self._ts_unbias):
                ts_x = ts_x - self._ts_bias
            if tf.logical_not(self._ts_unscale):
                ts_x = ts_x / self._ts_scale
        else:
            # need to first scale it before clipping
            ts_x = (ts_x - self._ts_bias) / self._ts_scale
            ts_x = tf.clip_by_value(ts_x, self._ts_thre[0], self._ts_thre[1])
            # check if we need to scale it back
            if self._ts_unscale:
                ts_x = ts_x * self._ts_scale
                if self._ts_unbias:
                    ts_x = ts_x + self._ts_bias
            else:
                if self._ts_unbias:
                    ts_x = ts_x + self._ts_bias / self._ts_scale
        return ts_x


def _tfNormalizerDecorator(cls):
    """ A decorator for adding a tf operator equivalent of Normalizer.predict

        It reuses all the functionalties of the original Normalizer and
        additional tf.Variables for defining the tf operator.
    """
    assert issubclass(cls, Normalizer)

    class decorated_cls(cls):

        def __init__(self, shape, *args, **kwargs):
            super().__init__(shape, *args, **kwargs)
            # add additional tf.Variables
            thre_shape = (1,1) if self._thre is None else self._thre.shape
            self.klayer = ClipNormalizer(shape, thre_shape)
            self._update_tf_vars()

        def ts_predict(self, ts_x):
            return self.klayer(ts_x)

        def ts_normalize(self, ts_x):
            return self.ts_predict(ts_x)

        # make sure the tf.Variables are synchronized
        def update(self, x):
            super().update(x)
            self._update_tf_vars()

        def reset(self):
            super().reset()
            self._update_tf_vars()

        def assign(self, other):
            super().assign(other)
            self._update_tf_vars()

        def _update_tf_vars(self):
            # synchronize the tf.Variables
            if self.klayer is not None:
                self.klayer._ts_bias.assign(self._bias)
                self.klayer._ts_scale.assign(self._scale)
                self.klayer._ts_unbias.assign(self._unbias)
                self.klayer._ts_unscale.assign(self._unscale)
                self.klayer._ts_initialized.assign(self._initialized)
                self.klayer._ts_clip.assign(self._thre is not None)
                if self.klayer._ts_clip:
                    self.klayer._ts_thre.assign(self._thre)

    # make them look the same as intended
    decorated_cls.__name__ = cls.__name__
    decorated_cls.__qualname__ = cls.__qualname__
    return decorated_cls


@_tfNormalizerDecorator
class tfNormalizerStd(NormalizerStd):
    pass

@_tfNormalizerDecorator
class tfNormalizerMax(NormalizerMax):
    pass
