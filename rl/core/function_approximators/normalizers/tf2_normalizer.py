# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import numpy as np
import copy
from abc import ABC, abstractmethod

from rl.core.function_approximators.normalizers import normalizer as pynor
from rl.core.utils.tf2_utils import tf_float


def make_tf_normalizer(cls):
    """ A decorator for adding a tf operator equivalent of Normalizer.predict

        It reuses all the functionalties of the original Normalizer and
        additional tf.Variables for defining the tf operator.
    """
    assert issubclass(cls, pynor.Normalizer)

    class decorated_cls(cls):

        def __init__(self, shape, *args, **kwargs):
            super().__init__(shape, *args, **kwargs)
            # add additional tf.Variables
            self._ts_bias = tf.Variable(self._bias, dtype=tf_float, trainable=False)
            self._ts_scale = tf.Variable(self._scale, dtype=tf_float, trainable=False)
            self._ts_unscale = tf.Variable(self._unscale, dtype=tf.bool, trainable=False)
            self._ts_unbias = tf.Variable(self._unbias, dtype=tf.bool, trainable=False)
            self._ts_initialized = tf.Variable(self._initialized, dtype=tf.bool, trainable=False)
            self._ts_clip = tf.Variable(self._thre is not None, dtype=tf.bool, trainable=False)
            if self._thre is not None:
                self._ts_thre = tf.Variable(self._thre, dtype=tf_float, trainable=False)
            else:  # just a dummy
                self._ts_thre = tf.Variable(0.0, dtype=tf_float, trainable=False)
            # make sure they are sync
            self._update_tf_vars()

        def ts_predict(self, ts_x):
            # mimic the same behavior as `predict`
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

        def ts_normalize(self, ts_x):  # alias
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
            self._ts_bias.assign(self._bias)
            self._ts_scale.assign(self._scale)
            self._ts_unbias.assign(self._unbias)
            self._ts_unscale.assign(self._unscale)
            self._ts_initialized.assign(self._initialized)
            self._ts_clip.assign(self._thre is not None)
            if self._thre is not None:
                self._ts_thre.assign(self._thre)

    # make them look the same as intended
    decorated_cls.__name__ = cls.__name__
    decorated_cls.__qualname__ = cls.__qualname__
    return decorated_cls


@make_tf_normalizer
class tfNormalizerClip(pynor.NormalizerClip):
    pass

@make_tf_normalizer
class tfNormalizerStd(pynor.NormalizerStd):
    pass

@make_tf_normalizer
class tfNormalizerMax(pynor.NormalizerMax):
    pass


