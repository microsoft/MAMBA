# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np
import tensorflow as tf
from math import ceil
from rl.core import function_approximators as fa
from rl.core.function_approximators.supervised_learners import SupervisedLearner

def robust_keras_supervised_learner(cls):
    """ A decorator for creating basic supervised learners from RobustKerasFuncApp. """
    assert issubclass(cls, fa.KerasFuncApp)
    class decorated_cls(cls, SupervisedLearner):

        def __init__(self, x_shape, y_shape, name='k_robust_super_learner',
                     lr=0.001, loss='mse', metrics=('mae','mse'), **kwargs):

            super().__init__(x_shape, y_shape, name=name, **kwargs)
            self._lr =lr
            self._loss = loss
            self._metrics = metrics
            self._output_bias = 0.
            self._output_scale = 1.0
            self.kmodel.compile(optimizer=tf.keras.optimizers.Adam(self._lr),
                                loss=self._loss, metrics=list(self._metrics))

        def update_funcapp(self, clip_y=False,
                            batch_size=128, n_steps=100,
                            normalize_output=None, # can be None, 'white', or 'rms'
                            epochs=None, **kwargs):  # for keras.Model.fit
            """ `clip_y`: whether to clip the target """


            if isinstance(self, fa.RobustKerasFuncApp):
                xs = self._x_nor(self._dataset['xs'])
                ys = self._dataset['ys']
                if clip_y:
                    ys = self._y_nor(ys)
            else:
                xs, ys = self._dataset['xs'], self._dataset['ys']

            # shuffle
            ind = np.random.permutation(len(xs))
            xs = xs[ind,:]
            ys = ys[ind,:]

            # whiten the output for supervised learning
            assert normalize_output in [None, 'white', 'rms']
            if normalize_output=='white':
                self._output_bias = np.mean(ys, axis=0)
                self._output_scale = np.maximum(1.0, np.std(ys,axis=0))
            elif normalize_output=='rms':
                self._output_bias = 0.0
                self._output_scale = np.maximum(1.0, np.sqrt(np.mean(ys**2,axis=0)))

            ys = (ys-self._output_bias)/self._output_scale

            if epochs is None:
                epochs = ceil(n_steps*batch_size/len(ys))
            steps_per_epoch = n_steps if epochs==1 else None

            results = self.kmodel.fit(xs, ys, sample_weight=self._dataset['ws'], verbose=0,
                                   batch_size=batch_size, epochs=epochs,
                                   steps_per_epoch=steps_per_epoch,
                                   **kwargs)
            tf.keras.backend.clear_session()
            return results

        def __setstate__(self, d):
            super().__setstate__(d)
            self.kmodel.compile(optimizer=tf.keras.optimizers.Adam(self._lr),
                                loss=self._loss, metrics=list(self._metrics))

        def predict(self, *args, **kwargs):
            # prevent memory-leak
            return self.k_predict(*args, **kwargs)

        def k_predict(self, xs, **kwargs):
            return super().k_predict(xs, **kwargs)*self._output_scale+self._output_bias

        def ts_predict(self, ts_xs, **kwargs):
            return super().ts_predict(ts_xs, **kwargs)*self._output_scale+self._output_bias

    # to make them look the same as intended
    decorated_cls.__name__ = cls.__name__
    decorated_cls.__qualname__ = cls.__qualname__
    return decorated_cls


@robust_keras_supervised_learner
class SuperRobustKerasFuncApp(fa.RobustKerasFuncApp):
    pass

@robust_keras_supervised_learner
class SuperRobustKerasMLP(fa.RobustKerasMLP):
    pass

@robust_keras_supervised_learner
class SuperKerasMLP(fa.KerasMLP):
    pass
