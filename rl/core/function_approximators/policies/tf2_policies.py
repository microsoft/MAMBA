# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import tensorflow as tf
import numpy as np
from rl.core.function_approximators.policies import Policy
from rl.core.function_approximators.function_approximator import online_compatible
from rl.core.function_approximators.tf2_function_approximators import tfFuncApp, RobustKerasMLP, KerasFuncApp, RobustKerasFuncApp, tfRobustMLP, tfConstant
from rl.core.utils.misc_utils import zipsame
from rl.core.utils.tf2_utils import tf_float, array_to_ts, ts_to_array
from rl.core.utils.misc_utils import flatten, unflatten


class tfPolicy(tfFuncApp, Policy):
    """ A stochastic version of tfFuncApp.

        The user need to define `ts_predict`, `ts_variables`, and optionally
        `ts_logp`, `ts_kl` and `ts_fvp`.

        By default, `ts_logp` returns log(delta), i.e. the function is assumed
        to be deterministic. Therefore, it can be used a wrapper of subclass of
        `tfFuncApp`. For example, for a subclass `A`, one can define

            class B(tfPolicy, A):
                pass

        which creates a deterministic tfPolicy.
    """
    def __init__(self, x_shape, y_shape, name='tf_policy', **kwargs):
        super().__init__(x_shape, y_shape, name=name, **kwargs)

    # `predict` has been defined by tfFuncApp
    # Users may choose to implement `exp_fun`, `exp_grad`, `noise`, `derandomize`.

    @online_compatible
    def logp(self, xs, ys, **kwargs):  # override
        return self.ts_logp(array_to_ts(xs), array_to_ts(ys), **kwargs).numpy()

    def logp_grad(self, xs, ys, fs, **kwargs):
        ts_grad = self.ts_logp_grad(array_to_ts(xs), array_to_ts(ys),
                                 array_to_ts(fs), **kwargs)
        return flatten([v.numpy() for v in ts_grad])

    def kl(self, other, xs, reversesd=False, **kwargs):
        """ Return the KL divergence for each data point in the batch xs. """
        return ts_to_array(self.ts_kl(other, array_to_ts(xs), reversesd=reversesd))

    def fvp(self, xs, g, **kwargs):
        """ Return the product between a vector g (in the same formast as
        self.variable) and the Fisher information defined by the average
        over xs. """
        gs = unflatten(g, shapes=self.var_shapes)
        ts_fvp = self.ts_fvp(array_to_ts(xs), array_to_ts(gs), **kwargs)
        return flatten([v.numpy() for v in ts_fvp])

    # New methods of tfPolicy
    def ts_predict(self, ts_xs, stochastic=True, **kwargs):
        """ Define the tf operators for predict """
        return super().ts_predict(ts_xs, stochastic=stochastic, **kwargs)

    def ts_logp(self, ts_xs, ts_ys):
        """ Define the tf operators for logp """
        ts_p = tf.cast(tf.equal(self.ts_predict(ts_xs), ts_ys), dtype=tf_float)
        return tf.math.log(ts_p)  # indicator

    def ts_logp_grad(self, ts_xs, ts_ys, ts_fs):
        """ Sum over samples. """
        with tf.GradientTape() as gt:
            gt.watch(self.ts_variables)
            ts_logp = self.ts_logp(ts_xs, ts_ys)
            ts_fun = tf.reduce_sum(ts_logp*ts_fs)
        return gt.gradient(ts_fun, self.ts_variables)

    # Some useful functions
    def ts_kl(self, other, xs, reversesd=False, **kwargs):
        """ Computes KL(self||other), where other is another object of the
            same policy class. If reversed is True, return KL(other||self).
        """
        raise NotImplementedError

    def ts_fvp(self, ts_xs, ts_gs, **kwargs):
        """ Computes F(self.pi)*g, where F is the Fisher information matrix and
        g is a np.ndarray in the same shape as self.variable """
        raise NotImplementedError


class _RobustKerasPolicy(tfPolicy, RobustKerasFuncApp):
    pass  # for debugging


class _RobustKerasMLPPolicy(tfPolicy, RobustKerasMLP):
    pass  # for debugging


LOG_TWO_PI = tf.constant(np.log(2*np.pi), dtype=tf_float)
def gaussian_logp(xs, ms, lstds):
     # log probability of Gaussian with diagonal variance over batches xs
    axis= tf.range(1,tf.rank(xs))
    qs = tf.reduce_sum(-0.5*tf.square(xs-ms)/tf.exp(2.*lstds), axis=axis)
    logs = tf.reduce_sum(-lstds -0.5*LOG_TWO_PI,axis=axis)
    return qs + logs

def gaussian_kl(ms_1, lstds_1, ms_2, lstds_2):
    # KL(p1||p2)  support batches
    axis= tf.range(1,tf.rank(ms_1))
    vars_1, vars_2 = tf.exp(lstds_1*2.), tf.exp(lstds_2*2.)
    kls = lstds_2 - lstds_1 - 0.5
    kls += (vars_1 + tf.square(ms_1-ms_2)) / (2.0*vars_2)
    kls = tf.reduce_sum(kls, axis=axis)
    return kls

def gaussian_exp(ms, vs, As, bs, cs, canonical, diagonal_A, diagonal_vs):
    # TODO
    raise NotImplementedError


class tfGaussianPolicy(tfPolicy):
    """ A wrapper class for augmenting tfFuncApp with diagonal Gaussian noises.

        For example, for a subclass `A` of tfFuncApp, one can define

            class B(tfGaussianPolicy, A):
                pass

        which creates a Gaussian policy with the mean specified by `A`.
    """
    def __init__(self, x_shape, y_shape, name='tf_gaussian_policy',
                 init_lstd=None, min_std=1e-12,  # new attribues
                 **kwargs):
        """ The user needs to provide init_lstd. """
        assert init_lstd is not None
        init_lstd = np.broadcast_to(init_lstd, y_shape)
        self._ts_lstd = tf.Variable(array_to_ts(init_lstd), dtype=tf_float)
        self._ts_min_lstd = tf.constant(np.log(min_std), dtype=tf_float)
        super().__init__(x_shape, y_shape, name=name, **kwargs)
        self._mean_var_shapes = None

    # Some convenient properties
    @online_compatible
    def mean(self, xs):
        return self(xs, stochastic=False)

    def ts_mean(self, xs):
        return self.ts_predict(xs, stochastic=False)

    @property
    def ts_mean_variables(self):
        return super().ts_variables

    @property
    def mean_variable(self):
        return flatten(ts_to_array(super().ts_variables))

    @mean_variable.setter
    def mean_variable(self, val):
        vals = unflatten(val, shapes=self.mean_var_shapes)
        [var.assign(val) for var, val in zipsame(self.ts_mean_variables, vals)]

    @property
    def mean_var_shapes(self):
        if self._mean_var_shapes is None:
            self._mean_var_shapes = [var.shape.as_list() for var in self.ts_mean_variables]
        return self._mean_var_shapes


    @property
    def lstd(self):
        return self.ts_lstd.numpy()

    @lstd.setter
    def lstd(self, val):
        return self._ts_lstd.assign(val)

    @property
    def ts_lstd(self):
        return tf.maximum(self._ts_lstd, self._ts_min_lstd)

    # Methods of Policy
    @online_compatible
    def predict_w_noise(self, xs, stochastic=True, **kwargs):
        ts_ys, ts_ms, _ = self.ts_predict_all(array_to_ts(xs),
                                              stochastic=stochastic, **kwargs)
        ys, ms = ts_to_array(ts_ys), ts_to_array(ts_ms)
        ns = ys - ms
        return ys, ms

    @online_compatible
    def noise(self, xs, ys):
        return ys - self.mean(xs)

    @online_compatible
    def derandomize(self, xs, noises):
        return self.mean(xs) + noises

    @online_compatible
    def exp_fun(self, xs, As, bs, cs, canonical=True, diagonal_A=True):
        """
            If canonical is True, computes
                E[ 0.5 y'*A*y + b'y + c]
            Else computes
                E[ 0.5 (y-m)'*A*(y-m) + b'*y + c]
        """
        return self.ts_exp_fun(array_to_ts(xs), array_to_ts(As),
                               array_to_ts(bs), array_to_ts(cs),
                               canonical, diagonal_A).numpy()

    def exp_grad(self, xs, As, bs, cs, canonical=True, diagonal_A=True):
        """ See exp_fun. """
        ts_grad = self.ts_exp_grad(array_to_ts(xs), array_to_ts(As),
                                   array_to_ts(bs), array_to_ts(cs),
                                   canonical, diagonal_A)
        return flatten([v.numpy() for v in ts_grad])

    # Methods of tfPolicy/tfFuncApp
    @property
    def ts_variables(self):
        return super().ts_variables + [self._ts_lstd]

    # ts_predict, ts_logp, ts_fvp, ts_kl
    def ts_predict(self, ts_xs, stochastic=True, **kwargs):
        """ Define the tf operators for predict """
        ts_ys, _, _ = self.ts_predict_all(ts_xs, stochastic=stochastic, **kwargs)
        return ts_ys

    def ts_predict_all(self, ts_xs, stochastic=True, **kwargs):
        """ Define the tf operators for predict """
        ts_ms = super().ts_predict(ts_xs, **kwargs)
        shape = [ts_xs.shape[0]]+list(self.y_shape)
        if stochastic:
            ts_noises = tf.exp(self.ts_lstd) * tf.random.normal(shape)
            ts_ys = ts_ms +  ts_noises  # more stable
            return ts_ys, ts_ms, ts_noises
        else:
            ts_noises = tf.zeros(shape)
            return ts_ms, ts_ms, ts_noises

    # `ts_noise` and `ts_derandomize` should only be used within tf operations
    def ts_noise(self, ts_xs, ts_ys):
        return ts_ys - self.ts_mean(ts_xs)

    def ts_derandomize(self, ts_xs, ts_noises):
        return self.ts_mean(ts_xs) + ts_noises

    def ts_logp(self, ts_xs, ts_ys):  # overwrite
        ts_ms = self.ts_mean(ts_xs)
        ts_lstds = tf.broadcast_to(self.ts_lstd, ts_ms.shape)
        return gaussian_logp(ts_ys, ts_ms, ts_lstds)

    def ts_exp_fun(self, ts_xs, ts_As, ts_bs, ts_cs, canonical=True, diagonal_A=True):
        ts_ms = self.ts_mean(ts_xs)
        ts_vs= tf.exp(2.0*self.ts_lstd)
        ts_vs = tf.broadcast_to(ts_vs, ts_ms.shape)
        ts_fun = gaussian_exp(ts_ms, ts_vs, ts_As, ts_bs, ts_cs,
                              canonical=canonical, diagonal_A=diagonal_A)
        return ts_fun

    def ts_exp_grad(self, ts_xs, ts_As, ts_bs, ts_cs, canonical=True, diagonal_A=True):
        with tf.GradientTape() as gt:
            gt.watch(self.ts_variables)
            ts_fun = self.ts_exp_fun(ts_xs, ts_As, ts_bs, ts_cs, canonical, diagonal_A)
        return gt.gradient(ts_fun, self.ts_variables)

    def ts_kl(self, other, ts_xs, reversesd=False, p1_sg=False, p2_sg=False):
        """ Computes KL(self||other), where other is another object of the
            same policy class. If reversed is True, return KL(other||self).
        """
        def get_m_and_lstd(p, stop_gradient):
            ts_ms = p.ts_mean(ts_xs)
            ts_lstds = tf.broadcast_to(p.ts_lstd, ts_ms.shape)
            if stop_gradient:
                ts_ms, ts_lstds = tf.stop_gradient(ts_ms), tf.stop_gradient(ts_lstds)
            return ts_ms,  ts_lstds
        ts_ms_1, ts_lstds_1 = get_m_and_lstd(self,  p1_sg)
        ts_ms_2, ts_lstds_2 = get_m_and_lstd(other, p2_sg)
        if reversesd:
            return tf.reduce_mean(gaussian_kl(ts_ms_1, ts_lstds_1, ts_ms_2, ts_lstds_2))
        else:
            return tf.reduce_mean(gaussian_kl(ts_ms_2, ts_lstds_2, ts_ms_1, ts_lstds_1))

    def ts_fvp(self, ts_xs, ts_gs):
        """ Computes F(self.pi)*g, where F is the Fisher information matrix and
        g is a np.ndarray in the same shape as self.variable """
        with tf.GradientTape() as gt:
            gt.watch(self.ts_variables)
            with tf.GradientTape() as gt2:
                gt2.watch(self.ts_variables)  #  TODO add sample weight below??
                ts_kl = self.ts_kl(self, ts_xs, p1_sg=True)
            ts_kl_grads = gt2.gradient(ts_kl, self.ts_variables)
            ts_pd = tf.add_n([tf.reduce_sum(kg*v) for (kg, v) in zipsame(ts_kl_grads, ts_gs)])
        ts_fvp = gt.gradient(ts_pd, self.ts_variables)
        return ts_fvp


class RobustKerasMLPGassian(tfGaussianPolicy, RobustKerasMLP):

    def __init__(self, x_shape, y_shape, name='robust_k_MLP_gaussian_policy', **kwargs):
        """ The user needs to provide init_lstd and optionally min_std. """
        super().__init__(x_shape, y_shape, name=name, **kwargs)

class tfRobustMLPGaussian(tfGaussianPolicy, tfRobustMLP):
    def __init__(self, x_shape, y_shape, name='robust_MLP_gaussian_policy', **kwargs):
        """ The user needs to provide init_lstd and optionally min_std. """
        super().__init__(x_shape, y_shape, name=name, **kwargs)

class tfGaussian(tfGaussianPolicy, tfConstant):
    """ A Gaussian distribution with learnable mean and diagonal covariance. """

    def __init__(self, x_shape, y_shape, name='tfGaussian', **kwargs):
        """ The user needs to provide init_lstd and optionally min_std. """
        super().__init__(x_shape, y_shape, name=name, **kwargs)

