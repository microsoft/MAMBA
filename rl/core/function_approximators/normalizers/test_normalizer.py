# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest
import os
import copy
import numpy as np
import tensorflow as tf
from rl.core.function_approximators import normalizers as Nor
import functools

def assert_array(a,b):
    assert np.all(np.isclose(a-b,0.0, atol=1e-5))

def test_basics(cls):
    shape = (1,3,4)
    nor = cls(shape)
    xs = np.random.random([10]+list(shape))
    nxs = nor.normalize(xs)
    assert np.all(np.isclose(xs,nxs))  # should be identity before any call of update

    # single instance
    x = np.random.random(shape)
    nx = nor(x)
    assert np.all(np.isclose(x,nx))

    # copy
    nor2 = copy.deepcopy(nor)
    nor.update(xs)
    assert nor._initialized and not nor2._initialized

    # save and load
    import tempfile
    with tempfile.TemporaryDirectory() as path:
        nor.save(path)
        nor2.restore(path)
        assert_array(nor(x), nor2(x))

    # reset
    nor.reset()
    assert nor._initialized is False

class Tests(unittest.TestCase):

    def test_normalizers(self):
        test_basics(Nor.Normalizer)
        test_basics(Nor.NormalizerStd)
        test_basics(Nor.NormalizerMax)
        test_basics(Nor.tfNormalizerStd)
        test_basics(Nor.tfNormalizerMax)

    def test_normalizer_std(self):
        shape = (1,2,3)
        nor = Nor.NormalizerStd(shape)
        xs = np.random.random([10]+list(shape))
        nor.update(xs)
        assert np.all(np.isclose(nor._bias-np.mean(xs,axis=0),0.0))
        assert np.all(np.isclose(nor._scale-np.std(xs,axis=0),0.0))
        assert nor._initialized  is True

        xs2 = np.random.random([10]+list(shape))
        xs = np.concatenate((xs,xs2))
        nor.update(xs2)
        assert np.all(np.isclose(nor._bias-np.mean(xs,axis=0),0.0))
        assert np.all(np.isclose(nor._scale-np.std(xs,axis=0),0.0))


    def test_tf_normalizers(self):

        def _test(cls, tf_cls):
            shape = (1,2,3)
            nor = cls(shape)
            tf_nor = tf_cls(shape)
            for _ in range(1000):
                xs = np.random.random([10]+list(shape))
                nor.update(xs)
                tf_nor.update(xs)
                assert nor._initialized is True
                assert tf_nor._initialized is True
                xs = np.random.random([10]+list(shape))
                nxs1 = nor.normalize(xs)
                nxs2 = tf_nor.ts_normalize(tf.constant(xs,dtype=tf.float32)).numpy()
                assert_array(nxs1, nxs2)

            # save and load
            tf_nor2 = copy.deepcopy(tf_nor)
            import tempfile
            xs = np.random.random([10]+list(shape))
            tf_nor.update(xs)
            with tempfile.TemporaryDirectory() as path:
                tf_nor.save(path)
                tf_nor2.restore(path)
                assert_array(tf_nor.predict(xs), tf_nor2.predict(xs))
                nxs1 = tf_nor.ts_normalize(tf.constant(xs,dtype=tf.float32)).numpy()
                nxs2 = tf_nor2.ts_normalize(tf.constant(xs,dtype=tf.float32)).numpy()
                assert_array(nxs1, nxs2)


        _test(Nor.NormalizerStd, Nor.tfNormalizerStd)
        _test(Nor.NormalizerMax, Nor.tfNormalizerMax)

        NormalizerClip = functools.partial(Nor.NormalizerClip, clip_thre=3.0)
        tfNormalizerClip = functools.partial(Nor.tfNormalizerClip, clip_thre=3.0)

        _test(NormalizerClip, tfNormalizerClip)


    def test_normalizer_clip(self):
        nor =Nor.NormalizerClip(shape=(1,), clip_thre=1)
        assert np.isclose(nor(np.array(10000)),1)





if __name__ == '__main__':

    unittest.main()
