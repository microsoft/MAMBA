# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import unittest

from collections import namedtuple
import numpy as np
from rl.core.datasets.dataset import Dataset, data_namedtuple



def assert_array(a,b):
    assert np.all(np.isclose(a-b, 1e-8))


class Tests(unittest.TestCase):

    def test_basics(self):


        dataset = Dataset(max_n_batches=0, max_n_samples=0)
        dataset.append(list(range(5)))
        print(dataset[None])
        dataset.append(list(range(5,10)))
        print(dataset[None])


        dataset = Dataset(max_n_batches=2, max_n_samples=3)
        dataset.append(list(range(3)))
        assert_array(dataset[None], np.arange(3))
        dataset.append(list(range(3,6)))
        assert_array(dataset[None], np.arange(3,6))
        dataset.append(list(range(6,8)))
        assert_array(dataset[None], np.arange(3,8))


        MyData = data_namedtuple('MyData', ['x','y','z'])
        data = MyData(y=np.arange(10), x=np.random.random((10,1)), z= np.random.normal(size=(10,13)))
        dataset = Dataset([data])
        data = MyData(y=np.arange(10), x=np.random.random((10,1)), z= np.random.normal(size=(10,13)))
        dataset.append(data)
        data = MyData(y=np.arange(10), x=np.random.random((10,1)), z= np.random.normal(size=(10,13)))
        dataset.append(data)
        dataset['x']
        dataset['y']
        dataset['z']

if __name__=='__main__':
    unittest.main()
