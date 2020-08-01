"""
Minibatch functions.
"""
import numpy as np
import math


def generate_batches(arrays, n_batches=None, batch_size=None, shuffle=True, include_final_partial_batch=True):
    assert (n_batches is None) != (batch_size is None), 'Provide num_batches or batch_size, but not both'
    arrays = tuple(map(np.asarray, arrays))
    n = arrays[0].shape[0]
    assert all(array.shape[0] == n for array in arrays[1:])  # make sure the first dimension are the same.
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)  # modify in place
    sections = np.arange(0, n, batch_size)[1:] if n_batches is None else n_batches
    for batch_idx in np.array_split(idx, sections):
        if include_final_partial_batch or len(batch_idx) == batch_size:
            yield tuple(array[batch_idx] for array in arrays)


def apply_in_batches(func, x, batch_size, y_element_shape=[]):
    n = len(x)
    y_shape = [n] + list(y_element_shape)
    y = np.zeros(shape=y_shape, dtype='float32')
    n_batches = math.ceil(n / batch_size)
    for i_batch in range(n_batches):
        idx = range(i_batch * batch_size, min((i_batch + 1) * batch_size, n))
        y[idx] = func(x[idx])
    # XXX reshape???
    return y
