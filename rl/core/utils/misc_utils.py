# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Licensed under the MIT License.

import sys
import time
import collections
import copy
import numpy as np
import tensorflow as tf
from contextlib import contextmanager


def set_randomseed(seed):
    if tf.__version__[0]=='2':
        tf.random.set_seed(seed)
    else:
        tf.set_random_seed(seed)  # graph-level seed
    np.random.seed(seed)

def profile():
    import line_profiler
    import atexit
    pf = line_profiler.LineProfiler()
    atexit.register(pf.print_stats)
    return pf

def safe_assign(obj, *args):
    assert any([isinstance(obj, cls) for cls in args])
    return obj

# To be compatible with python3.4.
def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z

def deepcopy_from_list(old, new, attrs, excludes=()):
    [setattr(old, attr, copy.deepcopy(getattr(new, attr))) \
            for attr in attrs if not attr in excludes]

def copy_from_list(old, new, attrs, excludes=()):
    [setattr(old, attr, copy.copy(getattr(new, attr))) \
            for attr in attrs if not attr in excludes]


color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)


def cprint(string, color='red', bold=False, highlight=False):
    """Print with color."""
    print(colorize(string, color, bold, highlight))


@contextmanager
def timed(msg):
    print(colorize(msg, color='magenta'), end='', flush=True)
    tstart = time.perf_counter()
    yield
    print(colorize(" in %.3f seconds" % (time.perf_counter() - tstart), color='magenta'))


def dict_update(d, u):
    """Update dict d based on u recursively."""
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def check_required_params(params, required_params):
    for param in required_params:
        assert param in params, '{} is not included in params'.format(param)


def flatten(vs):
    if type(vs) is list:
        return np.concatenate([np.reshape(v, [-1]) for v in vs], axis=0)
    else:
        return vs


def unflatten(v, template=None, shapes=None):
    """Shape a flat v in to a list of array with shapes as in template, or with shapes specified by shapes.
    Args:
        v: a np array.
        template: a list of arrays.
        shapes: a list of tuples.
    """
    assert (template is None) != (shapes is None)  # XOR
    start = 0
    vs = []
    if template:
        for w in template:
            vs.append(np.reshape(v[start:start+w.size], w.shape))
            start += w.size
    else:
        for shape in shapes:
            size = int(np.prod(shape))
            vs.append(np.reshape(v[start:start+size], shape))
            start += size
    return vs


def zipsame(*seqs):
    length = len(seqs[0])
    assert all(len(seq) == length for seq in seqs[1:])
    return zip(*seqs)
