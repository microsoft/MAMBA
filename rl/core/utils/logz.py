# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Licensed under the MIT License.

"""

Some simple logging functionality, inspired by rllab's logging.
Assumes that each diagnostic gets logged each iteration

Call logz.configure_output_dir() to start logging to a
tab-separated-values file (some_folder_name/log.txt)

To load the learning curves, you can do, for example

A = np.genfromtxt('/tmp/expt_1468984536/log.txt',delimiter='\t',dtype=None, names=True)
A['EpRewMean']

"""
import json
import os.path as osp
import time
import os
import pickle
import tensorflow as tf

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


class LOG:
    output_dir = None
    output_file = None
    first_row = True
    log_headers = []
    log_current_row = {}


def configure_output_dir(d=None):
    """
    Set output directory to d, or to /tmp/somerandomnumber if d is None
    """
    LOG.output_dir = d or "/tmp/experiments/%i" % int(time.time())
    # assert not osp.exists(
    #     LOG.output_dir), "Log dir %s already exists! Delete it first or use a different dir" % LOG.output_dir
    if not os.path.exists(LOG.output_dir):
        os.makedirs(LOG.output_dir)
    LOG.output_file = open(osp.join(LOG.output_dir, "log.txt"), 'w')
    # atexit.register(LOG.output_file.close)
    print(colorize("Logging data to %s" % LOG.output_file.name, 'green', bold=True))


def log_tabular(key, val):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    """
    def append_new_key(key):
        f_name = osp.join(LOG.output_dir, "log.txt")
        temp_f_name = osp.join(LOG.output_dir, "temp_log.txt")
        LOG.output_file.close()
        f = open(f_name, 'r')
        temp_f = open(temp_f_name, 'w')
        line = f.readline()
        temp_f.write('{}\t{}\n'.format(line.rstrip(), key))
        line = f.readline()
        while line:
            temp_f.write('{}\t{}\n'.format(line.rstrip(), "0"))
            line = f.readline()
        f.close()
        temp_f.close()
        os.rename(temp_f_name, f_name)
        LOG.output_file = open(f_name, 'a')

    if LOG.first_row:
        LOG.log_headers.append(key)
    else:
        if key not in LOG.log_headers:
            print(colorize('Trying to introduce a new key {} that is not included in the first iteration'.format(key),
                           'green'))
            # Need to accommodate for the new key.
            LOG.log_headers.append(key)
            if LOG.output_dir:
                append_new_key(key)
    assert key not in LOG.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()" % key
    LOG.log_current_row[key] = val


def save_params(params):
    with open(osp.join(LOG.output_dir, "params.json"), 'w') as out:
        out.write(json.dumps(params, separators=(',\n', '\t:\t'), sort_keys=True))


def pickle_tf_vars():
    """
    Saves tensorflow variables
    Requires them to be initialized first, also a default session must exist
    """
    _dict = {v.name: v.eval() for v in tf.global_variables()}
    with open(osp.join(LOG.output_dir, "vars.pkl"), 'wb') as f:
        pickle.dump(_dict, f)


def dump_tabular():
    """
    Write all of the diagnostics from the current iteration
    """
    vals = []
    key_lens = [len(key) for key in LOG.log_headers]
    max_key_len = max(15, max(key_lens))
    keystr = '%' + '%d' % max_key_len
    fmt = "| " + keystr + "s | %15s |"
    n_slashes = 22 + max_key_len
    print("-" * n_slashes)
    for key in LOG.log_headers:
        val = LOG.log_current_row.get(key, "")
        if hasattr(val, "__float__"):
            valstr = "%8.4g" % val
        else:
            valstr = val
        print(fmt % (key, valstr))
        vals.append(val)
    print("-" * n_slashes)
    if LOG.output_file is not None:
        if LOG.first_row:
            LOG.output_file.write("\t".join(LOG.log_headers))
            LOG.output_file.write("\n")
        LOG.output_file.write("\t".join(map(str, vals)))
        LOG.output_file.write("\n")
        LOG.output_file.flush()
    LOG.log_current_row.clear()
    LOG.first_row = False
