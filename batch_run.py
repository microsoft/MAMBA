# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse, copy, importlib
import multiprocessing as mp
import itertools
import os
import json
from functools import partial

try: # Restrict TensorFlow to use limited memory
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
except:
    pass

# `rl` has to loaded after tesorflow has been configured.
from rl.core.utils.misc_utils import zipsame, dict_update
from rl.core.utils.mp_utils import JobRunner, Worker


def get_combs_and_keys(ranges):

    keys = []
    values = []
    for r in ranges:
        keys += r[::2]
    values = [list(zipsame(*r[1::2])) for r in ranges]
    cs = itertools.product(*values)
    combs = []
    for c in cs:
        comb = []
        for x in c:
            comb += x
        # print(comb)
        combs.append(comb)
    return combs, keys


def load_config(script_name, config_name):
    """ Load config template by names. """
    def load_attr(module_name, attr_name):
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)

    def get_CONFIG(script_name):
        try:
            CONFIG = load_attr('scripts.'+script_name+'_configs','CONFIG')
        except:
            CONFIG = load_attr('scripts.'+script_name, 'CONFIG')
        return CONFIG

    if config_name is None:  # use the default one
        return get_CONFIG(script_name)

    config_name = 'config_'+config_name
    try:  # try the user-provided version
        config = load_attr('scripts.'+script_name+'_configs', config_name)
    except (AttributeError, ImportError):
        try:  # try the default ones
            config = load_attr('scripts.'+script_name, config_name)
        except AttributeError:
            try:  # try to compose one
                config = copy.deepcopy(get_CONFIG(script_name))
                config_part = load_attr('scripts.configs', config_name)
                dict_update(config, config_part)
            except AttributeError:
                print('Fails to create configs from config_name.')
                raise ValueError
    return config

def save_range(r, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "range.json"), 'w') as out:
        out.write(json.dumps(r, separators=(',\n', '\t:\t'), sort_keys=False))

def main(script_name, range_names, n_processes=-1, config_name=None):
    """ Run the `main` function in script_name.py in parallel with
        `n_processes` with different configurations given by `range_names`.

        Each configuration is jointly specificed by `config_name` and
        `range_names`. If `config_name` is None, it defaults to use the
        `CONFIG` dict in the script file. A valid config is a dict and must
        contains a key 'exp_name' whose value will be used to create the
        indentifier string to log the experiments.

        `range_names` is a list of string, which correspond to a range that
        specifies a set of parameters in the config dict one wish to experiment
        with. For example, if a string "name" is in `range_names`, the dict
        named `range_name` in script_name_ranges.py will be loaded. If
        script_name_ranges.py does not exist, it loads ranges.py.  The values
        of these experimental parameters will be used, jointly with `exp_name`,
        to create the identifier in logging.
    """
    # Set to the number of workers.
    # It defaults to the cpu count of your machine.
    if n_processes == -1:
        n_processes = mp.cpu_count()
    script = importlib.import_module('scripts.'+script_name)
    template = load_config(script_name, config_name)

    try:
        script_ranges = importlib.import_module('scripts.'+script_name+'_ranges')
    except ImportError:
        script_ranges = importlib.import_module('scripts.ranges')

    # Create the configs for all the experiments.
    tps = []
    for range_name in range_names:
        r = getattr(script_ranges, 'range_'+range_name)
        combs, keys = get_combs_and_keys(r)

        # Save the range file
        last_keys = [key[-1] for key in keys]
        if 'top_log_dir' in last_keys:
            ind = last_keys.index('top_log_dir')
            assert all(combs[0][ind]==comb[ind] for comb in combs), 'multiple top_log_dir found'
            top_log_dir = combs[0][ind]
        else:
            top_log_dir = template['top_log_dir']
        save_range(r, top_log_dir)

        print('Total number of combinations: {}'.format(len(combs)))
        for _, comb in enumerate(combs):
            tp = copy.deepcopy(template)
            # Generate a unique exp name based on the provided ranges.
            # The description string start from the the exp name.
            value_strs = [tp['exp_name']]
            for (value, key) in zip(comb, keys):
                entry = tp
                for k in key[:-1]:  # walk down the template tree
                    entry = entry[k]
                # Make sure the key is indeed included in the template,
                # so that we set the desired flag.
                assert key[-1] in entry, 'missing {} in the config'.format(key[-1])
                entry[key[-1]] = value
                if key[-1]=='seed' or key[-1]=='top_log_dir':
                    continue # do not include seed number or the log directory
                else:
                    if value is True:
                        value = 'T'
                    if value is False:
                        value = 'F'
                    value = str(value).replace('/','-')
                    value_strs.append(value)
                    # value_strs.append(str(value).split('/')[0])
                   
            tp['exp_name'] = '-'.join(value_strs)
            tps.append(tp)

    # Launch the experiments.
    n_processes = min(n_processes, len(combs))
    print('# of CPU (threads): {}. Running {} processes'.format(mp.cpu_count(), n_processes))

    # with mp.Pool(processes=n_processes, maxtasksperchild=1) as p:
    #     p.map(script.main, tps, chunksize=1)
    #     # p.map(func, tps, chunksize=1)

    # workers = [Worker(method=script.main) for _ in range(n_processes)]
    # job_runner = JobRunner(workers)
    # jobs = [((tp,),{}) for tp in tps]
    # job_runner.run(jobs)

    workers = [Worker() for _ in range(n_processes)]
    job_runner = JobRunner(workers)
    jobs = [(partial(run_script, script.main, tp),(), {}) for tp in tps]
    job_runner.run(jobs)

def run_script(main, config):
    w = Worker(method=main)
    j = ((config,),{})
    return JobRunner([w]).run([j])


def func(tp):
    print(tp['exp_name'], tp['seed']) #, tp['algorithm']['lambd'])

if __name__ == '__main__':
    mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('script_name')
    parser.add_argument('-r', '--range_names', nargs='+')
    parser.add_argument('-c', '--config_name', type=str)
    parser.add_argument('-n', '--n_processes', type=int, default=-1)
    args = parser.parse_args()
    main(args.script_name, args.range_names,
         n_processes=args.n_processes, config_name=args.config_name)
