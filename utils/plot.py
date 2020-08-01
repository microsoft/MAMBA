# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Licensed under the MIT License.

import matplotlib
matplotlib.use('Agg')  # in order to be able to save figure through ssh
# matplotlib.use('TkAgg')  # Can change to 'Agg' for non-interactive mode
from matplotlib import pyplot as plt
from matplotlib import cm
import csv, os, argparse
import numpy as np
from utils.plot_configs import Configs


def configure_plot(fontsize, usetex):
    fontsize = fontsize
    matplotlib.rc("text", usetex=usetex)
    matplotlib.rcParams['axes.linewidth'] = 0.1
    matplotlib.rc("font", family="Times New Roman")
    matplotlib.rcParams["font.family"] = "serif"
    matplotlib.rcParams["font.serif"] = "Times"
    matplotlib.rcParams["figure.figsize"] = 12, 8
    matplotlib.rc("xtick", labelsize=fontsize)
    matplotlib.rc("ytick", labelsize=fontsize)


def truncate_to_same_len(arrs):
    min_len = np.min([x.size for x in arrs])
    arrs_truncated = [x[:min_len] for x in arrs]
    return arrs_truncated


def read_attr(csv_path, attr):
    with open(csv_path) as csv_file:
        reader = csv.reader(csv_file, delimiter='\t')
        try:
            row = next(reader)
        except Exception:
            return None
        if attr not in row:
            return None
        idx = row.index(attr)  # the column number for this attribute
        vals = []
        for row in reader:
            vals.append(row[idx])

    vals = [np.nan if v=='' else v for v in vals]
    return np.array(vals, dtype=np.float64)


def main(logdir, value, output_dir=None, filename=None, style=None,
         y_axis=None, y_higher=None, y_lower=None, n_iters=None, legend_loc=0,
         curve_style='percentile'):

    attr = value
    y_axis = attr if y_axis is None else y_axis
    conf = Configs(style)
    subdirs = sorted(os.listdir(logdir))
    subdirs = [d for d in subdirs if d[0] != '.']  # filter out weird things, e.g. .DS_Store
    subdirs = conf.sort_dirs(subdirs)
    fontsize = 30 # 36 if style else 12  # for non style plots, exp name can be quite long
    usetex = False # True if style else False
    configure_plot(fontsize=fontsize, usetex=usetex)
    linewidth = 10 if style else 4

    if style:
        fig, ax = plt.subplots()
        plt.setp(ax.spines.values(), linewidth=3)

        #for axis in ['top','bottom','left','right']:
        #ax.spines[axis].set_linewidth(0.5)

    n_curves = 0
    for exp_name in subdirs:
        exp_dir = os.path.join(logdir, exp_name)
        if not os.path.isdir(exp_dir):
            continue
        data = []
        for root, _, files in os.walk(exp_dir):
            if 'log.txt' in files:
                d = read_attr(os.path.join(root, 'log.txt'), value)
                if d is not None:
                    data.append(d)
        if not data:
            continue
        n_curves += 1
        data = np.array(truncate_to_same_len(data))
        if curve_style == 'std':
            mean, std = np.mean(data, axis=0), np.std(data, axis=0)
            low, mid, high = mean - std, mean, mean + std
        elif curve_style == 'percentile':
            low, mid, high = np.percentile(data, [25, 50, 75], axis=0)
        if n_iters is not None:
            mid, high, low = mid[:n_iters], high[:n_iters], low[:n_iters]
        iters = np.arange(mid.size)
        if not exp_name:
            continue
        color = conf.color(exp_name)

        mask =  np.isfinite(mid)
        plt.plot(iters[mask], mid[mask], label=conf.label(exp_name), color=color, linewidth=linewidth)
        plt.fill_between(iters[mask], low[mask], high[mask], alpha=0.15, facecolor=color)
    if n_curves == 0:
        print('Nothing to plot.')
        return 0
    #if not style:
    plt.xlabel('Iteration', fontsize=fontsize)
    plt.ylabel(y_axis, fontsize=fontsize)

    if not ('dip_all_55_4_exps' in logdir or 'dip_experts_effects_55_09' in logdir):
        legend = plt.legend(loc=legend_loc, fontsize=fontsize-4, frameon=False)
        for line in legend.get_lines():
            line.set_linewidth(6.0)

    plt.autoscale(enable=True, tight=True)
    plt.tight_layout()
    plt.ylim(y_lower, y_higher)
    if not style:
        plt.grid(linestyle='--', linewidth='0.2')

    output_dir = output_dir or logdir
    output_filename = filename or '{}.pdf'.format(value)
    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path)
    plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d','--dir', help='The dir of experiments', type=str)
    parser.add_argument('-v', '--value', help='The column name in the log.txt file', type=str)
    parser.add_argument('-y', '--y_axis', help='Name of the y-axis', type=str, default=None)
    parser.add_argument('-o', '--output_dir', type=str, help='Output directory')
    parser.add_argument('-f,','--filename', type=str, default='', help='Output filename')
    parser.add_argument('--style', type=str, default='', help='Plotting style')
    parser.add_argument('--y_higher', nargs='?', type=float)
    parser.add_argument('--y_lower', nargs='?', type=float)
    parser.add_argument('--n_iters', nargs='?', type=int)
    parser.add_argument('--legend_loc', type=int, default=0)
    parser.add_argument('--curve', type=str, default='percentile', help='percentile, std')

    args = parser.parse_args()

    main(logdir=args.dir,
         value=args.value,
         y_axis=args.y_axis,
         output_dir=args.output_dir,
         filename=args.filename,
         style=args.style,
         y_higher=args.y_higher,
         y_lower=args.y_lower,
         legend_loc=args.legend_loc,
         curve_style=args.curve)

