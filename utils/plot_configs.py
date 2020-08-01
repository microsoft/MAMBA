# Copyright (c) 2019 Georgia Tech Robot Learning Lab
# Licensed under the MIT License.

from matplotlib import cm
from itertools import chain


SET1COLORS = cm.get_cmap('Set1').colors
SET2COLORS = cm.get_cmap('Set2').colors
SET3COLORS = cm.get_cmap('Set3').colors
TAB20COLORS = cm.get_cmap('tab20').colors

COLOR = {
        'red': SET1COLORS[0],
        'blue': SET1COLORS[1],
        'lightblue': SET2COLORS[2],
        'green': SET1COLORS[2],
        'purple': SET1COLORS[3],
        'grey': SET1COLORS[8],
        'darkgreen': SET2COLORS[0],
        'orange': SET1COLORS[4],
        'pink': SET2COLORS[3],
        'lightgreen': SET2COLORS[4],
        'gold': SET2COLORS[5],
        'brown': SET1COLORS[6],
        }



# configs for mamba paper
mamba_configs = {
    'mamba-0.9-max': ('MAMBA-0.9-max',  COLOR['red']),
    'mamba-0.9-mean': ('MAMBA-0.9-mean',  COLOR['blue']),
    'mamba-0.9-max_': ('MAMBA-0.9-max',  COLOR['purple']),
    'mamba-0.5-max': ('MAMBA-0.5-max',  COLOR['gold']),
    'mamba-0.1-max': ('MAMBA-0.1-max',  COLOR['pink']),
    'order': ['mamba-0.9-max', 'mamba-0.9-mean',
              'mamba-0.9-max_', 'mamba-0.5-max', 'mamba-0.1-max']
}

mamba_n_experts_color = ['red', 'orange', 'lightblue', 'purple']
for lambd in ['0.0', '0.1', '0.5', '0.9']:
    for i, n in enumerate([8,4,2,1]):
        dir_name = 'mamba-'+lambd+'-max('+str(n)+')'
        leg_name = 'MAMBA-'+lambd+'-max('+str(n)+')'
        color = COLOR[mamba_n_experts_color[i]]
        mamba_configs[dir_name] = (leg_name,color)
        mamba_configs['order'].append(dir_name)


for n in range(8):
    dir_name = 'aggrevated-expert'+str(n)
    leg_name = 'AggreVaTeD-'+str(n)
    if n<7:
        color = SET2COLORS[n]
    else:
        color = COLOR['blue']

    mamba_configs[dir_name] = (leg_name,color)
    mamba_configs['order'].append(dir_name)

mamba_configs['pg-gae-0.9'] = ('PG-GAE-0.9', COLOR['grey'])
mamba_configs['aggrevated'] = ('AggreVaTeD',  COLOR['green'])
mamba_configs['order'].extend(['aggrevated', 'pg-gae-0.9'])


class Configs(object):
    def __init__(self, style=None, colormap=None):
        if not style:
            self.configs = None
            if colormap is None:
                c1 = iter(cm.get_cmap('Set1').colors)
                c2 = iter(cm.get_cmap('Set2').colors)
                c3 = iter(cm.get_cmap('Set3').colors)
                self.colors = chain(c1, c2, c3)
            else:
                self.colors = iter(cm.get_cmap(colormap).colors)
        else:
            self.configs = globals()[style + '_configs']
            for exp_name in self.configs['order']:
                assert exp_name in self.configs, 'Unknown exp: {}'.format(exp_name)

    def color(self, exp_name):
        if self.configs is None:
            color = next(self.colors)
        else:
            color = self.configs[exp_name][1]
        return color

    def label(self, exp_name):
        if self.configs is None:
            return exp_name
        return self.configs[exp_name][0]

    def sort_dirs(self, dirs):
        if self.configs is None:
            return dirs

        def custom_key(exp_name):
            if exp_name in self.configs['order']:
                return self.configs['order'].index(exp_name)
            else:
                return 100
        return sorted(dirs, key=custom_key)
