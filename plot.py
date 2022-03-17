import os
import sys
#sys.path.append('../data/')
from pathlib import Path
from glob import glob

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
from mpl_toolkits import mplot3d
from matplotlib import cm

# seaborn style options
sns.set_theme()
#sns.set_style("whitegrid")
colors = sns.color_palette("vlag", as_cmap=True)
#colors = sns.diverging_palette(220, 20, as_cmap=True)

# numpy print options
float_formatter = "{:.1f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

# matplotlib style options
#plt.style.use('ggplot')
cmap = cm.get_cmap('winter')
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.markersize'] = 3
plt.rcParams.update({'axes.titlesize': '15'})


def path_to_array(path, param1, param2, num_seeds, grav=False):
    if not isinstance(path, str):
        path = str(path)
    data = np.zeros([len(param1) * len(param2) * num_seeds])
    paths = sorted(glob(path))
    print(len(paths))
    for i, f in enumerate(paths):
        #if i > 999: print(i)
        data[i] = np.load(f)
        
    data.resize([len(param1), len(param2), num_seeds])
    
    #data = np.sum(data, axis=2)
    print(data.shape)
    return data

## Constants
dirs = ['Hopper-trainMass-testGravX']
final_predict = Path('results*/final_predict.npy')
seeds = [6, 10, 24, 42, 45, 56, 58, 66, 73, 97]
masses = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
ood = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
ood = [np.round(i*9.81, decimals=2) for i in ood]
n = 10

data = np.zeros((11, 10, 10))

all_files = sorted(glob('Hopper-trainMass-testGravX/*/*'))

gravity_vals = ['gravity0.0', 'gravity1.9', 'gravity3.9', 'gravity5.8',
                'gravity7.8', 'gravity9.8', 'gravity11.7', 'gravity13.7',
                'gravity15.7', 'gravity17.6', 'gravity19']

for i, val in enumerate(gravity_vals):
    dirs = [f for f in all_files if val in f]
    c = 0
    for j in range(10):
        for k in range(10):
            data[i,j,k] = np.load(dirs[c])
            c += 1

data = data.sum(axis=2)
oood = [0.0] + ood
ood_ticks = [np.round(i, 1) for i in oood]
sns.heatmap(data.T, annot=True, cmap=colors,
            xticklabels=ood_ticks, yticklabels=masses)
plt.ylabel('Test Mass')
plt.xlabel('Test Gravity (X)')
plt.title('Hopper - Train Mass, Test Gravity (X)')
plt.savefig('Hopper-trainMass-testGravX')
plt.show()

KL = np.zeros((10, 11, 10))
main_path = Path('kl_data/Hopper-trainMass-testGravX/')
all_dirs = os.listdir(main_path)
all_dirs = [i for i in all_dirs if 'planner' not in i]

for i, mass in enumerate(mass_paths):
    for j, grav in enumerate(gravityX_paths):
        threshold = np.zeros(len(baseline_seeds))
        for k, seed in enumerate(baseline_seeds):
            data_path = main_path/Path(f'baseline_{mass}_{grav}_{seed}')/Path('kl.npy')
            threshold[k] = np.load(data_path)
        threshold = np.mean(threshold)
        for k, seed in enumerate(seed_paths):
            data_path = main_path/Path(f'results_{mass}_{grav}_{seed}')/Path('kl.npy')
            curr_kl = np.load(data_path)
            KL[i,j,k] = curr_kl > threshold
data = KL.sum(axis=2)

#ood_ticks = [np.round(i, 1) for i in ood]

sns.heatmap(data, annot=True, cmap=colors, vmax=10, vmin=0,
            xticklabels=oood, yticklabels=masses)

plt.ylabel('Test Mass')
plt.xlabel('Test Gravity (X)')
plt.title('Hopper - Train Mass, Test Gravity (X) Baseline')
plt.savefig('Hopper-trainMass-testGravX-baseline')
