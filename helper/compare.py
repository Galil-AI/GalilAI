""" Execute KL div between two distributions in resp. dirs """
import os
from pathlib import Path
from glob import glob
import numpy as np
from helper.agreement import kl_div
#from agreement import kl_div
DATA = Path('ensemble_mean_var.npy')


def compare(args):
    print('You must currently specify exactly what random seeds you used during training')
    print("Otherwise this script won't work yet")
    os.chdir(args.base_path)
    ## preliminaries
    planner = 'planner'
    baseline = 'baseline'
    results = 'results'
    ## Change this to suit your needs!
    seeds = ['seed14', 'seed28', 'seed33', 'seed64', 'seed101', 
             'seed109', 'seed144', 'seed171', 'seed181', 'seed189']
    baseline_seeds = ['seed111', 'seed123', 'seed145', 'seed156', 'seed164']
    dirs = os.listdir()

    for seed in baseline_seeds:
        print(seed)
        seed_dirs = [d for d in dirs if seed in d]
        test_dirs = [d for d in seed_dirs if planner not in d]
        planner_dir = Path([d for d in seed_dirs if planner in d][0])

        training_data = np.load(planner_dir/DATA)
        n = int(len(training_data) / 2)
        training_mean = training_data[:n]
        training_var = training_data[n:]

        for test_env in seed_dirs:
            env = Path(test_env)
            test_data = np.load(env/DATA)
            test_mean = training_data[:n]
            test_var = test_data[n:]
            KL = kl_div(test_mean, test_var, training_mean, training_var)
            np.save(env/'kl.npy', KL)

    for seed in seeds:
        print(seed)
        seed_dirs = [d for d in dirs if seed in d]
        test_dirs = [d for d in seed_dirs if planner not in d]
        planner_dir = Path([d for d in seed_dirs if planner in d][0])
        training_data = np.load(planner_dir/DATA)
        n = int(len(training_data) / 2)
        training_mean = training_data[:n]
        training_var = training_data[n:]

        for test_env in seed_dirs:
            env = Path(test_env)
            test_data = np.load(env/DATA)
            test_mean = training_data[:n]
            test_var = test_data[n:]
            KL = kl_div(test_mean, test_var, training_mean, training_var)
            np.save(env/'kl.npy', KL)
            #print(KL)


