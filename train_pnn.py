""" Predict the next state given an input (state,action)-pair using a PNE.
    Estimate (mean,var) of transition function.
"""
import os
from pathlib import Path
from glob import glob
#from tqdm import tqdm
#from concurrent import futures
#from itertools import repeat, product
import numpy as np
from sklearn.utils import shuffle as sk_shuffle
## local imports
from helper.compare import compare
from pnn_nd import PNN
from helper.data import StateActionPairs
from helper.agreement import kl_div

ACTIONS = Path('meta_actions.npy')
STATES = Path('meta_observations.npy')
OUTPUT = Path('ensemble_mean_var.npy')

def train(data, output_path):
    ## PNN Params
    state_dim = data.state_dim
    action_dim = data.action_dim
    n_epochs = 40  ## paper recommendation is 40
    lr = 1e-3
    seeds = [10, 20, 30, 40, 50, 11, 23, 4321, 43, 111]
    #seeds = [i for i in range(10)]

    models = [PNN(state_dim=state_dim, action_dim=action_dim, learning_rate=lr, seed=s) 
              for s in seeds]

    ensemble_mean, ensemble_var = float(0), float(0)
    best_loss = float('inf')
    for i, pnn in enumerate(models):
        train_X, train_y, test_X, test_y = data.train_test_split(train_split=0.90,
                                                                 seed=pnn.seed)
        for epoch in range(n_epochs):
            if epoch != 0:
                train_X, train_y = sk_shuffle(train_X, train_y)
            pnn.step(train_X, train_y)
            #if epoch % 10 == 0:
                #train_error, test_error = pnn.compute_errors(train_X, train_y,
                #                                             test_X, test_y)
                #print(f'----- Training Model {i}, epoch {epoch} -------------------------------------')
                #print(f'----- training error: {train_error:0.2f}, test error: {test_error:0.2f} -----')
            
        mean, var = pnn.forward(data.X)
        #ensemble_mean = max(ensemble_mean, mean)
        #ensemble_var = max(ensemble_var, var)
        #loss, _ = pnn.compute_errors(data.X, data.y)
        print(f'\t---------- Finished processing Model {i} ----------')
        #print(f'\t---------- overall loss: {loss:0.2f} -------------------')
        #if loss < best_loss:
            #best_loss = loss
            #ensemble_mean = mean[0]
            #ensemble_var = var[0]

        ensemble_mean += mean
        ensemble_var += var + np.square(mean)

    # Combine ensembles
    ensemble_mean = np.mean(ensemble_mean, axis=0) / (len(seeds))
    ensemble_var = np.mean(ensemble_var, axis=0) / (len(seeds)) #+ len(data.X))
    ensemble_var = ensemble_var - np.square(ensemble_mean)
    #print(ensemble_var.shape)

    #ensemble_loss = NLL_static(ensemble_mean, ensemble_var, data.y)
    #print(f'ensemble loss: {ensemble_loss:0.2f}')

    np.save(output_path/Path('ensemble_mean_var.npy'), np.concatenate((ensemble_mean, ensemble_var)))
    np.save(output_path/Path('ensemble_mean.npy'), ensemble_mean)
    np.save(output_path/Path('ensemble_var.npy'), ensemble_var)

    return None


#def NLL_static(mean, var, truth):
    #import torch
    #mean = torch.Tensor(mean)
    #var = torch.Tensor(var)
    #truth = torch.Tensor(truth)
    #diff = torch.sub(truth, mean)
    #softplus = torch.log(1 + torch.exp(var))
    #softplus = torch.where(softplus==float('inf'), var, softplus)
    #loss = torch.div(torch.square(diff), 2 * softplus)
    #loss += 0.5 * torch.log(softplus) #+ np.log(2 * np.pi)
    ##print(loss.shape)
    #loss = torch.sum(loss, dim=1)
    ##print(loss.shape)
    #return torch.mean(loss).numpy()
    
def main(args):
    os.chdir(args.base_path)
    dirs = sorted(glob('*'))

    for d in dirs:
        data_path = Path(d)
        valid_path = data_path/ACTIONS
        finished = data_path/OUTPUT
        if valid_path.exists() and not finished.exists():
            print(f'### Training {str(data_path)} ###')
            data = StateActionPairs(args.action_mode, data_path)
            data.build_data()
            print('starting training...')
            train(data, data_path)
            print('finished training...')
        elif finished.exists():
            print('dir already processed')
        else:
            print('invalid dir: meta_actions.npy does not exist')

if __name__ == '__main__':
    from argparse import ArgumentParser
    from datetime import datetime

    starttime = datetime.now()
    parser = ArgumentParser()
    parser.add_argument('--base_path', help='directory containing data, and to hold results',
                        type=str, required=True)
    parser.add_argument('--action_mode', type=str, required=True, 
                        choices=['hopper', 'cheetah', 'walker', 'ant'],
                        help='mujoco env that generated data')
    parser.add_argument('--compare_only', type=bool, default=False,
                        help='If false, trains PNNS. If true, computes KL divs.')
    #parser.add_argument('--glob', type=str, default='*')
    args = parser.parse_args()

    if args.compare_only is True:
        print('Did you remember to change the seed values in compare.py?')
        compare(args)
    else:
        main(args)

    endtime = datetime.now()
    print('all finished!')
    print('total runtime: {}.'.format(endtime - starttime))
