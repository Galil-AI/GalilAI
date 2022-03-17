""" Sample (state,action) pairs for training & testing """

import random
from pathlib import Path
#from itertools import count
import numpy as np
from sklearn.utils import shuffle as sk_shuffle

ACTIONS = Path('meta_actions.npy')
STATES = Path('meta_observations.npy')

class StateActionPairs:
    def __init__(self, action_mode, folder, action_repeat=10):
        self.folder = Path(folder) if isinstance(folder, str) else folder
        self.action_mode = action_mode
        self.action_repeat = action_repeat  
        self.n_iter = 20
        self.n_plans = 20
        self.n_frames = 60
        self.X, self.y = None, None

        if action_mode == 'hopper':
            self.state_dim, self.action_dim = 11, 3
        elif action_mode == 'cheetah':
            self.state_dim, self.action_dim = 17, 6
        elif action_mode == 'walker':
            self.state_dim, self.action_dim = 17, 6
        #elif action_mode == 'ant':
        #    self.state_dim, self.action_dim = 111, 8
        else:
            raise NotImplementedError

    def build_data(self):
        print('Building data...')
        actions = np.load(self.folder/ACTIONS)
        states = np.load(self.folder/STATES)
        #self.n_iter, self.n_plans, self.n_envs, self.n_frames, state_dim = states.shape
        #_, _, self.action_repeat, action_dim = actions.shape
        # Test for mismatched shapes, fingers crossed lol 
        #assert action_dim == self.action_dim and state_dim == self.state_dim
        self._label_data(states, actions)
        del states
        del actions


    def _label_data(self, states, actions):
        X, y = [], []

        for i, (s_t1, a_t) in enumerate(zip(states, actions)):
            if i % self.n_frames == 0:
                pass
            s_t = states[i-1]
            X.append(np.concatenate((s_t, a_t)))
            y.append(s_t1)
        
        self.X = np.array(X)
        self.y = np.array(y)
        del X
        del y


    def train_test_split(self, train_split=0.8, seed=None):
        n_train = int(train_split * len(self.X))
        X, y = sk_shuffle(self.X, self.y, random_state=seed)

        train_X = X[:n_train]
        train_y = y[:n_train]
        test_X = X[n_train:]
        test_y = y[n_train:]

        return train_X, train_y, test_X, test_y


    def shape(self):
        """ Returns dimentions of state and action space for given action mode """
        return (self.state_dim, self.action_dim)
        

    def __len__(self):
        return len(self.X)
