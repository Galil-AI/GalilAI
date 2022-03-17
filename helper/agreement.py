""" Wrapper methods to calculate agreement score b/w distributions """
import os
from pathlib import Path
import numpy as np
from numpy.linalg import det, pinv
#import scipy.special
#from torch import nn

def kl_div(p_mean, q_mean, p_var, q_var):
    """ KL divergence of two multivariate gaussians N_P, N_Q
        relative entropy from Q to P
        for more info see: The Matrix Cookbook
    """
    # preliminaries
    k = len(p_mean)
    p_var_diag = np.diag(p_var)
    q_var_diag = np.diag(q_var)
    #print(q_var.shape)
    #print(q_var_diag)
    q_var_inv = pinv(q_var_diag)
    diff = p_mean - q_mean
    # compute divergence
    div = np.log(np.divide(det(p_var_diag), det(q_var_diag)))
    #print(div)
    div -= k
    div += diff.T @ q_var_inv @ diff
    div += np.trace(q_var_inv @ p_var_diag)
    div *= 0.5
    return div

if __name__ == '__main__':
    print('wrong file.')

