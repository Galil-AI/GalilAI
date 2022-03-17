""" 
Causal Confusion CEM Planner.
Test if causal curiosity can help distinguish between two causal factors
not introduced in training, but in testing.
"""
import random
import time
import os
from datetime import datetime
from pathlib import Path
from functools import partial
import gym
from os.path import exists

import numpy as np
import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from loguru import logger
from tqdm import tqdm
import dill

from gym.wrappers.monitoring.video_recorder import VideoRecorder

import multiprocessing_on_dill as mp
import psutil
# local imports
from cem_planner import CEMPlanner
from plan_action_spaces import get_plan_action_space
from cem.uniform_bounds import UniformBounds
from causal_world.task_generators.task import task_generator
from causal_world.envs.causalworld import CausalWorld


# these are causal factor values used in loops
# modify these for different experiment settings
seeds = [122, 233, 344, 455, 566, 677, 788, 889, 900, 1111]  # random seed
masses = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
sizes = [0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08, 0.085, 0.09] 
frame_skips = [1,2,3,4,5,6,7,8,9]
frictions = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
dampings = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# action state types
action_mode = 'RGB_asym'
obs_dim = 3



class CEMPlannerCW(CEMPlanner):
    """
    Extension of CEMPlanner class.
    New plan function that allows us to specify which env to train in.

    Return: add to the current env list with a list of envs with size len(masses) *  len(sizes) *  len(frictions) *  len(dampings) *  len(gravities) * len(frame_skips)
    
    e.g.  
    envs = []
    envs = addEnvs(envs, sizes = sizes) (create 9 envs with varying sizes with other attributes being constant)
    envs = addEnvs(envs) (add a new env with all default attributes to the env lists (total 10 after))
    envs = addEnvs(envs, masses = [0.1]) (add a new env with mass 0.1, other attributes being constant (total 11 after))

    """
    # plan envs 
    def addEnvs(self, envs, masses = [0.5], sizes = [0.07], frictions = [0.1], dampings = [0.5],  frame_skips = [1]):
        
        for mass in masses:
            for size in sizes:
                for friction in frictions:
                    for damping in dampings:
                        for frame_skip in frame_skips:
                            task = task_generator(task_generator_id ='lifting',
                                tool_block_mass = mass,
                                tool_block_shape = 'cube',
                                tool_block_size = size)
                            env = CausalWorld(task=task, 
                                skip_frame = self.frame_skip,
                                enable_visualization = False)
                            env.setFrictionDamping(f =friction, d= damping)
                            envs.append(env)

        return envs
   


# for multiprocessing
def worker(i, q, training_action_plan, training_rel_duration_plan, training_observations, training_km_sdtw, training_planner):
    # p = psutil.Process()
    # # Params. Play around with these settings. They are not yet optimized.
    # #scenario = 'lift'
    # p.cpu_affinity([i])


    total_budget = 400
    plan_horizon = 6
    n_plan_iterations = 20
    sampler = 'uniform'
    warm_starts = False
    warm_start_relaxation = 0.0
    elite_fraction = 0.1
    viz_progress = True
    n_frames = 198

    plan_action_repeat = np.floor_divide(n_frames, plan_horizon)
    n_plan_cache_k = plan_horizon
    n_plans = np.floor_divide(total_budget * n_plan_cache_k,
                              plan_horizon * n_plan_iterations)
    n_plan_elite = max(1, int(round(elite_fraction * n_plans)))

    if n_plans <= n_plan_elite:
        n_plan_elite = n_plans - 1
    # action_space is gym `Box` env that defines vals for each action [-1,1]
    # action_transf is a func that returns an array for real-val actions? 
    action_space, action_transformation = get_plan_action_space(action_mode)    
   

    training_predict= training_km_sdtw.predict(training_observations)


    for i_size, size in enumerate(sizes):
        for i_mass, mass in enumerate(masses):
            
            envs = training_planner.addEnvs([],masses = [mass], sizes = [size])
            test_observations = training_planner.simulate(envs[0], training_action_plan, training_rel_duration_plan)
            test_predict = training_km_sdtw.predict(test_observations)
            belief_size_cluster = [sizes[i] for i, label in enumerate(training_predict) if label == test_predict[0]]
            
            for i_seed, seed in enumerate(seeds):
                # if exists(f'SizeDamping/Kmeans{i_size}_{i_damping}_{i_seed}.pickle') or exists(f'SizeDamping/temp/Kmeans{i_size}_{i_damping}_{i_seed}.pickle'):
                #     continue            ## start from the last saved value
                # with open(f'SizeDamping/temp/Kmeans{i_size}_{i_damping}_{i_seed}.pickle','a') as f:
                #     pass

                if i_size != 0 or i_mass != 0 or i_seed != 0:
                    continue


                print(f'start training: ({i_size},{i_mass},{i_seed})')
                rng = np.random.RandomState(seed)
                np.random.seed(seed)
          
                belief_planner = CEMPlannerCW(n_plans=n_plans,
                           horizon=plan_horizon,
                           action_space=action_space,
                           sampler=sampler,
                           n_iterations=20,
                           n_elite=n_plan_elite,
                           cache_k=n_plan_cache_k,
                           obs_dim=obs_dim,
                           warm_starts=warm_starts,
                           warm_start_relaxation=warm_start_relaxation,
                           plan_action_repeat=plan_action_repeat,
                           action_transformation=action_transformation,
                           rng=rng,
                           viz_progress=viz_progress,
                           )


                envs = belief_planner.addEnvs([], sizes = belief_size_cluster)
                envs = belief_planner.addEnvs(envs, sizes = [size], masses = [mass])

                print("num of envs to plan: ", len(envs))
                belief_action_plan, belief_duration_plan, belief_observations, belief_km_sdtw, belief_best_return = belief_planner.plan(envs)
                
                labels = belief_km_sdtw.predict(belief_observations)
                print(labels)
                reward = belief_best_return
                score = 1 if (sum(labels) == 1 and labels[-1] == 1) or (sum(labels) == len(labels)-1 and labels[-1] == 0) else 0 
                
                for i in range(len(envs)-1):
                    belief_planner.record_video(envs[i], belief_action_plan,belief_duration_plan, name = f"SizeMass_Training_{i}_", i = i_size, j = i_mass, k=i_seed)
                belief_planner.record_video(envs[-1], belief_action_plan,belief_duration_plan, name = "SizeMass_Causal", i = i_size, j = i_mass, k=i_seed)


                    # dill.dump((score, reward), open(f'SizeDamping/Kmeans{i_size}_{i_damping}_{i_seed}.pickle', 'wb'))



def main(output_dir):
    manager = mp.Manager()
    # use 1 if you don't want multi-processing
    cpu_count = mp.cpu_count()


    # train a new training planner
    retrain_training_planner = True

    # clear the current processing planner's flag
    clear_processing = True
    
    # change into output directory
    if not output_dir.exists():
        os.mkdir(output_dir)
    os.chdir(output_dir)
 

    total_budget = 400
    plan_horizon = 6
    n_plan_iterations = 20
    sampler = 'uniform'
    warm_starts = False
    warm_start_relaxation = 0.0
    elite_fraction = 0.1
    viz_progress = True
    n_frames = 198
    seed = 123
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    plan_action_repeat = np.floor_divide(n_frames, plan_horizon)
    n_plan_cache_k = plan_horizon
    n_plans = np.floor_divide(total_budget * n_plan_cache_k,
                              plan_horizon * n_plan_iterations)
    n_plan_elite = max(1, int(round(elite_fraction * n_plans)))

    if n_plans <= n_plan_elite:
        n_plan_elite = n_plans - 1

    # action_space is gym `Box` env that defines vals for each action [-1,1]
    # action_transf is a func that returns an array for real-val actions? 
    action_space, action_transformation = get_plan_action_space(action_mode) 


    if clear_processing:
        for i in range(9):
            for j in range(9):
                for k in range(10):
                    if exists(f'SizeMass/temp/Kmeans{i}_{j}_{k}.pickle') and not exists(f'SizeMass/Kmeans{i}_{j}_{k}.pickle'):
                        os.remove((f'SizeMass/temp/Kmeans{i}_{j}_{k}.pickle'))
    

    # load pre-trained training planner or retrain
    if retrain_training_planner:
        
        training_planner = CEMPlannerCW(n_plans=n_plans,
                               horizon=plan_horizon,
                               action_space=action_space,
                               sampler=sampler,
                               n_iterations=n_plan_iterations,
                               n_elite=n_plan_elite,
                               cache_k=n_plan_cache_k,
                               obs_dim = obs_dim,
                               warm_starts=warm_starts,
                               warm_start_relaxation=warm_start_relaxation,
                               plan_action_repeat=plan_action_repeat,
                               action_transformation=action_transformation,
                               rng=rng,
                               viz_progress=viz_progress,
                               )

        envs = training_planner.addEnvs([], sizes = sizes)
        print("training envs: ", sizes)
        training_action_plan, training_rel_duration_plan, training_observations, training_km_sdtw, training_best_return = training_planner.plan(envs)
        
        dill.dump(training_action_plan, open('training_action_plan.pickle', 'wb'))
        dill.dump(training_rel_duration_plan, open('training_rel_duration_plan.pickle', 'wb'))
        dill.dump(training_observations, open('training_observations.pickle', 'wb'))
        dill.dump(training_km_sdtw, open('training_km_model.pickle', 'wb'))
        dill.dump(training_best_return, open('training_best_return.pickle', 'wb'))
        dill.dump(training_planner, open('training_planner.pickle', 'wb'))
        
   
    with open('training_action_plan.pickle','rb') as f:
        training_action_plan = dill.load(f)
    with open('training_rel_duration_plan.pickle','rb') as f:
        training_rel_duration_plan = dill.load(f)
    with open('training_observations.pickle','rb') as f:
        training_observations = dill.load(f)
    with open('training_km_model.pickle','rb') as f:
        training_km_sdtw = dill.load(f)
    with open('training_best_return.pickle','rb') as f:
        training_best_return = dill.load(f)
    with open('training_planner.pickle','rb') as f:
        training_planner = dill.load(f)

    manager = mp.Manager()
    q = manager.Queue()
    print("cpu count: ", mp.cpu_count())
    pool = mp.Pool(mp.cpu_count()+2)

    jobs = []
    for i in range(mp.cpu_count()-1):
        job = pool.apply_async(worker, (i, q, training_action_plan, training_rel_duration_plan, training_observations, training_km_sdtw, training_planner))
        jobs.append(job)
        time.sleep(2)


    for job in jobs:
        job.get()

    pool.close()
    pool.join()




if __name__ == '__main__':
    print("CEM Planner CausalWorld")

    ## change this line to store your data in whatever folder you'd like
    output_dir = Path('./pickle/CW')
    print(f"Output dir: {output_dir}")
    print("Make sure this directory is what you expected, if not change it!")
    main(output_dir=output_dir)


