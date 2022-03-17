'''
Frame skip planner to be used with CCPlanner
'''
import os
import pickle
#import dill
from pathlib import Path
from itertools import repeat, product
from concurrent import futures
#import multiprocessing

import numpy as np
from loguru import logger
import matplotlib.pyplot as plt
from tslearn.clustering import silhouette_score, TimeSeriesKMeans

# local imports
from helper.plan_action_spaces import get_plan_action_space
from causal_confusion_planner import CausalEnvironments, ConfusionPlanner
#from hopper_trainmass_testgrav import print_args

############# glob ###################################################
TRUE = 1
FALSE = 0
MASS = 0
SIZE = 1
SHAPE = 2
STD_FS = 10
######################################################################

def init_planner(seed=1235):
    ''' wrapper method to return untrained planner with vanilla params '''
    # Params. Play around with these settings. They are not yet optimized.
    #scenario = 'lift'
    n_frames_per_episode = 198; total_budget = 400; plan_horizon = 6
    n_plan_iterations = 20; action_mode = 'RGB_asym'
    sampler = 'uniform'; warm_starts = False; warm_start_relaxation = 0.0
    elite_fraction = 0.1; viz_progress = True; n_plan_iterations = 20

    plan_action_repeat = np.floor_divide(n_frames_per_episode, plan_horizon)
    n_plan_cache_k = plan_horizon
    n_plans = np.floor_divide(total_budget * n_plan_cache_k,
                              plan_horizon * n_plan_iterations)
    n_plan_elite = max(1, int(round(elite_fraction * n_plans)))
    if n_plans <= n_plan_elite:
        n_plan_elite = n_plans - 1

    ## seed value is now an optional function argument
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    action_space, action_transformation = get_plan_action_space(action_mode)
    planner = ConfusionPlanner(n_plans=n_plans,
                               horizon=plan_horizon,
                               action_space=action_space,
                               sampler=sampler,
                               n_iterations=n_plan_iterations,
                               n_elite=n_plan_elite,
                               cache_k=n_plan_cache_k,
                               warm_starts=warm_starts,
                               warm_start_relaxation=warm_start_relaxation,
                               plan_action_repeat=plan_action_repeat,
                               action_transformation=action_transformation,
                               rng=rng,
                               viz_progress=viz_progress)

    return planner

def fit_and_predict(obs, test_obs):
    '''
    wrapper method to 1) fit TSKMModel to initial observations data,
                  and 2) predict cluster membership of both inital and test data
    '''
    model = TimeSeriesKMeans(
                    n_clusters=2,
                    metric="softdtw",
                    max_iter=100,
                    max_iter_barycenter=5,
                    metric_params={"gamma": .5},
                    random_state=0
                    ).fit(obs)

    return model.predict(obs), model.predict(test_obs)

def predict(obs):
    ''' wrapper method to fit and predict TSKMeans model to observation data '''
    y = TimeSeriesKMeans(
                    n_clusters=2,
                    metric="softdtw",
                    max_iter=100,
                    max_iter_barycenter=5,
                    metric_params={"gamma": .5},
                    random_state=0
                    ).fit_predict(obs)

    return y


def reward(observations, predictions):
    ''' wrapper method for silhouette_score '''
    return silhouette_score(observations, predictions, metric='dtw')


def plan_frameskip(main_env, args, seed):
    assert isinstance(args, tuple)
    #print(args)
    test_fs, test_mass = args
    test_env = CausalEnvironments([test_mass], 
                                  [0.05], ['cube'],
                                  frame_skip=test_fs)

    seed_path = Path(f'planner_seed{seed}')
    _p = f'results_fs{test_env.frame_skip:0.1f}_mass{test_env.masses[MASS]:0.2f}_seed{seed}'
    p = Path(_p)
    if not p.exists():
        p.mkdir()
    elif Path(_p+'final_predict.npy').exists():
        print('This environment permutation has already been processed.')
        return None, None

    seed_path_continue = p/Path('initial_observations.npy')    

    ## grab a vanilla planner
    planner = init_planner(seed)

    if not seed_path_continue.exists():
        if not seed_path.exists():
            seed_path.mkdir()


        initial_actions, init_obs, rel_dur, _, _ = planner.plan_environment(
                                                        main_env,
                                                        frame_skip=main_env.frame_skip
                                                    )
        # saving...lol
        np.save(seed_path/'initial_actions.npy', initial_actions)
        np.save(seed_path/'initial_observations.npy', init_obs)
        np.save(seed_path/'rel_dur_plan.npy', rel_dur)
    else:
        initial_actions = np.load(seed_path/'initial_actions.npy')
        init_obs = np.load(seed_path/'initial_observations.npy')
        rel_dur = np.load(seed_path/'rel_dur_plan.npy')

    ## perform initial simulation on test_env with planner trained on main_env
    sim_obs = planner.do_simulation(
        test_env, initial_actions,
        rel_dur, frame_skip=test_env.frame_skip
    )

    ## fit and predict initial observations, predict sim observations
    init_preds, sim_preds = fit_and_predict(init_obs, sim_obs)

    ## save results from initial training
    initial_observations = np.concatenate([init_obs, sim_obs])
    initial_predictions = np.concatenate([init_preds, sim_preds])
    np.save(p/'initial_predictions.npy', initial_predictions)
    np.save(p/'initial_observations.npy', initial_observations)

    ## reward for initial clusters
    initial_reward = reward(initial_observations, initial_predictions)
    np.save(p/'initial_reward.npy', np.array([initial_reward]))

    belief_cluster = CausalEnvironments([], [], [])
    for i, env in enumerate(main_env.envs):
        if init_preds[i] == sim_preds: 
            belief_cluster.appendEnv(env)

    # for edge case of cluster of 2, copy first env over twice
    if len(belief_cluster.envs) == 1:
        belief_cluster.envs.append(belief_cluster.envs[0])
        belief_cluster.envs.append(test_env.envs[0])
        belief_cluster.envs.append(test_env.envs[0])
        belief_fs = [STD_FS, STD_FS,
                     test_env.frame_skip, test_env.frame_skip]
        belief_cluster.frame_skip = belief_fs
    else:
        belief_cluster.appendEnv(test_env.envs[0])
        belief_fs = [STD_FS for _ in range(len(belief_cluster.envs)-1)]
        belief_fs.append(test_env.frame_skip)
        setattr(belief_cluster, 'frame_skip', belief_fs)

    with open(p/f"belief_cluster_seed{seed}_mass{test_env.envs[0][MASS]}.pke", "wb") as fp:
        pickle.dump(belief_cluster.envs, fp)

    cluster_actions, cluster_observations, _, cluster_meta_obs , cluster_meta_actions = planner.plan_environment(
        belief_cluster,
        frame_skip=belief_fs
    )

    cluster_predictions = predict(cluster_observations)
    cluster_reward = reward(cluster_observations, cluster_predictions)
    np.save(p/'meta_cluster_actions.npy', cluster_meta_actions)
    np.save(p/'meta_cluster_observations.npy', cluster_meta_obs)
    np.save(p/'cluster_reward.npy', np.array([cluster_reward]))
    np.save(p/'cluster_observations.npy', cluster_observations)
    np.save(p/'cluster_predictions.npy', cluster_predictions)
    np.save(p/'cluster_actions.npy', cluster_actions)

    ## visualize belief cluster action plan
    #file_name = _p+f"/main_cluster_actions_seed{seed}_mass{belief_cluster.envs[0][MASS]}.mp4"
    #planner.record_video(belief_cluster.envs[0], cluster_actions, rel_dur,
             #            frame_skip=main_env.frame_skip,
             #            file_name=file_name)
    #file_name = _p+f"/test_cluster_seed{seed}_mass{belief_cluster.envs[-1][MASS]}.mp4"
    #planner.record_video(belief_cluster.envs[-1], cluster_actions, rel_dur,
             #            frame_skip=test_env.frame_skip,
             #            file_name=file_name)

    ## for success, last env should be in its own cluster
    unique = np.unique(cluster_predictions[:-1])
    if len(unique) == 1 and unique[0] != cluster_predictions[-1]:
        np.save(p/'final_predict.npy', np.array([TRUE]))
        return TRUE, cluster_reward
    else:
        np.save(p/'final_predict.npy', np.array([FALSE]))
        return FALSE, cluster_reward


def print_args(masses, test_mass, args):
    params = f'test mass: {test_mass}'
    params+= f'(test fs, seed): {args}'
    return params



def main(output_dir):
    print("Starting training...")

    if not output_dir.exists():
        output_dir.mkdir()
    os.chdir(output_dir)

    seeds = [2, 6, 29, 39, 47] #, 50, 71, 79, 91, 98]
    fs_vals = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    #global STD_FS
    #STD_FS = 5
    train_masses = [0.09, 0.121, 0.159, 0.182, 0.203, 0.244, 0.29, 0.322, 0.35, 0.38]
    test_masses = [0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.32]
    size = [0.05]
    shape = ['cube']

    ## create causal environments
    main_env = CausalEnvironments(train_masses, size, shape, frame_skip=STD_FS)

    test_envs = []
    for fs in fs_vals:
        for mass in test_masses:
            tmp = CausalEnvironments([mass], size, shape, frame_skip=fs)
            test_envs.append(tmp)

    ## save envs to a text file for later
    with open("main_env.pke", "wb") as fp:
        pickle.dump(main_env, fp)
    with open("test_envs.pke", "wb") as fp:
        pickle.dump(test_envs, fp)

    final_predictions = []
    final_rewards = []

    ## run each test_env for a particular seed in parallel
    ## then iteratate thru seed vals and repeat...
    with futures.ProcessPoolExecutor() as executor:
        for seed in seeds:
            results = executor.map(
                plan_frameskip,
                repeat(main_env),
                product(fs_vals, test_masses),
                repeat(seed)
            )

            for j, result in enumerate(results):
                print(f'result {j}: ', result)

    print("All finished !")


if __name__ == '__main__':
    from datetime import datetime

    begin = datetime.now()
    print('Start time: ', begin)

    output_dir = Path("CWpush-FS-trainmassFS10-testFS")
    main(output_dir)

    finish = datetime.now() - begin
    print(f'Total execution time: {finish}')




