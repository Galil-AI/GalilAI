"""
plan CC experiments on mujoco environments
"""
import os
import pickle
from itertools import repeat, product
from pathlib import Path

import numpy as np
#import moviepy.editor as mpy
import gym
from gym.wrappers import Monitor
from tslearn.clustering import silhouette_score, TimeSeriesKMeans
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from concurrent import futures

# local imports
from plan_action_spaces import get_plan_action_space
from CC_Classes import ConfusionPlanner, CausalEnvironments, vanilla_planner
from frame_skip_planner import predict, fit_and_predict, reward

## glob
TRUE = 1
FALSE = 0
g_X = 0
g_Y = 1
g_Z = 2

def record_video(planner, env, action_plan, rel_dur, friction=1, mass=1, file_name='test_vid.mp4'):

        effective_horizon = planner.horizon * planner.plan_action_repeat
        plan_durations = np.round(rel_dur * effective_horizon).astype(int)

        tmp = env.model.body_mass
        env.model.body_mass[:] = mass*tmp
        tmp = env.model.geom_friction
        env.model.geom_friction[:] = friction*tmp
        env.reset()

        #e = Monitor(env, 'mujoco_vids', force=True)
        recorder = VideoRecorder(env, file_name)
        #print('test')
        recorder.capture_frame()
        #print('test')

        for i, action in enumerate(action_plan):
            action_repeat = plan_durations[i]
            for i_repeat in range(action_repeat):
                next_state, reward, _, _ = env.step(action)
                recorder.capture_frame()

        recorder.close()
        print("Finished recording.")

        return

def run_mujoco(masses, test_mass, args):

    assert isinstance(args, tuple)
    test_grav_z, seed = args

    seed_path = Path(f'planner_seed{seed}')
    #seed_path_continue = Path(f'planner_seed{seed}/initial_actions.npy')
    _p = f"results_testgrav{test_grav_z:0.1f}_testmass{test_mass:0.1f}_seed{seed}"
    p = Path(_p)
    if not p.exists():
        p.mkdir()
    elif Path(_p+'/final_predict.npy').exists():
        print('This environment permutation has already been processed.')
        return None, None

    ## set up initial planner
    action_mode = 'hopper'
    n_frames = 100 #50 #100
    planner = vanilla_planner(action_mode=action_mode, n_frames=n_frames, seed=seed)

    seed_path_continue = p/Path('meta_observations.npy')

###############
    if not seed_path_continue.exists():
        if not seed_path.exists():
            seed_path.mkdir()
        train_gravs = [9.8 for i in range(len(masses))]

        init_actions, init_obs, rel_dur_plan, meta_obs, meta_actions = planner.plan_mujoco(
                                                                                    masses=masses,
                                                                                    gravity=train_gravs,
                                                                                    gravity_axis=g_X
                                                                                )

        #np.save(p/'meta_observations.npy', np.array(meta_obs))
        #np.save(p/'meta_actions.npy', np.array(meta_actions))
        np.save(seed_path/'initial_actions.npy', init_actions)
        np.save(seed_path/'init_obs.npy', init_obs)
        np.save(seed_path/'rel_dur_plan.npy', rel_dur_plan)
    else:
        init_actions = np.load(seed_path/'initial_actions.npy')
        init_obs = np.load(seed_path/'init_obs.npy')
        rel_dur_plan = np.load(seed_path/'rel_dur_plan.npy')
###############

    ## perform simulation on slippery environment
    sim_obs = planner.sim_mujoco(init_actions, rel_dur_plan, gravity=test_grav_z, mass=test_mass)

    init_preds, sim_preds = fit_and_predict(init_obs, sim_obs)

    initial_predictions = np.concatenate([init_preds, sim_preds])
    np.save(p/'initial_prediction.npy', initial_predictions)

    initial_observations = np.concatenate([init_obs, sim_obs])
    np.save(p/'initial_observations.npy', initial_observations)

    initial_reward = reward(initial_observations, initial_predictions)
    np.save(p/'initial_reward.npy', np.array((initial_reward)))

    #initial_predictions = np.load(seed_path/'initial_prediction.npy')
    #initial_observations = np.load(seed_path/'initial_observations.npy')

    #env = gym.make('Hopper-v3')
    #file_name = _p + '/initial_test_actions.mp4'
    #record_video(planner, env, initial_actions, rel_dur_plan,
    #             friction=test_friction, mass=test_mass, file_name=file_name)

    #file_name = _p + '/initial_main_actions.mp4'
    #record_video(planner, env, initial_actions, rel_dur_plan,
    #             friction=1, mass=1, file_name=file_name)

    belief_cluster_masses = []
    cluster_grav_z = []
    for i, mass in enumerate(masses):
        if initial_predictions[i] == initial_predictions[-1]:
            belief_cluster_masses.append(mass)
            cluster_grav_z.append(9.8)
    if len(belief_cluster_masses) == 1:
        belief_cluster_masses.append(belief_cluster_masses[0])
        belief_cluster_masses.append(test_mass)
        belief_cluster_masses.append(test_mass)
        cluster_grav_z = [9.8, 9.8, test_grav_z, test_grav_z]
    else:
        belief_cluster_masses.append(test_mass)
        cluster_grav_z.append(test_grav_z)

    assert len(cluster_grav_z) == len(belief_cluster_masses) and "need to be same len"

    ## plan on belief cluster
    cluster_actions, cluster_observations, _, meta_cluster_obs, meta_cluster_actions= planner.plan_mujoco(
                action_mode=action_mode,
                masses=belief_cluster_masses,
                gravity=cluster_grav_z,
                gravity_axis=g_X
    )

    cluster_predictions = predict(cluster_observations)
    cluster_reward = reward(cluster_observations, cluster_predictions)
    print("Belief cluster predictions are: ", cluster_predictions)
    print("Cluster reward is: ", cluster_reward)

    np.save(p/'meta_cluster_actions.npy', meta_cluster_actions)
    np.save(p/'meta_cluster_observations.npy', meta_cluster_obs)
    np.save(p/'cluster_predictions.npy', cluster_predictions)
    np.save(p/'cluster_observations.npy', cluster_observations)
    np.save(p/'cluster_actions.npy', cluster_actions)
    np.save(p/'cluster_reward.npy', np.array((cluster_reward)))
    #cluster_actions = np.load(p/'cluster_actions.npy')
    #cluster_observations = np.load(p/'cluster_observations.npy')
    #cluster_actions = np.load(p/'cluster_actions.npy')

    #file_name = _p + '/final_test_actions.mp4'
    #record_video(planner, env, cluster_actions, rel_dur_plan,
    #             friction=test_friction, mass=test_mass, file_name=file_name)

    #file_name = _p + '/final_main_actions.mp4'
    #record_video(planner, env, cluster_actions, rel_dur_plan,
    #             friction=1, mass=1, file_name=file_name)

# return values
    unique = np.unique(cluster_predictions[:-1])
    if len(unique) == 1 and unique[0] != cluster_predictions[-1]:
        np.save(p/'final_predict.npy', np.array([TRUE]))
        return TRUE, cluster_reward
    else:
        np.save(p/'final_predict.npy', np.array([FALSE]))
        return FALSE, cluster_reward

def print_args(masses, test_mass, args):
    #params = "Training masses: {} ".format(masses)
    params = "t_mass: {}".format(test_mass)
    params += "(test_grav_z, seed: {} ".format(args)
    return params

def main(output_dir):

    if not output_dir.exists():
        output_dir.mkdir()
    os.chdir(output_dir)

    train_masses = [0.202, 0.403, 0.601, 0.709, 1.1, 1.202, 1.37, 1.602, 1.799]
    test_mass = [0.2, 0.6, 1.0, 1.4, 1.8]
    #test_grav_z = [0.2, 0.6, 1.0, 1.4, 1.8]
    test_grav_x = [2.0, 5.9, 9.8, 13.7, 17.6]  ## REMEMBER TO CHANGE CC_Classes AS WELL!!!
    # only do the first half of the seeds for now
    seeds = [4, 8, 28, 41, 44] #, 58, 59, 59, 94, 97]


    #final_predictions = []
    #final_rewards = []

    with futures.ProcessPoolExecutor() as executor:
        for m in test_mass:
            results = executor.map(
                run_mujoco, #print_args, #run_mujoco,
                repeat(train_masses),
                repeat(m),
                product(test_grav_x, seeds)
            )

            for j, result in enumerate(results):
                print(f'result {j}: ', result)
                #final_predictions.append(result[0])
                #final_rewards.append(result[1])

    #print(f"final predictions: {final_predictions}")
    #print(f"final rewards: {final_rewards}")

    #np.save("final_predictions.npy", np.array(final_predictions))
    #np.save("final_rewards.npy", np.array(final_rewards))

    print("All finished!")

if __name__ == "__main__":
    from datetime import datetime
    begin_time = datetime.now()
    print("Start time: ", begin_time)

    output_dir = Path("hopper-trainMG9_8X-testGX")
    main(output_dir)

    total_time = datetime.now() - begin_time
    print(f"Total execution time: {total_time}")


