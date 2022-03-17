""" plan CC experiments on mujoco environments """
import os
import pickle
from itertools import repeat, product
from pathlib import Path

import numpy as np
import gym
from gym.wrappers import Monitor
from tslearn.clustering import silhouette_score, TimeSeriesKMeans
from gym.wrappers.monitoring.video_recorder import VideoRecorder

from concurrent import futures

# local imports
from helper.plan_action_spaces import get_plan_action_space
from causal_confusion_planner_mujoco import ConfusionPlanner, vanilla_planner
from frame_skip_planner import predict, fit_and_predict, reward

## glob
TRUE = 1
FALSE = 0

#def record_video(planner, env, action_plan, rel_dur, friction=1, mass=1, file_name='test_vid.mp4'):
#
#        effective_horizon = planner.horizon * planner.plan_action_repeat
#        plan_durations = np.round(rel_dur * effective_horizon).astype(int)
#
#        tmp = env.model.body_mass
#        env.model.body_mass[:] = mass*tmp
#        tmp = env.model.geom_friction
#        #env.model.geom_friction[:] = friction*tmp
#        env.reset()
#
##        recorder = VideoRecorder(env, file_name)
#        recorder.capture_frame()
#
#        for i, action in enumerate(action_plan):
#            action_repeat = plan_durations[i]
#            for i_repeat in range(action_repeat):
#                next_state, reward, _, _ = env.step(action)
#                recorder.capture_frame()
#
#        recorder.close()
#        print("Finished recording.")

def run_mujoco(train_id, test_id, args, product_args):
    assert isinstance(product_args, tuple)
    causal_ood, seed = product_args

    if args.experiment == 'mass':
        id_name == 'friction'
    else:
        id_name == 'mass'

    #print(args.action_mode)
    if args.baseline_only is True:
        _p = f'baseline_test{id_name}{test_mass:0.2f}_test{args.experiment}{causal_ood:0.2f}_seed{seed}'
    else:
        _p = f'results_test{id_name}{test_mass:0.2f}_test{args.experiment}{causal_ood:0.2f}_seed{seed}'
    p = Path(_p)
    if not p.exists():
        p.mkdir()

    action_mode = str(args.action_mode)
    n_frames = 60
    planner = vanilla_planner(action_mode=action_mode, n_frames=n_frames, seed=seed)

    if args.baseline_only is True:
        seed_path = Path(f'planner_baseline_seed{seed}')
    else:
        seed_path = Path(f'planner_seed{seed}')

    seed_path_continue = seed_path/Path('meta_observations.npy')
    ###############
    if not seed_path_continue.exists():
        if not seed_path.exists():
            seed_path.mkdir()

        #train_masses = [1 for _ in range(len(masses))]
        if args.experiment == 'gravity' or args.experiment == 'wind':
            init_actions, init_obs, meta_obs, meta_actions = planner.plan_mujoco(masses=train_id)
        else:
            init_actions, init_obs, meta_obs, meta_actions = planner.plan_mujoco(friction_vals=train_id)
        #else:
        #    in_distribution = [args.gravity_id for i in range(len(masses))]
        #    init_actions, init_obs, rel_dur_plan, meta_obs, meta_actions = planner.plan_mujoco(
        #        masses=masses,
        #        gravity=in_distribution,
        #        gravity_axis=args.gravity_axis
        #    )
        np.save(seed_path/'meta_observations.npy', np.array(meta_obs))
        np.save(seed_path/'meta_actions.npy', np.array(meta_actions))
        np.save(seed_path/'initial_actions.npy', init_actions)
        np.save(seed_path/'init_obs.npy', init_obs)
        #np.save(seed_path/'rel_dur_plan.npy', rel_dur_plan)
    else:
        init_actions = np.load(seed_path/'initial_actions.npy')
        init_obs = np.load(seed_path/'init_obs.npy')
        #rel_dur_plan = np.load(seed_path/'rel_dur_plan.npy')
    ###############

    test_observations_continue = p/Path('meta_observations.npy')
    if not test_observations_continue.exists():
        if 'meta_actions' not in locals():
            #rel_dur_plan = np.load(seed_path/'rel_dur_plan.npy')
            meta_actions = np.load(seed_path/'meta_actions.npy')

        if args.experiment == 'mass':
            meta_test_observations = planner.test_env_baselines(meta_actions, friction=test_id, mass=causal_ood)
        elif args.experiment == 'gravity':
            meta_test_observations = planner.test_env_baselines(meta_actions, mass=test_id,
                                                                gravity=causal_ood, gravity_axis=2)
        else: ## wind
            meta_test_observations = planner.test_env_baselines(meta_actions, mass=test_id,
                                                                gravity=causal_ood, gravity_axis=0)

        np.save(p/'meta_observations.npy', meta_test_observations)
        np.save(p/'meta_actions.npy', meta_actions)

    if args.baseline_only is True:
        return None, None

    if args.experiment == 'mass':
        sim_obs = planner.sim_mujoco(init_actions,
                                     friction=test_id, mass=causal_ood)
    elif args.experiment == 'gravity':
        sim_obs = planner.sim_mujoco(init_actions, mass=test_mass,
                                     gravity=causal_ood, gravity_axis=2)
    else:
        sim_obs = planner.sim_mujoco(init_actions, mass=test_mass,
                                     gravity=causal_ood, gravity_axis=0)

    init_preds, sim_preds = fit_and_predict(init_obs, sim_obs)

    initial_predictions = np.concatenate([init_preds, sim_preds])
    np.save(p/'initial_prediction.npy', initial_predictions)

    initial_observations = np.concatenate([init_obs, sim_obs])
    np.save(p/'initial_observations.npy', initial_observations)

    initial_reward = reward(initial_observations, initial_predictions)
    np.save(p/'initial_reward.npy', np.array((initial_reward)))

    belief_cluster_id = []
    cluster_ood = []
    for i, in_dist in enumerate(train_id):
        if initial_predictions[i] == initial_predictions[-1]:
            belief_cluster_id.append(in_dist)
            if args.experiment == 'mass':
                cluster_ood.append(1) # in distribution value for friction
            elif args.experiment == 'gravity':
                cluster_ood.append(-9.8)  # in distribution value for gravity
            else:
                cluster_ood.append(0.0)  # in distribution value for wind
    if len(belief_cluster_masses) == 1:
        belief_cluster_id.append(belief_cluster_masses[0])
        belief_cluster_id.append(test_id)
        belief_cluster_id.append(test_id)
        if args.experiment == 'mass':
            cluster_ood = [1, 1, causal_ood, causal_ood]
        elif args.experiment == 'gravity':
            cluster_ood = [-9.81, -9.81, causal_ood, causal_ood]
        else:
            cluster_ood = [0, 0, causal_ood, causal_ood]
    else:
        belief_cluster_masses.append(test_id)
        cluster_ood.append(causal_ood)

    assert len(cluster_ood) == len(belief_cluster_masses) and "need to be same len"

    if args.experiment == 'mass':
        cluster_actions, cluster_observations, _, _= planner.plan_mujoco(
                    action_mode=action_mode,
                    masses=cluster_ood,
                    friction_vals=belief_cluster_id
        )
    elif args.experiment == 'gravity':
        cluster_actions, cluster_observations, _, _ = planner.plan_mujoco(
                    action_mode=action_mode,
                    masses=belief_cluster_id,
                    gravity=cluster_ood,
                    gravity_axis=2
        )
    else:
        cluster_actions, cluster_observations, _, _ = planner.plan_mujoco(
                    action_mode=action_mode,
                    masses=belief_cluster_id,
                    gravity=cluster_ood,
                    gravity_axis=0
        )

    cluster_predictions = predict(cluster_observations)
    cluster_reward = reward(cluster_observations, cluster_predictions)
    print("Belief cluster predictions are: ", cluster_predictions)
    print("Cluster reward is: ", cluster_reward)

    #np.save(p/'meta_cluster_actions.npy', meta_cluster_actions)
    #np.save(p/'meta_cluster_observations.npy', meta_cluster_obs)
    np.save(p/'cluster_predictions.npy', cluster_predictions)
    np.save(p/'cluster_observations.npy', cluster_observations)
    np.save(p/'cluster_actions.npy', cluster_actions)
    np.save(p/'cluster_reward.npy', np.array((cluster_reward)))

    unique = np.unique(cluster_predictions[:-1])
    if len(unique) == 1 and unique[0] != cluster_predictions[-1]:
        np.save(p/'final_predict.npy', np.array([TRUE]))
        return TRUE, cluster_reward
    else:
        np.save(p/'final_predict.npy', np.array([FALSE]))
        return FALSE, cluster_reward

def print_args(masses, action_mode, test_mass, args):
    #params = "Training masses: {} ".format(masses)
    params = "t_mass: {}".format(test_mass)
    params += "(causal_factor, seed: {} ".format(args)
    return params

def main(args):

    output_dir = Path(args.output)

    if not output_dir.exists():
        output_dir.mkdir() 
    os.chdir(output_dir)

    train_id = [0.202, 0.403, 0.595, 0.804, 1.05, 1.202, 1.39, 1.602, 1.794, 2.05]
    test_id = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    test_ood = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    #if args.experiment == 'friction' and args.friction_id == 2:
        #test_ood = [1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8]
    if args.experiment == 'gravity':
        test_ood = [np.round(i * -9.81, decimals=3) for i in test_ood]
    elif args.experiment == 'wind':
        test_ood = [np.round(i * 9.81, decimals=3) for i in test_ood]

    #train_masses = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    #test_mass = [0.2, 0.6, 1.0, 1.4, 1.8, 2.0]
    #test_mass = [0,2]
    #test_ood = [0.2, 0.6, 1.0, 1.4, 1.8, 2.0]
    #if args.experiment == 'gravity' and args.gravity_axis == 2:
        #test_ood = [np.round(i * -9.81, decimals=3) for i in test_ood]
    #elif args.experiment == 'gravity' and args.gravity_axis == 0:
    #    test_ood = [np.round(i * 9.81, decimals=3) for i in test_ood]

    if args.baseline_only:
        seeds = [111, 123, 145, 156, 164]
    else:
        seeds = [14, 28, 33, 64, 101, 109, 144, 171, 181, 189]

    with futures.ProcessPoolExecutor() as executor:
        for m in test_id:
            results = executor.map(
                run_mujoco,
                repeat(train_id),
                repeat(m),
                repeat(args),
                product(test_ood, seeds)
            )

            for j, result in enumerate(results):
                print(f'result {j}: ', result)

    print("All finished!")

if __name__ == "__main__":
    from datetime import datetime
    from argparse import ArgumentParser

    begin_time = datetime.now()
    print("Start time: ", begin_time)

    parser = ArgumentParser()

    parser.add_argument('--action_mode', type=str, required=True,
                        choices=['hopper', 'cheetah', 'walker'],
                        help='Mujoco env name.')
    parser.add_argument('--experiment', type=str, required=True, 
                        choices=['mass', 'gravity', 'wind'],
                        help='What causal factor to change at test time?')
    #parser.add_argument('--gravity_axis', default=None, required=False,
                        #choices=[None, 0, 1, 2], type=int, help='gX: 0, gY: 1, gZ: 2')
    #parser.add_argument('--gravity_id', default=-9.8, type=float, required=False,
                        #help='In Distribution gravity value along the desired gravity axis')
    parser.add_argument('--output', type=str, required=True, help='output directory name')
    parser.add_argument('--baseline_only', type=bool, 
                        default=False, required=False, help='T/F do we optimize CC')
    #parser.add_argument('--friction_id', type=float, required=False)
    #parser.add_argument('--num_workers', type=int, default=8)


    args = parser.parse_args()

    #if args.experiment == 'friction':
    #    assert args.friction_id is not None

    #if args.experiment == 'gravity':
    #    assert args.gravity_axis is not None and 'Must specify!!'
    #print(args.gravity_id)

    main(args)

    total_time = datetime.now() - begin_time
    print(f"Total execution time: {total_time}")


