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
from itertools import count
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import gym
#from sklearn.cluster import KMeans
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
from loguru import logger
from tqdm import tqdm
import dill
from gym.wrappers.monitoring.video_recorder import VideoRecorder

# local imports
from cem_planner_vanilla_cw import CEMPlanner
from plan_action_spaces import get_plan_action_space
from cem.uniform_bounds import UniformBounds
from causal_world.task_generators.task import task_generator
from causal_world.envs.causalworld import CausalWorld


class ConfusionPlanner(CEMPlanner):
    """
    Extension of CEMPlanner class.
    New plan function that allows us to specify which env to train in.
    Returns: best-trained action plan, observations, trained KMeans model
    """
    def plan_environment(self, causal_envs, frame_skip=1):
        if self.warm_starts:
            assert self.warm_starts
            self.action_dist.shift_t(1, action_space=self.action_space)
        else:
            print('Initializing planner from action space')
            self.action_dist.init_from_action_space(self.action_space,
                                                    self.horizon)

        ## we can use this to add additional params we might want to tweak in the future
        ## other option is super(ConfusionPlanner, self).__init__(...), but this may be simplier
        fskip_list = False
        if type(frame_skip) is list:
            fskip_list = True
            assert len(causal_envs.envs) == len(frame_skip)
            #print("nice ")

        best_action_plan = None
        best_return = -float('inf')
        best_reward_sequence = None
        best_observations = None

        envs = []

        #print(f'n_plans: {self.n_plans}')

        if self.viz_progress:
            prog_bar = partial(tqdm, desc='Planning')
        else:
            prog_bar = lambda x: x

        meta_observations, meta_actions = [], []
        for i in prog_bar(range(self.n_iterations)):
            action_plans, rel_duration_plans = self.action_dist.sample(self.n_plans, self.rng)
            effective_horizon = self.horizon * self.plan_action_repeat
            discrete_duration_plans = np.round(rel_duration_plans * effective_horizon).astype(int)

            reward_seq = []
            reward_cluster = []
            all_observations = []

            for i_plan in range(self.n_plans):

                plan_actions = action_plans[i_plan]
                plan_durations = discrete_duration_plans[i_plan]
                observations = np.zeros((len(causal_envs.envs),199,56))


                for i_causal_env, causal_env in enumerate(causal_envs.envs):
                    mass = causal_env[0]
                    size = causal_env[1]
                    shape = causal_env[2]
                    if i == 0 and i_plan == 0:
                        task = task_generator(task_generator_id ='pushing',
                                              tool_block_mass = mass,
                                              tool_block_shape = shape,
                                              tool_block_size = size)

                        if not fskip_list:
                            curr_env = CausalWorld(task=task,
                                                   skip_frame = frame_skip,
                                                   enable_visualization = False)
                        else:
                            curr_env = CausalWorld(task=task,skip_frame=frame_skip[i_causal_env],
                                                    enable_visualization=False)

                        # creates 30 envs during first iteration of i_plan and i
                        envs.append(curr_env)

                    else:
                        curr_env = envs[i_causal_env]

                    initial_obs = curr_env.reset()
                    print(initial_obs.shape)
                    observations[i_causal_env, 0, :] = initial_obs
                    #return

                    counter = count(1)
                    for i_step in range(self.horizon):
                        #step_reward = float(0)
                        action = plan_actions[i_step]
                        action_repeat = plan_durations[i_step]
                        #print("action repeat", action_repeat)

                        if self.action_transformation is not None:
                            action = self.action_transformation(action)

                        for i_repeat in range(action_repeat):
                            next_state, reward, _, _ = curr_env.step(action)
                            #print("next state", next_state.shape)
                            observations[i_causal_env,next(counter),:] = next_state

                y = TimeSeriesKMeans(
                    n_clusters=2,
                    metric="softdtw",
                    max_iter=100,
                    max_iter_barycenter=5,
                    metric_params={"gamma": .5},
                    random_state=0
                    ).fit_predict(observations)
                all_observations.append(observations)

                if len(np.unique(y)) == 1:
                    distance = -0.99
                else:
                    #print(len(y))
                    distance = silhouette_score(observations, y, metric='dtw')

                reward_cluster.append(distance)
                #print('predictions: ', km_sdtw.predict(observations))

            #print('reward_cluster: ', reward_cluster)
            # take elite samples
            meta_actions.append(action_plans)
            meta_observations.append(all_observations)
            plan_returns = np.array(reward_cluster)
            elite_idxs = np.argsort(-plan_returns)[:self.n_elite]
            elite_action_plans = action_plans[elite_idxs, :, :]
            elite_duration_plans = rel_duration_plans[elite_idxs, :]
            self.action_dist.fit_to(
                elite_action_plans, elite_duration_plans,
                action_space=self.action_space)

            if np.max(plan_returns) > best_return:
                best_return = np.max(plan_returns)
                best_idx = np.argmax(plan_returns)
                best_action_plan = action_plans[best_idx]
                best_rel_duration_plan = rel_duration_plans[best_idx]
                best_observations = all_observations[best_idx]

        print(f'best_return: {best_return}')
        print(f'best_reward_sequence: {best_reward_sequence}')
        print(f'best_action_plan: {best_action_plan}')
        print(f'best_rel_duration_plan: {best_rel_duration_plan}')

        return best_action_plan, best_observations, best_rel_duration_plan, meta_observations, meta_actions


    def do_simulation(self, causal_envs, plan_actions, rel_duration_plans, frame_skip=1):
        """
        Do simulation on ONE causal environment
        Args: len(causal_envs.envs) == 1
        Return: observation after applying the plan actions
        """

        ## set optional keyword args, currently only want frame skip
        #frame_skip = kwargs.get('frame_skip') if 'frame_skip' in kwargs else 1

        # pick a random element from env
        mass = causal_envs.envs[0][0]
        size = causal_envs.envs[0][1]
        shape = causal_envs.envs[0][2]

        effective_horizon = self.horizon * self.plan_action_repeat
        plan_durations = np.round(rel_duration_plans * effective_horizon).astype(int)

        task = task_generator(task_generator_id = 'pushing', tool_block_mass = mass,
                              tool_block_size = size, tool_block_shape = shape)
        curr_env = CausalWorld(task=task, skip_frame=frame_skip, enable_visualization=False)

        observation = np.zeros((1, 199, 56)) #1: one block, 198: frames/ep.
        init_obs = curr_env.reset()

        observations[0,0,:] = init_obs
        counter = count(1)
        for i_step in range(self.horizon):
            action = plan_actions[i_step]
            action_repeat = plan_durations[i_step]
            if self.action_transformation is not None:
                action = self.action_transformation(action)
                for i_repeat in range(action_repeat):
                    next_state, reward, _, _ = curr_env.step(action)
                    observation[0,next(counter),:] = next_state

        return observation

    def record_video(self, causal_envs, action_plan, rel_duration_plans, frame_skip=1, file_name='test_vid.mp4'):

        # going back and forth between if we should pass a list or a CEnv object...
        mass = causal_envs[0]
        size = causal_envs[1]
        shape = causal_envs[2]

        effective_horizon = self.horizon * self.plan_action_repeat
        plan_durations = np.round(rel_duration_plans * effective_horizon).astype(int)

        task = task_generator(task_generator_id = 'lifting', tool_block_mass = mass,
                              tool_block_size = size, tool_block_shape = shape)
        curr_env = CausalWorld(task=task, skip_frame=frame_skip, enable_visualization=False)
        curr_env.reset()

        recorder = VideoRecorder(curr_env, file_name)
        recorder.capture_frame()

        for i, action in enumerate(action_plan):
            action_repeat = plan_durations[i]
            for i_repeat in range(action_repeat):
                next_state, reward, _, _ = curr_env.step(action)
                recorder.capture_frame()

        recorder.close()
        print("FINISHED RECORDING\n")

        return

    def plan_mujoco(self, masses=None, action_mode=None, friction_vals=None, gravity=None, gravity_axis=None):
        '''plan and record video for mujoco'''
        assert masses is not None and "not optional!"
        if gravity:
            assert isinstance(gravity, list) and "sizes must be an iterable list for now"
            assert gravity_axis is not None and "Be sure to specify this!!!"
            assert len(gravity) == len(masses)
        assert isinstance(masses, list) and "sizes must be an iterable list for now"
        if friction_vals is not None:
            assert isinstance(friction_vals, list) and "all or nothing.."
            assert len(masses) == len(friction_vals) and "need 2 b same len"
        if action_mode is not None:
            self.action_mode = action_mode

        assert not self.warm_starts and "add this in later if we need it"
        self.action_dist.init_from_action_space(
            self.action_space,
            self.horizon
        )
        ## save best returns
        best_action_plan = None
        best_return = -float('inf')
        best_reward_sequence = None
        best_observations = None

        envs = []
        #prog_bar = lambda x: x
        #if self.viz_progress:
        prog_bar = partial(tqdm, desc='Planning')

        meta_observations, meta_actions = [], []
        #print('n iterations: ', self.n_iterations)
        for i in prog_bar(range(self.n_iterations)):
            ## ACTION PLANS
            action_plans, rel_duration_plans = self.action_dist.sample(self.n_plans, self.rng)
            effective_horizon = self.horizon * self.plan_action_repeat
            discrete_duration_plans = np.round(rel_duration_plans * effective_horizon).astype(int)
            #print("discrete_duration_plans", discrete_duration_plans)
            #print('action_plans: ', action_plans)

            all_observations = []
            reward_cluster = []

            print('n_plans:', self.n_plans)

            for i_plan in range(self.n_plans):
                plan_actions = action_plans[i_plan]
                #print(plan_actions) --> (6, 3)

                plan_durations = discrete_duration_plans[i_plan]
                if self.action_mode=="cheetah":
                    observations = np.zeros((len(masses), self.n_frames, 17))
                elif self.action_mode=="humanoid":
                    observations = np.zeros((len(masses), self.n_frames, 376))
                elif self.action_mode=="ant":
                    observations = np.zeros((len(masses), self.n_frames, 111))
                elif self.action_mode=="walker":
                    observations = np.zeros((len(masses), self.n_frames, 17))
                elif self.action_mode=="hopper":
                    observations = np.zeros((len(masses), self.n_frames, 11))
                else:
                    raise Exception("Invalid action_mode")

                for i_mass, mass in enumerate(masses):
                    if i == 0 and i_plan == 0:
                        if self.action_mode=="cheetah":
                            curr_env = gym.make('HalfCheetah-v3')
                        elif self.action_mode=="humanoid":
                            curr_env = gym.make('Humanoid-v3')
                        elif self.action_mode=="ant":
                            curr_env = gym.make('Ant-v3')
                        elif self.action_mode=="walker":
                            curr_env = gym.make('Walker2d-v3')
                        elif self.action_mode=="hopper":
                            curr_env = gym.make('Hopper-v3')
                        else:
                            raise Exception("Invalid action_mode")
                        ## set body mass
                        # mass attr is read only by default, have to copy over in two steps
                        tmp = curr_env.model.body_mass
                        curr_env.model.body_mass[:] = masses[i_mass] * tmp
                        if friction_vals is not None:
                            tmp = curr_env.model.geom_friction
                            curr_env.model.geom_friction[:] = friction_vals[i_mass]*tmp
                        if gravity:
                            ###################
                            ## gravity axis 0: X, 1: Y, 2: Z
                            ###################
                            tmp = curr_env.model.opt.gravity[gravity_axis] 
                            curr_env.model.opt.gravity[gravity_axis] = gravity[i_mass]
                            print(curr_env.model.opt.gravity[:])
                        envs.append(curr_env)
                    else:
                        curr_env= envs[i_mass]

                    init_state = curr_env.reset()

                    counter = 0
                    for i_step in range(self.horizon):
                        action = plan_actions[i_step]
                        action_repeat = plan_durations[i_step]
                        if self.action_transformation is not None:
                            action = self.action_transformation(action)
                        for i_repeat in range(action_repeat):
                            next_state, reward, _, _ = curr_env.step(action)
                            #print('next_state: ', next_state)
                            observations[i_mass,counter,:] = next_state
                            counter += 1

                all_observations.append(observations)

                y = TimeSeriesKMeans(
                        n_clusters=2,
                        metric="softdtw",
                        max_iter=100,
                        max_iter_barycenter=5,
                        metric_params={"gamma": .5},
                        random_state=0
                    ).fit_predict(observations)

                if len(np.unique(y)) == 1:
                    distance = -0.99
                else:
                    #print(len(y))
                    distance = silhouette_score(observations, y, metric='dtw')

                reward_cluster.append(distance)
                #print('predictions: ', km_sdtw.predict(observations))

            meta_observations.append(all_observations)
            meta_actions.append(action_plans)
            #print('reward_cluster: ', reward_cluster)
            # take elite samples
            plan_returns = np.array(reward_cluster)
            elite_idxs = np.argsort(-plan_returns)[:self.n_elite]
            elite_action_plans = action_plans[elite_idxs, :, :]
            elite_duration_plans = rel_duration_plans[elite_idxs, :]
            self.action_dist.fit_to(
                elite_action_plans, elite_duration_plans,
                action_space=self.action_space
            )

            if np.max(plan_returns) > best_return:
                best_return = np.max(plan_returns)
                best_idx = np.argmax(plan_returns)
                best_action_plan = action_plans[best_idx]
                best_rel_duration_plan = rel_duration_plans[best_idx]
                best_observations = all_observations[best_idx]

        print(f'best_return: {best_return}')
        print(f'best_reward_sequence: {best_reward_sequence}')
        #print(f'best_action_plan: {best_action_plan}')
        print(f'best_rel_duration_plan: {best_rel_duration_plan}')

        return best_action_plan, best_observations, best_rel_duration_plan, meta_observations, meta_actions

    def sim_mujoco(self, action_plan, rel_dur_plan, mass=None, friction=None, gravity=None, gravity_axis=None):
        if self.action_mode=="cheetah":
            observations = np.zeros((1, self.n_frames, 17))
            curr_env = gym.make('HalfCheetah-v3')
        elif self.action_mode=="humanoid":
            observations = np.zeros((1, self.n_frames, 376))
            curr_env = gym.make('Humanoid-v3')
        elif self.action_mode=="ant":
            observations = np.zeros((1, self.n_frames, 111))
            curr_env = gym.make('Ant-v3')
        elif self.action_mode=="walker":
            observations = np.zeros((1, self.n_frames, 17))
            curr_env = gym.make('Walker2d-v3')
        elif self.action_mode=="hopper":
            observations = np.zeros((1, self.n_frames, 11))
            curr_env = gym.make('Hopper-v3')
        else:
            raise Exception("Invalid action_mode")

        initial_state = curr_env.reset()

        if gravity:
            assert gravity_axis is not None and 'Dont forget to specify'
            tmp = curr_env.model.opt.gravity[gravity_axis]
            curr_env.model.opt.gravity[gravity_axis] = gravity

        if mass is not None:
            tmp = curr_env.model.body_mass
            curr_env.model.body_mass[:] = mass*tmp
        if friction is not None:
            tmp = curr_env.model.geom_friction
            curr_env.model.geom_friction[:] = friction*tmp

        initial_state = curr_env.reset()

        effective_horizon = self.horizon * self.plan_action_repeat
        plan_durations = np.round(rel_dur_plan * effective_horizon).astype(int)

        counter = count(0)
        for i_step in range(self.horizon):
            action = action_plan[i_step]
            action_repeat = plan_durations[i_step]
            if self.action_transformation is not None:
                action = self.action_transformation(action)
                for i_repeat in range(action_repeat):
                    next_state, reward, _, _ = curr_env.step(action)
                    observations[0, next(counter), :] = next_state

        return observations


class CausalEnvironments:
    """
    Training evirnments for planner.
    Combine a single env variables into a length 3 list
    """
    def __init__(self, masses, sizes, shapes, frame_skip=1):
        self.masses = masses
        self.sizes = sizes
        self.shapes = shapes
        self.frame_skip = frame_skip
        self.envs = []
        for mass in masses:
            for size in sizes:
                for shape in shapes:
                    self.envs.append([mass,size,shape])

    def appendEnv(self, env):
        self.envs.append(env)

    def appendEnv_copy(self, env_obj):
        assert isinstance(env_obj, CausalEnvironments)
        new = deepcopy(self)
        new.envs.extend(env_obj.envs)
        return new

    def __str__(self):
        return f'CEnv({self.masses}, {self.sizes}, {self.shapes}, fs={self.frame_skip})'


def vanilla_planner(seed=1235, n_frames=198, action_mode='RGB_asym'):
    ''' wrapper method to return untrained planner with vanilla params '''
    # Params. Play around with these settings. They are not yet optimized.
    #scenario = 'lift'
    n_frames_per_episode = n_frames; total_budget = 400; plan_horizon = 6
    n_plan_iterations = 20; viz_progress = True; n_plan_iterations = 20
    sampler = 'uniform'; warm_starts = False; warm_start_relaxation = 0.0
    elite_fraction = 0.1;

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

    planner.action_mode = action_mode
    planner.n_frames = n_frames

    return planner


def main(output_dir):
    """
    Do one simulation on causal_env, using plan_cations and plan_durations
    Return: the observation
    """
    ########################
    # Set to false if you want to use the pickled obj files for debuging
    retrain_planner_1 = False # 10 vary-mass envs
    retrain_planner_2 = False # vary-mass envs in the same cluster as test-size env
    retrain_planner_3 = False # 10 vary-size envs
    retrain_planner_4 = False # vary-size envs in the same cluster as test-shape env

    # change into output directory
    if not output_dir.exists():
        output_dir.chdir()
    os.chdir(output_dir)
    ########################

    # Params. Play around with these settings. They are not yet optimized.
    #scenario = 'lift'
    n_frames_per_episode = 198
    total_budget = 400
    plan_horizon = 6
    n_plan_iterations = 20
    frame_skip = 1
    action_mode = 'RGB_asym'
    sampler = 'uniform'
    warm_starts = False
    warm_start_relaxation = 0.0
    elite_fraction = 0.1
    viz_progress = True

    plan_action_repeat = np.floor_divide(n_frames_per_episode, plan_horizon)
    n_plan_cache_k = plan_horizon
    n_plans = np.floor_divide(total_budget * n_plan_cache_k,
                              plan_horizon * n_plan_iterations)
    n_plan_elite = max(1, int(round(elite_fraction * n_plans)))

    if n_plans <= n_plan_elite:
        n_plan_elite = n_plans - 1

    seed = 1235
    logger.info(f'seed: {seed}')
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    # action_space is gym `Box` env that defines vals for each action [-1,1]
    # action_transf is a func that returns an array for real-val actions?
    action_space, action_transformation = get_plan_action_space(action_mode)

    #if retrain_planner:
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

    episode_actions = []
    real_rewards = []
    frames = []

    ## train the planner on env with 10 varied mass, same shape, size
    masses = [0.4, 0.43, 0.46, 0.49, 0.52, 0.55, 0.58, 0.61, 0.64, 0.68]

    vary_mass_envs = CausalEnvironments(masses, [0.05], ['cube'])
    test_size_envs = CausalEnvironments([0.4], [0.04], ['cube'])

    if retrain_planner_1:
        action_plan_1, rel_duration_plan_1, observations_1, km_sdtw_1, best_return_1 = planner.plan_environment(vary_mass_envs)
        dill.dump(action_plan_1, open('action_plan_1.pickle', 'wb'))
        dill.dump(rel_duration_plan_1, open('rel_duration_plan_1.pickle', 'wb'))
        dill.dump(observations_1, open('observations_1.pickle', 'wb'))
        dill.dump(km_sdtw_1, open('km_model_1.pickle', 'wb'))
        dill.dump(best_return_1, open('best_return_1.pickle', 'wb'))
    else:
        with open('action_plan_1.pickle','rb') as f:
            action_plan_1 = dill.load(f)
        with open('rel_duration_plan_1.pickle','rb') as f:
            rel_duration_plan_1 = dill.load(f)
        with open('observations_1.pickle','rb') as f:
            observations_1 = dill.load(f)
        with open('km_model_1.pickle','rb') as f:
            km_sdtw_1 = dill.load(f)
        with open('best_return_1.pickle','rb') as f:
            best_return_1 = dill.load(f)

    # run one simulation on test env
    test_observations_1 = planner.do_simulation(test_size_envs, action_plan_1, rel_duration_plan_1)

    # predict labels for train and test
    train_predict_1 = km_sdtw_1.predict(observations_1)
    test_predict_1 = km_sdtw_1.predict(test_observations_1)

    # combine observations and labels
    combined_faulty_observations = np.append(observations_1,test_observations_1, axis = 0)
    y = np.append(train_predict_1, test_predict_1, axis = 0)

    # calculate faulty reward
    faulty_reward = silhouette_score(combined_faulty_observations, y, metric='dtw')


    print('\n------------------------------planner 1 (10 vary-mass envs)------------------------------\n')

    print('training:')
    print('best return: ', best_return_1)
    print('predictions of training observations: ', train_predict_1)

    print('\ntesting:')
    print('predictions of test observations: ', test_predict_1)
    print('faulty reward: ', faulty_reward)


    ## train the planner on belief cluster + test_env
    belief_mass_cluster = [masses[i] for i, label in enumerate(train_predict_1) if label == test_predict_1[0]]

    belief_and_test_envs = CausalEnvironments(belief_mass_cluster, [0.05], ['cube'])
    belief_and_test_envs.appendEnv([4, 0.04, 'cube'])

    if retrain_planner_2:
        action_plan_2, rel_duration_plan_2, observations_2, km_sdtw_2, best_return_2 = planner.plan_environment(belief_and_test_envs)
        dill.dump(action_plan_2, open('action_plan_2.pickle', 'wb'))
        dill.dump(rel_duration_plan_2, open('rel_duration_plan_2.pickle', 'wb'))
        dill.dump(observations_2, open('observations_2.pickle', 'wb'))
        dill.dump(km_sdtw_2, open('km_model_2.pickle', 'wb'))
        dill.dump(best_return_2, open('best_return_2.pickle', 'wb'))

    else:
        with open('action_plan_2.pickle','rb') as f:
            action_plan_2 = dill.load(f)
        with open('rel_duration_plan_2.pickle','rb') as f:
            rel_duration_plan_2 = dill.load(f)
        with open('observations_2.pickle','rb') as f:
            observations_2 = dill.load(f)
        with open('km_model_2.pickle','rb') as f:
            km_sdtw_2 = dill.load(f)
        with open('best_return_2.pickle','rb') as f:
            best_return_2 = dill.load(f)

    # predict labels for train and test
    train_predict_2 = km_sdtw_2.predict(observations_2)
    print('\n------------------------------planner 2 (vary-mass belief cluster + test-size env)------------------------------\n')
    print('belief_mass_cluster: ', belief_mass_cluster)
    print('best_return: ', best_return_2)
    print('predictions of training observations: ', train_predict_2)


    ## train the planner on env with 10 varied size, same shape, mass
    sizes = [0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13]
    vary_size_envs = CausalEnvironments([0.5], sizes, ['cube'])
    test_shape_envs = CausalEnvironments([0.5], [0.04], ['sphere'])

    if retrain_planner_3:
        action_plan_3, rel_duration_plan_3, observations_3, km_sdtw_3, best_return_3 = planner.plan_environment(vary_size_envs)
        dill.dump(action_plan_3, open('action_plan_3.pickle', 'wb'))
        dill.dump(rel_duration_plan_3, open('rel_duration_plan_3.pickle', 'wb'))
        dill.dump(observations_3, open('observations_3.pickle', 'wb'))
        dill.dump(km_sdtw_3, open('.km_model_3.pickle', 'wb'))
        dill.dump(best_return_3, open('best_return_3.pickle', 'wb'))
    else:
        with open('action_plan_3.pickle','rb') as f:
            action_plan_3 = dill.load(f)
        with open('rel_duration_plan_3.pickle','rb') as f:
            rel_duration_plan_3 = dill.load(f)
        with open('observations_3.pickle','rb') as f:
            observations_3 = dill.load(f)
        with open('km_model_3.pickle','rb') as f:
            km_sdtw_3 = dill.load(f)
        with open('best_return_3.pickle','rb') as f:
            best_return_3 = dill.load(f)

    # run one simulation on test env
    test_observations_3 = planner.do_simulation(
        test_shape_envs, action_plan_3, rel_duration_plan_3
    )

    # predict labels for train and test
    train_predict_3 = km_sdtw_3.predict(observations_3)
    test_predict_3 = km_sdtw_3.predict(test_observations_3)


    combined_faulty_observations = np.append(observations_3,test_observations_3, axis = 0)
    y = np.append(train_predict_3, test_predict_3, axis = 0)

    faulty_reward = silhouette_score(combined_faulty_observations, y, metric='dtw')

    print('\n------------------------------planner 3 (10 vary-size envs)------------------------------\n')
    print('training:')
    print('best return: ', best_return_3)
    print('predictions of training observations: ', train_predict_3)

    print('\ntesting:')
    print('predictions of test observations: ', test_predict_3)
    print('faulty reward: ', faulty_reward)


    ## train the planner on belief cluster + test_env
    belief_size_cluster = [sizes[i] for i, label in enumerate(train_predict_3) if label == test_predict_3[0]]

    belief_and_test_envs = CausalEnvironments([0.5], belief_size_cluster, ['cube'])
    belief_and_test_envs.appendEnv([0.5, 0.04, 'sphere'])

    if retrain_planner_4:
        action_plan_4, rel_duration_plan_4, observations_4, km_sdtw_4, best_return_4 = planner.plan_environment(belief_and_test_envs)
        dill.dump(action_plan_4, open('action_plan_4.pickle', 'wb'))
        dill.dump(rel_duration_plan_4, open('rel_duration_plan_4.pickle', 'wb'))
        dill.dump(observations_4, open('observations_4.pickle', 'wb'))
        dill.dump(km_sdtw_4, open('km_model_4.pickle', 'wb'))
        dill.dump(best_return_4, open('best_return_4.pickle', 'wb'))

    else:
        with open('action_plan_4.pickle','rb') as f:
            action_plan_4 = dill.load(f)
        with open('rel_duration_plan_4.pickle','rb') as f:
            rel_duration_plan_4 = dill.load(f)
        with open('observations_4.pickle','rb') as f:
            observations_4 = dill.load(f)
        with open('km_model_4.pickle','rb') as f:
            km_sdtw_4 = dill.load(f)
        with open('best_return_4.pickle','rb') as f:
            best_return_4 = dill.load(f)

    # predict labels for train and test
    train_predict_4 = km_sdtw_4.predict(observations_4)
    print('\n------------------------------planner 4 (vary-size belief cluster + test-shape env)------------------------------\n')
    print('belief_size_cluster: ', belief_size_cluster)
    print('best_return: ', best_return_4)
    print('predictions of training observations: ', train_predict_4)



if __name__ == '__main__':
    print("Causal Confusion CEM Planner")

    output_dir = Path('./pickle')
    print(f"Output dir: {output_dir}")
    print("Make sure this directory is what you expected, if not change it!")

    main(output_dir=output_dir)



