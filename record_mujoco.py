""" Visualization methods for causal curiosity experiments. """
import os
from pathlib import Path

import numpy as np
import gym
from gym.wrappers import Monitor
from gym.wrappers.monitoring.video_recorder import VideoRecorder

CLUSTER_ACTIONS = Path('cluster_actions.npy')
FINAL_PREDICT = Path('final_predict.npy')

def record_video(action_mode, action_plan, 
                 action_repeat=10, friction=None, mass=None,
                 gravity=None, gravity_axis=None,
                 file_name='recording.mp4'):
    
    env = gym.make(action_mode)
    env.reset()

    if mass:
        tmp = env.model.body_mass
        env.model.body_mass[:] = mass*tmp
    if friction:
        tmp = env.model.geom_friction
        env.model.geom_friction[:] = friction*tmp
    if gravity:
        assert gravity_axis is not None
        env.model.opt.gravity[gravity_axis] = gravity

    env.reset()
    recorder = VideoRecorder(env, file_name)
    recorder.capture_frame()

    for i, action in enumerate(action_plan):
        for i_repeat in range(action_repeat):
            #env.render()  # do not render, throws an error about GLEW initialization
            obs, _, _, _ = env.step(action)
            recorder.capture_frame()

    return

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--action_mode', type=str, required=True)
    parser.add_argument('--mass', type=float)
    parser.add_argument('--friction', type=float)
    parser.add_argument('--gravity', type=float)
    parser.add_argument('--gravity_axis', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--path', type=str)
    parser.add_argument('--file_name', type=str, default=None)
    args = parser.parse_args()



    if args.action_mode == 'walker':
        action_mode = 'Walker2d-v3'
    elif args.action_mode == 'cheetah':
        action_mode = 'HalfCheetah-v3'
    elif args.action_mode == 'hopper':
        action_mode = 'Hopper-v3'
    else:
        raise NotImplementedError

    path = Path(args.path)
    #seed = args.seed
    mass = args.mass
    friction = args.friction
    gravity = args.gravity
    axis = args.gravity_axis
    file_name = args.file_name

    #if args.file_name:
    #    pass
    #elif friction is not None:
    #    file_name = Path(f'recording_{action_mode}_friction{friction}_mass{mass}.mp4')
    #else:
    #    file_name = Path(f'recording_{action_mode}_mass{mass}_gravity{gravity}.mp4')

    decision = np.load(path/FINAL_PREDICT)
    if decision == 0:
        file_name = file_name + '-false.mp4'
    elif decision == 1:
        file_name = file_name + '-true.mp4'

    action_plan = np.load(path/'cluster_actions.npy')

    record_video(action_mode, action_plan,
                 mass=mass, friction=friction, 
                 gravity=gravity, gravity_axis=axis, 
                 file_name=file_name)

    print('finished recording')



