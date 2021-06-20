import gym
import numpy as np
# from pybullet_fingers.gym_wrapper import utils


def get_plan_action_space(action_mode):
    single_finger_space = gym.spaces.Box(-1, 1, shape=(3,))

    if action_mode == 'GB_sym':
        action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))

        def action_transformation(action):
            assert action.shape == (3,)
            red_finger_action = np.array([0., -0.8, 0.0])
            # red_finger_action = utils.scale(red_finger_action, single_finger_space)
            return np.concatenate((red_finger_action, action[:3], action[:3]))
    elif action_mode == 'GB_asym':
        action_space = gym.spaces.Box(low=-1, high=1, shape=(6,))

        def action_transformation(action):
            assert action.shape == (6,)
            red_finger_action = np.array([0., -0.8, 0.0])
            # red_finger_action = utils.scale(red_finger_action, single_finger_space)
            return np.concatenate((red_finger_action, action[:3], action[3:6]))
    elif action_mode == 'RGB_sym':
        action_space = gym.spaces.Box(low=-1, high=1, shape=(6,))

        def action_transformation(action):
            assert action.shape == (6,)
            return np.concatenate((action[0:3], action[3:6], action[3:6]))
    elif action_mode == 'RGB_asym':
        action_space = gym.spaces.Box(low=-1, high=1, shape=(9,))

        def action_transformation(action):
            assert action.shape == (9,)
            return action

    elif action_mode == 'cheetah':
        action_space = gym.spaces.Box(low=-1, high=1, shape=(6,))

        def action_transformation(action):
            assert action.shape == (6,)
            return action

    elif action_mode == 'humanoid':
        action_space = gym.spaces.Box(low=-0.4, high=0.4, shape=(17,))

        def action_transformation(action):
            assert action.shape == (17,)
            return action

    elif action_mode == 'ant':
        action_space = gym.spaces.Box(low=-1, high=1, shape=(8,))

        def action_transformation(action):
            assert action.shape == (8,)
            return action
    elif action_mode == 'walker':
        action_space = gym.spaces.Box(low=-1, high=1, shape=(6,))

        def action_transformation(action):
            assert action.shape == (6,)
            return action
    elif action_mode == 'hopper':
        action_space = gym.spaces.Box(low=-1, high=1, shape=(3,))

        def action_transformation(action):
            assert action.shape == (3,)
            return action

    else:
        raise ValueError()
    return action_space, action_transformation