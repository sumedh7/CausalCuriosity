

# high-dimensional action space
import pybullet_utils.bullet_client as bc
import pybullet

import gym
import numpy as np
from loguru import logger

from cem_planning.cem.cem_planner import CEMPlanner


class TimeAdaptiveSanityCheckEnv(gym.Env):
    def __init__(self):
        self.action_dim = 2
        self.first_part_n_steps = 4
        self.observation_space = gym.spaces.Box(-1, 1, shape=(1,))
        self.action_space = gym.spaces.Box(-1, 1, shape=(self.action_dim,))
        self.bullet_client = bc.BulletClient(connection_mode=pybullet.DIRECT)
        self.time_step = 0

    def step(self, action):
        obs = self.get_obs()
        action = np.asarray(action)

        if self.time_step < self.first_part_n_steps:
            reward = -np.abs(action - 0.5).sum()
        else:
            reward = -np.abs(action + 0.5).sum()

        done = False
        info = dict()
        self.time_step += 1

        assert self.time_step <= 12

        return obs, reward, done, info

    def get_obs(self):
        obs = np.zeros(1, )
        return obs

    def reset(self):
        self.time_step = 0
        return self.get_obs()

    def render(self, mode='human'):
        pass


def main():
    for n_elite in [4, 8]:
        rng = np.random.RandomState(1234)
        env = TimeAdaptiveSanityCheckEnv()
        _, r, _, _ = env.step(rng.uniform(-1, 1, (16,)))
        logger.info(f'n_elite: {n_elite}')
        print(f'r: {r}')

        # ta_uniform with horizon performs better than uniform
        horizon = 2
        plan_action_repeat = 6
        planner = CEMPlanner(n_plans=64, horizon=horizon,
                             action_space=env.action_space,
                             sampler='ta_uniform',
                             n_iterations=16, n_elite=n_elite, cache_k=0,
                             warm_starts=False, warm_start_relaxation=None,
                             plan_action_repeat=plan_action_repeat,
                             action_transformation=None,
                             rng=rng,
                             viz_progress=True)

        def save_state():
            return env.time_step

        def restore_state(state):
            env.time_step = state

        env.reset()
        action = planner.plan(env, save_state, restore_state)
        _, r, _, _ = env.step(action)
        print(f'r: {r}')


if __name__ == '__main__':
    main()