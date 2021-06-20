

# high-dimensional action space
import pybullet_utils.bullet_client as bc
import pybullet

import gym
import numpy as np
from loguru import logger

from cem_planning.cem.cem_planner import CEMPlanner


class CemSanityCheckEnv(gym.Env):
    def __init__(self):

        self.observation_space = gym.spaces.Box(-1, 1, shape=(1,))
        self.action_space = gym.spaces.Box(-1, 1, shape=(16,))
        self.bullet_client = bc.BulletClient(connection_mode=pybullet.DIRECT)

    def step(self, action):
        obs = self.get_obs()
        reward = -np.abs(np.asarray(action)).sum()
        done = False
        info = dict()
        return obs, reward, done, info

    def get_obs(self):
        obs = np.zeros(1, )
        return obs

    def reset(self):
        return self.get_obs()

    def render(self, mode='human'):
        pass


def main():
    for n_elite in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        rng = np.random.RandomState(1234)
        env = CemSanityCheckEnv()
        _, r, _, _ = env.step(rng.uniform(-1, 1, (16,)))
        logger.info(f'n_elite: {n_elite}')
        print(f'r: {r}')

        planner = CEMPlanner(n_plans=256, horizon=1, action_space=env.action_space,
                             n_iterations=256, n_elite=n_elite, cache_k=0,
                             warm_starts=False, warm_start_relaxation=None,
                             plan_action_repeat=1, action_transformation=None,
                             rng=rng,
                             viz_progress=True)
        action = planner.plan(env, lambda : None, lambda x: None)
        _, r, _, _ = env.step(action)
        print(f'r: {r}')


if __name__ == '__main__':
    main()
