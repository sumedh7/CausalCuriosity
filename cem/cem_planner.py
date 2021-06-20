"""
Cross-entropy planner for pybullet-finger envs.
Uses Uniform distributions instead of Gaussians.
"""

from collections import defaultdict, Callable
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from loguru import logger
from tqdm import tqdm

from cem.diagonal_truncnorm import DiagonalTruncNorm
from cem.time_adaptive_uniform_bounds import TimeAdaptiveUniformBounds
from cem.uniform_bounds import UniformBounds


class CEMPlanner:
    def __init__(self, n_plans, horizon, action_space, sampler,
                 n_iterations, n_elite, cache_k,
                 warm_starts=False, warm_start_relaxation=0.0,
                 plan_action_repeat=1,
                 action_transformation=None,
                 rng=None,
                 viz_progress=False
                 ):

        # plan_action_repeat: independently of the environment frame-skip,
        # repeat actions in the plan this many times
        self.viz_progress = viz_progress
        self.plan_action_repeat = plan_action_repeat
        self.n_plans = n_plans
        self.horizon = horizon
        self.n_iterations = n_iterations
        self.n_elite = n_elite
        self.action_space = action_space
        self.action_transformation = action_transformation

        self.cache_k = cache_k
        self.plan_cache = defaultdict(list)
        self.warm_starts = warm_starts
        self.warm_start_relaxation = warm_start_relaxation
        self.sampler = sampler
        if sampler == 'uniform':
            self.action_dist = UniformBounds(relax=self.warm_start_relaxation)
        elif sampler == 'ta_uniform':
            self.action_dist = TimeAdaptiveUniformBounds(relax=self.warm_start_relaxation)
        elif sampler == 'diag_normal':
            assert not self.warm_starts
            assert self.warm_start_relaxation == 0.0
            self.action_dist = DiagonalTruncNorm(action_space=self.action_space)
        else:
            raise ValueError(f'Unknown sampler: {sampler}')

        self.action_dist.init_from_action_space(self.action_space, self.horizon)

        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng

    def plan(self, env, save_state, restore_state):
        """
        :return: The action that was determined to be the best
        """
        if not self.warm_starts:
            self.action_dist.init_from_action_space(self.action_space, self.horizon)
        else:
            assert self.warm_starts
            self.action_dist.shift_t(1, action_space=self.action_space)

        if len(self.plan_cache[env]) > 0:
            return self.plan_cache[env].pop(0)

        # initial_state = env.bullet_client.saveState()
        initial_state = save_state()

        best_action_plan = None
        best_return = -float('inf')
        best_reward_sequence = None

        _best_returns = []
        _worst_returns = []

        print(f'self.n_plans: {self.n_plans}')

        if self.viz_progress:
            maybe_progbar = partial(tqdm, desc='Planning')
        else:
            maybe_progbar = lambda x: x

        for i_iteration in maybe_progbar(range(self.n_iterations)):
            action_plans, rel_duration_plans = self.action_dist.sample(self.n_plans, self.rng)

            effective_horizon = self.horizon * self.plan_action_repeat
            discrete_duration_plans = np.round(rel_duration_plans * effective_horizon).astype(int)

            reward_sequences = []

            # For analysis only
            _reward_microsequences = []

            for i_plan in range(self.n_plans):
                restore_state(initial_state)
                reward_sequences.append([])
                _reward_microsequences.append([])

                plan_actions = action_plans[i_plan]
                plan_durations = discrete_duration_plans[i_plan]

                for i_step in range(self.horizon):
                    step_reward = 0.0
                    action = plan_actions[i_step]
                    action_repeat = plan_durations[i_step]
                    if self.action_transformation is not None:
                        action = self.action_transformation(action)

                    for i_repeat in range(action_repeat):
                        next_state, reward, done, _ = env.step(action)
                        step_reward += reward
                        _reward_microsequences[-1].append(reward)
                        if done:
                            break
                    reward_sequences[-1].append(step_reward)
                    if done:
                        padding_rewards = [0.] * (self.horizon - i_step - 1)
                        reward_sequences[-1].extend(padding_rewards)
                        break

            # Take elite samples
            plan_returns = np.sum(reward_sequences, axis=1)
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
                best_reward_sequence = reward_sequences[best_idx]
                self._best_reward_microsequence = _reward_microsequences[best_idx]

            # # Some plotting, to see if plans improve over iterations
            # _best_returns.append(np.max(plan_returns))
            # _worst_returns.append(np.min(plan_returns))
            # if (self.viz_progress and
            #         i_iteration % max(self.n_iterations // 6, 1) == 0
            #         # i_iteration == self.n_iterations - 1
            # ):
            #     plt.plot(_best_returns, marker='o')
            #     plt.plot(_worst_returns, marker='o')
            #     plt.show()
            #     plt.close(plt.gcf())
            #     plt.close('all')

        assert np.shape(reward_sequences) == (self.n_plans, self.horizon)

        logger.info(f'best_return: {best_return}')
        logger.info(f'best_reward_sequence: {best_reward_sequence}')
        logger.info(f'best_action_plan: {best_action_plan}')
        logger.info(f'best_rel_duration_plan: {best_rel_duration_plan}')
        best_plan_incl_repeats = [a for a in best_action_plan
                                  for _ in range(self.plan_action_repeat)]
        self.plan_cache[env] = best_plan_incl_repeats[1:self.cache_k * self.plan_action_repeat]
        restore_state(initial_state)
        return best_action_plan[0]


if __name__ == '__main__':
    # Some manual tests...

    rng = np.random.RandomState(1234)
    # low = np.array([0., -1., 10.], dtype=np.float32)
    # high = np.array([1., 2., 100.], dtype=np.float32)
    # bounds = UniformBounds.from_action_space(spaces.Box(low=low, high=high),
    #                                          horizon=2)
    # print(f'bounds.lower: {bounds.lower}')
    # print(f'bounds.upper: {bounds.upper}')
    #
    # data = np.random.uniform(0, 1, (5, 2, 3))
    # data[:, 0, 0] *= 10
    # data[:, 1, 2] = data[:, 1, 2] + 10
    #
    # print(f'np.min(data, axis=0): {np.min(data, axis=0)}')
    # print(f'np.max(data, axis=0): {np.max(data, axis=0)}')
    #
    # bounds = UniformBounds.fit_to(data)
    # print(f'bounds: {bounds}')

    # bounds = UniformBounds.from_action_space(action_space=spaces.Box(-1, 1, (2,)), horizon=1)
    #
    # for _ in range(30):
    #     actions = bounds.sample(n_plans=4, rng=rng)
    #     bounds = UniformBounds.fit_to(actions)
    #     print(f'bounds: {bounds}')
    #
    # # print(f'actions: {actions}')
