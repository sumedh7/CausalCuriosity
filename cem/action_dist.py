import numpy as np


class ActionDist:
    def sample_one_action_plan(self, rng):
        raise NotImplementedError()

    def sample_one_duration_plan(self, rng):
        raise NotImplementedError

    def sample(self, n_plans, rng):
        raise NotImplementedError()

    def shift_t(self, delta_t, action_space):
        raise NotImplementedError()

    def init_from_action_space(self, action_space, horizon):
        raise NotImplementedError()

    def fit_to(self, action_plans, duration_plans, action_space):
        raise NotImplementedError()

    def sample_independently(self, n_plans, rng):
        action_plans = np.array([self.sample_one_action_plan(rng) for _ in range(n_plans)])
        duration_plans = np.array([self.sample_one_duration_plan(rng) for _ in range(n_plans)])
        return action_plans, duration_plans
