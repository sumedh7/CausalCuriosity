import numpy as np

from cem.action_dist import ActionDist
from cem.util import fit_uniform_dist


class TimeAdaptiveUniformBounds(ActionDist):
    """
    # when fitting the uniform, we use
    # lower_limit = max - (max - min) (n + 2)/n
    # upper_limit = min + (max - min) (n + 2)/n
    """

    def __init__(self, relax=0.0):
        self.a_lower = None
        self.a_upper = None
        self.t_lower = None
        self.t_upper = None
        self.horizon = None
        self.relax = relax

    def sample_one_action_plan(self, rng):
        return rng.uniform(self.a_lower, self.a_upper)

    def sample_one_duration_plan(self, rng):
        raw_times = rng.uniform(self.t_lower, self.t_upper)
        return raw_times / np.sum(raw_times)

    def sample(self, n_plans, rng):
        return self.sample_independently(n_plans, rng)

    def shift_t(self, delta_t, action_space):
        self.a_lower[:-delta_t] = self.a_lower[delta_t:]
        self.a_upper[:-delta_t] = self.a_upper[delta_t:]

        extents = self.a_upper - self.a_lower
        assert np.all(extents >= 0)
        assert self.relax >= 0
        self.a_lower -= extents * self.relax
        self.a_upper += extents * self.relax
        self.a_lower, self.a_upper = restrict_range(action_space, self.a_lower, self.a_upper)

        self.a_lower[-delta_t:] = action_space.low[np.newaxis, :]
        self.a_upper[-delta_t:] = action_space.high[np.newaxis, :]

        # TODO: time

    def __str__(self):
        return f'TimeAdaptiveUniformBounds(\nlower={self.a_lower},\nupper={self.a_upper})'

    def init_from_action_space(self, action_space, horizon):
        self.a_lower = np.tile(action_space.low, (horizon, 1))
        self.a_upper = np.tile(action_space.high, (horizon, 1))
        self.t_lower = np.zeros(horizon)
        self.t_upper = np.ones(horizon) / horizon
        self.horizon = horizon

    def fit_to(self, action_plans, duration_plans, action_space):
        assert action_plans.shape[1] == self.horizon
        assert duration_plans.shape[1] == self.horizon
        assert action_plans.shape[:-1] == duration_plans.shape

        lower, upper = fit_uniform_dist(action_plans)
        if action_space is not None:
            lower, upper = restrict_range(action_space, lower, upper)
        self.a_lower = lower
        self.a_upper = upper

        # TODO: time
        t_lower, t_upper = fit_uniform_dist(duration_plans)
        t_lower = np.maximum(t_lower, 0.0)

        # Scale t_upper such that it corresponds to two timesteps
        # (to prevent collapse of times)
        t_upper = np.maximum(t_upper, 2 / self.horizon)

        self.t_lower = t_lower
        self.t_upper = t_upper


def restrict_range(action_space, lower, upper):
    lower = np.maximum(lower, action_space.low[np.newaxis, :])
    upper = np.minimum(upper, action_space.high[np.newaxis, :])
    return lower, upper
