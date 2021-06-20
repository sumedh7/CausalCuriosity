import numpy as np

from cem.action_dist import ActionDist
from cem.util import fit_uniform_dist


class UniformBounds(ActionDist):
    """
    # when fitting the uniform, we use
    # lower_limit = max - (max - min) (n + 2)/n
    # upper_limit = min + (max - min) (n + 2)/n
    """

    def __init__(self, relax=0.0):
        self.lower = None
        self.upper = None
        self.horizon = None
        self.relax = relax

    def sample_one_action_plan(self, rng):
        return rng.uniform(self.lower, self.upper)

    def sample_one_duration_plan(self, rng):
        return np.ones(self.horizon) / self.horizon

    def sample(self, n_plans, rng):
        return self.sample_independently(n_plans, rng)

    def shift_t(self, delta_t, action_space):
        self.lower[:-delta_t] = self.lower[delta_t:]
        self.upper[:-delta_t] = self.upper[delta_t:]

        extents = self.upper - self.lower
        assert np.all(extents >= 0)
        assert self.relax >= 0
        self.lower -= extents * self.relax
        self.upper += extents * self.relax
        self.lower, self.upper = restrict_range(action_space, self.lower, self.upper)

        self.lower[-delta_t:] = action_space.low[np.newaxis, :]
        self.upper[-delta_t:] = action_space.high[np.newaxis, :]

    def __str__(self):
        return f'UniformBounds(\nlower={self.lower},\nupper={self.upper})'

    def init_from_action_space(self, action_space, horizon):
        self.lower = np.tile(action_space.low, (horizon, 1))
        self.upper = np.tile(action_space.high, (horizon, 1))
        self.horizon = horizon

    def fit_to(self, action_plans, duration_plans, action_space):
        lower, upper = fit_uniform_dist(action_plans)
        if action_space is not None:
            lower, upper = restrict_range(action_space, lower, upper)
        self.lower = lower
        self.upper = upper


def restrict_range(action_space, lower, upper):
    lower = np.maximum(lower, action_space.low[np.newaxis, :])
    upper = np.minimum(upper, action_space.high[np.newaxis, :])
    return lower, upper
