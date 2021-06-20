import numpy as np

from cem.action_dist import ActionDist


class DiagonalTruncNorm(ActionDist):
    """
    # Not a proper truncated normal distribution - we clip at the boundary and
    # get high probability mass at the boundary values
    # However, the scipy version was slow both in sampling and fitting
    """

    def __init__(self, action_space):
        self.means = None
        self.stddevs = None
        self.a_low = action_space.low
        self.a_high = action_space.high

    def get_init_mean(self, ):
        return (self.a_low + self.a_high) / 2

    def get_init_stddev(self, ):
        epsilon = 1e-6
        return (self.a_high - self.a_low) / 2 + epsilon

    def sample_one_action_plan(self, rng):
        actions = rng.normal(self.means, self.stddevs)
        actions = np.clip(actions, self.a_low, self.a_high)
        return actions

    def sample(self, n_plans, rng):
        return self.sample_independently(n_plans, rng)

    def shift_t(self, delta_t, action_space):
        self.means[:-delta_t] = self.means[delta_t:]
        self.stddevs[:-delta_t] = self.stddevs[delta_t:]
        self.means[-delta_t:] = self.get_init_mean()[None, :]
        self.stddevs[-delta_t:] = self.get_init_stddev()[None, :]

    def __str__(self):
        return f'DiagonalTruncNorm(\nmeans={self.means},\nstddevs={self.stddevs})'

    def init_from_action_space(self, action_space, horizon):
        self.means = np.array(
            [self.get_init_mean() for _ in range(horizon)])
        self.stddevs = np.array(
            [self.get_init_stddev() for _ in range(horizon)])

    def fit_to(self, action_plans, duration_plans, action_space):
        # n = xs.shape[0]
        self.means = np.mean(action_plans, axis=0)
        self.stddevs = np.std(action_plans, axis=0)
