"""
Cross-entropy planner 
Uses Uniform distributions instead of Gaussians.
"""

from frameskip_wrapper import FrameSkip
from scipy.spatial.distance import euclidean

from tslearn.generators import random_walks
from tslearn.clustering import TimeSeriesKMeans, silhouette_score
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
from tslearn.metrics import cdist_dtw
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics

class CEMPlanner:

    def __init__(self, n_plans, horizon, action_space, sampler,
                 n_iterations, n_elite, cache_k, obs_dim,
                 warm_starts=False, warm_start_relaxation=0.0,
                 plan_action_repeat=1,
                 action_transformation=None,
                 rng=None,
                 viz_progress=False,
                 frame_skip=1,
                 n_frames = 198,
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
        self.rng = rng
        self.frame_skip = frame_skip
        self.n_frames = n_frames
        self.obs_dim = obs_dim

    
    def plan(self, envs):
        """
        :return: The action that was determined to be the best
        """
        if self.warm_starts:
            assert self.warm_starts
            self.action_dist.shift_t(1, action_space=self.action_space)
        else:
            print('init planner from action space')
            self.action_dist.init_from_action_space(self.action_space, 
                                                    self.horizon)

        ## we can use this to add additional params we might want to tweak in the future
        ## other option is super(ConfusionPlanner, self).__init__(...), but this may be simplier

        best_action_plan = None
        best_return = -float('inf')
        best_reward_sequence = None

        print(f'n_plans: {self.n_plans}')

        if self.viz_progress:
            prog_bar = partial(tqdm, desc='Planning')
        else:
            prog_bar = lambda x: x

        for i in prog_bar(range(self.n_iterations)):
            action_plans, rel_duration_plans = self.action_dist.sample(self.n_plans, self.rng)
            effective_horizon = self.horizon * self.plan_action_repeat
            discrete_duration_plans = np.round(rel_duration_plans * effective_horizon).astype(int)

            reward_seq = []
            reward_cluster = []

            for i_plan in range(self.n_plans):

                plan_actions = action_plans[i_plan]
                plan_durations = discrete_duration_plans[i_plan]
                observations = np.zeros((len(envs),self.n_frames,self.obs_dim))


                for i_env, env in enumerate(envs):

                    env.reset()

                    count = 0
                    for i_step in range(self.horizon):

                        action = plan_actions[i_step]
                        action_repeat = plan_durations[i_step]

                        if self.action_transformation is not None:
                            action = self.action_transformation(action)

                        for i_repeat in range(action_repeat):
                            next_state, reward, _, _ = env.step(action)
                            observations[i_env,count,:] = next_state
                            count += 1

                km_sdtw = TimeSeriesKMeans(
                    n_clusters=2, 
                    metric="softdtw", 
                    max_iter=100,
                    max_iter_barycenter=5,
                    metric_params={"gamma": .5},
                    random_state=0
                    ).fit(observations)

                y = km_sdtw.predict(observations)

                if len(np.unique(y)) == 1:
                    distance = -0.99
                else:
                    distance = silhouette_score(observations, y, metric='dtw')

                reward_cluster.append(distance)

            # take elite samples
            plan_returns = np.array(reward_cluster)
            #print('plan_returns shape: ', plan_returns.shape)
            elite_idxs = np.argsort(-plan_returns)[:self.n_elite]
            elite_action_plans = action_plans[elite_idxs, :, :]
            elite_duration_plans = rel_duration_plans[elite_idxs, :]
            self.action_dist.fit_to(
                elite_action_plans, elite_duration_plans,
                action_space=self.action_space)
            #print('action_dist.lower:', self.action_dist.lower)
            #print('action_dist.upper:', self.action_dist.upper)
            # np.save('./lower_MassShapeSize_cw.npy', self.action_dist.lower)
            # np.save('./upper_mass_cw.npy', self.action_dist.upper)

            if np.max(plan_returns) > best_return:
                best_return = np.max(plan_returns)
                best_idx = np.argmax(plan_returns)
                best_action_plan = action_plans[best_idx]
                best_rel_duration_plan = rel_duration_plans[best_idx]

        logger.info(f'best_return: {best_return}')
        logger.info(f'best_reward_sequence: {best_reward_sequence}')
        logger.info(f'best_action_plan: {best_action_plan}')
        logger.info(f'best_rel_duration_plan: {best_rel_duration_plan}')

        #best_plan_incl_repeats = [a for a in best_action_plan
        #                          for _ in range(self.plan_action_repeat)]

        return best_action_plan, best_rel_duration_plan, observations, km_sdtw, best_return

    def simulate(self, env, plan_actions, rel_duration_plans):
        """
        Return: observation after applying the plan actions
        """   
        effective_horizon = self.horizon * self.plan_action_repeat
        plan_durations = np.round(rel_duration_plans * effective_horizon).astype(int)
        
        env.reset()
        observation = np.zeros((1,self.n_frames,self.obs_dim)) 
        
        count = 0                
        for i_step in range(self.horizon):
            action = plan_actions[i_step]
            action_repeat = plan_durations[i_step]
            if self.action_transformation is not None:
                action = self.action_transformation(action)
            for i_repeat in range(action_repeat):
                    next_state, reward, _, _ = env.step(action)
                    observation[0,count,:] = next_state
                    count += 1

        return observation 

    def record_video(self, env, plan_actions, rel_duration_plans, file_name='test_vid'):

        # going back and forth between if we should pass a list or a CEnv object...

        effective_horizon = self.horizon * self.plan_action_repeat
        plan_durations = np.round(rel_duration_plans * effective_horizon).astype(int)

        env.reset()

        recorder = VideoRecorder(curr_env, "{}.mp4".format(file_name))
        recorder.capture_frame()
                         
        for i_step in range(self.horizon):
            action = plan_actions[i_step]
            action_repeat = plan_durations[i_step]
            if self.action_transformation is not None:
                action = self.action_transformation(action)
            for i_repeat in range(action_repeat):
                    next_state, reward, _, _ = env.step(action)
                    recorder.capture_frame()

        recorder.close()
        print("FINISHED RECORDING\n")
        return



if __name__ == '__main__':
    # Some manual tests...

    print('lol')
    #rng = np.random.RandomState(1234)
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
