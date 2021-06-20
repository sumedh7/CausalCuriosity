import gym


class FrameSkip(gym.Wrapper):
    def __init__(self, env, frame_skip):
        super().__init__(env)

        assert type(frame_skip) is int
        assert frame_skip >= 1
        self.frame_skip = frame_skip

    def step(self, action):
        total_reward = 0.0

        for i in range(self.frame_skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
        return obs, total_reward, done, info
