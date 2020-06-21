import gym
import numpy as np
import numpy.matlib as matlib
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE


class ActionMux(gym.Wrapper):
    def __init__(self, env, ensemble_size=None):
        # super(TimeLimit, self).__init__(env)
        gym.Wrapper.__init__(self, env)
        self.ensemble_size = ensemble_size
        self.active_policy = None
        if isinstance(self.action_space, gym.spaces.Box):
            low = matlib.repmat(self.action_space.low, m=ensemble_size, n=1)
            high = matlib.repmat(self.action_space.high, m=ensemble_size, n=1)
            self.action_space = gym.spaces.Box(low=low, high=high, dtype=self.action_space.dtype)
        elif isinstance(self.action_space, gym.spaces.Discrete):
            self.action_space = gym.spaces.MultiDiscrete([self.action_space.n for _ in range(ensemble_size)])
        print(f"Ensemble size: {ensemble_size}, ActionMux Action_Space: {self.env.action_space}")

    def step(self, action_dict):
        action = action_dict[self.active_policy]
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.active_policy = np.random.choice(range(self.ensemble_size))
        # print(f"Current Active Policy: {self.active_policy}")
        return obs

    def _dictionize(self, val):
        x = {}
        for i in range(self.ensemble_size):
            x[i] = val
        return x