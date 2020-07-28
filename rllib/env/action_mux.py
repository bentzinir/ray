import gym
import numpy as np
import numpy.matlib as matlib
from collections import deque


class ActionMux(gym.Wrapper):
    def __init__(self, env, ensemble_size=None):
        gym.Wrapper.__init__(self, env)
        self.ensemble_size = ensemble_size
        if isinstance(self.action_space, gym.spaces.Box):
            low = matlib.repmat(self.action_space.low, m=ensemble_size, n=1)
            high = matlib.repmat(self.action_space.high, m=ensemble_size, n=1)
            self.action_space = gym.spaces.Box(low=low, high=high, dtype=self.action_space.dtype)
        elif isinstance(self.action_space, gym.spaces.Discrete):
            self.action_space = gym.spaces.MultiDiscrete([self.action_space.n for _ in range(ensemble_size)])
        print(f"Ensemble size: {ensemble_size}, ActionMux Action_Space: {self.env.action_space}")
        self.ensemble_reward_ques = [deque(maxlen=1) for _ in range(ensemble_size)]
        self.active_member = np.random.choice(range(self.ensemble_size))
        self.episode_reward = 0

    def wrap_obs(self, obs):
        return obs

    def step(self, action_dict):
        action = action_dict[self.active_member]
        observation, reward, done, info = self.env.step(action)
        self.episode_reward += reward
        info["ensemble_rewards"] = self.ensemble_rewards
        info["active_member"] = self.active_member
        return self.wrap_obs(observation), reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.ensemble_reward_ques[self.active_member].append(self.episode_reward)
        self.episode_reward = 0
        self.active_member = np.random.choice(range(self.ensemble_size))
        return self.wrap_obs(obs)

    @property
    def ensemble_rewards(self):
        return [np.mean(ensemble) for ensemble in self.ensemble_reward_ques]