import gym
from collections import deque
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.tests.test_rollout_worker import MockEnv, MockEnv2
import numpy as np


def make_multiagent(env_name_or_creator):
    class MultiEnv(MultiAgentEnv):
        def __init__(self, config):
            num = config.pop("num_agents", 1)
            if isinstance(env_name_or_creator, str):
                self.agents = [
                    gym.make(env_name_or_creator) for _ in range(num)
                ]
            else:
                self.agents = [env_name_or_creator(config) for _ in range(num)]
            normalize_actions = config.pop("normalize_actions", True)
            if normalize_actions:
                from ray.rllib.env.normalize_actions import NormalizeActionWrapper
                self.agents = [NormalizeActionWrapper(agent) for agent in self.agents]
            self.dones = set()
            self.observation_space = self.agents[0].observation_space
            self.action_space = self.agents[0].action_space
            self.ensemble_reward_queues = [deque(maxlen=10) for _ in range(len(self.agents))]
            self.episode_rewards = {k: 0 for k in range(len(self.agents))}

        def update_agent_stats(self, i):
            self.ensemble_reward_queues[i].append(self.episode_rewards[i])
            self.episode_rewards[i] = 0

        def reset(self):
            self.dones = set()
            for aidx in range(len(self.agents)):
                self.update_agent_stats(aidx)
            return {i: a.reset() for i, a in enumerate(self.agents)}

        def step(self, action_dict):
            obs, rew, done, info = {}, {}, {}, {}
            for i, action in action_dict.items():
                if np.any(np.isnan(action)):
                    print("=================== (multi agent) Nan Detected =================")
                    input("(multi agent) Nan detected")
                obs[i], rew[i], done[i], info[i] = self.agents[i].step(action)
                info[i]["my_id"] = i
                info[i]["R"] = np.mean(self.ensemble_reward_queues[i])
                # rew[i] = rew[i] / len(self.agents)
                self.episode_rewards[i] += rew[i]
                if done[i]:
                    self.dones.add(i)
                    obs[i] = self.agents[i].reset()
                    self.update_agent_stats(i)
                    info[i]["R"] = np.mean(self.ensemble_reward_queues[i])
            done["__all__"] = len(self.dones) == len(self.agents)
            return obs, rew, done, info

        def render(self):
            [a.render() for a in self.agents]

    return MultiEnv


class BasicMultiAgent(MultiAgentEnv):
    """Env of N independent agents, each of which exits after 25 steps."""

    def __init__(self, num):
        self.agents = [MockEnv(25) for _ in range(num)]
        self.dones = set()
        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.Discrete(2)
        self.resetted = False

    def reset(self):
        self.resetted = True
        self.dones = set()
        return {i: a.reset() for i, a in enumerate(self.agents)}

    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}
        for i, action in action_dict.items():
            obs[i], rew[i], done[i], info[i] = self.agents[i].step(action)
            if done[i]:
                self.dones.add(i)
        done["__all__"] = len(self.dones) == len(self.agents)
        return obs, rew, done, info


class EarlyDoneMultiAgent(MultiAgentEnv):
    """Env for testing when the env terminates (after agent 0 does)."""

    def __init__(self):
        self.agents = [MockEnv(3), MockEnv(5)]
        self.dones = set()
        self.last_obs = {}
        self.last_rew = {}
        self.last_done = {}
        self.last_info = {}
        self.i = 0
        self.observation_space = gym.spaces.Discrete(10)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self):
        self.dones = set()
        self.last_obs = {}
        self.last_rew = {}
        self.last_done = {}
        self.last_info = {}
        self.i = 0
        for i, a in enumerate(self.agents):
            self.last_obs[i] = a.reset()
            self.last_rew[i] = None
            self.last_done[i] = False
            self.last_info[i] = {}
        obs_dict = {self.i: self.last_obs[self.i]}
        self.i = (self.i + 1) % len(self.agents)
        return obs_dict

    def step(self, action_dict):
        assert len(self.dones) != len(self.agents)
        for i, action in action_dict.items():
            (self.last_obs[i], self.last_rew[i], self.last_done[i],
             self.last_info[i]) = self.agents[i].step(action)
        obs = {self.i: self.last_obs[self.i]}
        rew = {self.i: self.last_rew[self.i]}
        done = {self.i: self.last_done[self.i]}
        info = {self.i: self.last_info[self.i]}
        if done[self.i]:
            rew[self.i] = 0
            self.dones.add(self.i)
        self.i = (self.i + 1) % len(self.agents)
        done["__all__"] = len(self.dones) == len(self.agents) - 1
        return obs, rew, done, info


class RoundRobinMultiAgent(MultiAgentEnv):
    """Env of N independent agents, each of which exits after 5 steps.

    On each step() of the env, only one agent takes an action."""

    def __init__(self, num, increment_obs=False):
        if increment_obs:
            # Observations are 0, 1, 2, 3... etc. as time advances
            self.agents = [MockEnv2(5) for _ in range(num)]
        else:
            # Observations are all zeros
            self.agents = [MockEnv(5) for _ in range(num)]
        self.dones = set()
        self.last_obs = {}
        self.last_rew = {}
        self.last_done = {}
        self.last_info = {}
        self.i = 0
        self.num = num
        self.observation_space = gym.spaces.Discrete(10)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self):
        self.dones = set()
        self.last_obs = {}
        self.last_rew = {}
        self.last_done = {}
        self.last_info = {}
        self.i = 0
        for i, a in enumerate(self.agents):
            self.last_obs[i] = a.reset()
            self.last_rew[i] = None
            self.last_done[i] = False
            self.last_info[i] = {}
        obs_dict = {self.i: self.last_obs[self.i]}
        self.i = (self.i + 1) % self.num
        return obs_dict

    def step(self, action_dict):
        assert len(self.dones) != len(self.agents)
        for i, action in action_dict.items():
            (self.last_obs[i], self.last_rew[i], self.last_done[i],
             self.last_info[i]) = self.agents[i].step(action)
        obs = {self.i: self.last_obs[self.i]}
        rew = {self.i: self.last_rew[self.i]}
        done = {self.i: self.last_done[self.i]}
        info = {self.i: self.last_info[self.i]}
        if done[self.i]:
            rew[self.i] = 0
            self.dones.add(self.i)
        self.i = (self.i + 1) % self.num
        done["__all__"] = len(self.dones) == len(self.agents)
        return obs, rew, done, info


MultiAgentCartPole = make_multiagent("CartPole-v0")
MultiAgentMountainCar = make_multiagent("MountainCarContinuous-v0")
MultiAgentPendulum = make_multiagent("Pendulum-v0")
