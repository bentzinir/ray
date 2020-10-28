import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.tests.test_rollout_worker import MockEnv, MockEnv2
import numpy as np
from collections import deque


def make_multiagent(env_name_or_creator):
    class MultiEnv(MultiAgentEnv):
        def __init__(self, config):
            self.nagents = config.pop("num_agents", 1)
            print(f"####################------------nagents: {self.nagents}-------------------#####################3")
            if isinstance(env_name_or_creator, str):
                self.agents = [
                    gym.make(env_name_or_creator) for _ in range(self.nagents)
                ]
            else:
                self.agents = [env_name_or_creator(config) for _ in range(self.nagents)]
            if config.pop("warp_obs", False):
                from ray.rllib.env.atari_wrappers import WarpFrame
                dim = config.pop("warp_dim", None)
                self.agents = [WarpFrame(agent, dim=dim) for agent in self.agents]
            self.dones = set()
            if hasattr(self.agents[0], 'spec'):
                # todo: do we need to scale the spec according to nagents? for example num_steps?
                self.spec = self.agents[0].spec
            self.observation_space = self.agents[0].observation_space
            self.action_space = self.agents[0].action_space
            self.reward_queues = [deque(maxlen=50) for _ in range(self.nagents)]
            self.episode_rewards = [0 for _ in range(self.nagents)]
            self.nresets = [0 for _ in range(self.nagents)]
            self.elapsed = [0 for _ in range(self.nagents)]

        def reset_i(self, i):
            self.nresets[i] += 1
            self.elapsed[i] = 0
            self.episode_rewards[i] = 0
            return self.agents[i].reset()

        def reset(self):
            self.dones = set()
            obses = {}
            for aidx in range(self.nagents):
                obses[self.idx2id(aidx)] = self.reset_i(aidx)
            return obses

        def id2idx(self, agent_id):
            return int(agent_id[0])

        def idx2id(self, idx):
            return f"{idx}_{self.nresets[idx]}"

        def step(self, action_dict):
            obs, rew, done, info = {}, {}, {}, {}
            for agent_id, action in action_dict.items():
                idx = self.id2idx(agent_id)
                self.elapsed[idx] += 1
                if np.any(np.isnan(action)):
                    input("(MultiEnv) Nan detected...")
                obs[agent_id], rew[agent_id], done[agent_id], info[agent_id] = self.agents[idx].step(action)
                self.episode_rewards[idx] += rew[agent_id]
                info[agent_id]['nresets'] = self.nresets[idx]
                if done[agent_id]:
                    self.dones.add(idx)
                    self.reward_queues[idx].append(self.episode_rewards[idx])
                    # reset agent at idx. This will also modify agent_id of idx.
                    obs_i = self.reset_i(idx)
                    # spawn a new agent id at idx
                    obs[self.idx2id(idx)] = obs_i
                    rew[self.idx2id(idx)] = 0
                    done[self.idx2id(idx)] = False
                    info[self.idx2id(idx)] = {}
            done["__all__"] = len(self.dones) == self.nagents
            return obs, rew, done, info

        def render(self):
            [a.render() for i, a in enumerate(self.agents)]

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
