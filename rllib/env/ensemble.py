import gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE
import copy


class Ensemble(MultiAgentEnv, gym.Wrapper):
    def __init__(self, env, ensemble_size=None):
        # super(TimeLimit, self).__init__(env)
        gym.Wrapper.__init__(self, env)
        self.env_vec = {k: copy.deepcopy(env) for k in range(ensemble_size)}
        self.ensemble_size = ensemble_size
        self.dones = set()

    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}
        for i, action in action_dict.items():
            obs[i], rew[i], done[i], info[i] = self.agents[i].step(action)
            if done[i]:
                self.dones.add(i)
        done["__all__"] = len(self.dones) == len(self.agents)
        return obs, rew, done, info

    def reset(self, **kwargs):
        self.dones = set()
        return {k: self.env_vec[k].reset(**kwargs) for k in range(self.ensemble_size)}
