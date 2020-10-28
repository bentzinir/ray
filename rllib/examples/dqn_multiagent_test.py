import ray
from ray.rllib.agents.dqn import DQNMATrainer as Trainer
from ray.rllib.examples.dqn_multiagent_train import get_config, get_args
import time
from ray.rllib.examples.env.multi_agent import make_multiagent
from ray.rllib.env.atari_wrappers import is_atari, wrap_deepmind
import numpy as np


def wrap(env):
    for i, a in enumerate(env.agents):
        if is_atari(a):
            env.agents[i] = wrap_deepmind(a)
    return env


if __name__ == "__main__":
    args, extra_args = get_args()

    ray.init()

    config = get_config(args)
    tester = Trainer(config=config)
    tester.restore(args.checkpoint_dir)
    env = make_multiagent(args.env)(config["env_config"])
    env = wrap(env)
    obs = env.reset()
    env.render()
    done = False
    cumulative_reward = 0
    counter = 0

    while True:
        action = {}
        counter += 1
        for agent_id, agent_obs in obs.items():
            policy_id = config['multiagent']['policy_mapping_fn'](agent_id)
            # if config["env_config"]["normalize_obs"]:
            #     agent_obs = agent_obs.astype(np.float32) / 255
            action[agent_id] = tester.compute_action(agent_obs, policy_id=policy_id, explore=False)
        print(f"counter: {counter}")
        obs, reward, done, info = env.step(action)
        if args.vis_sleep:
            time.sleep(args.vis_sleep)
        env.render()
        done = done['__all__']
        # sum up reward for all agents
        cumulative_reward += sum(reward.values())
        if done:
            obs = env.reset()
            print(f"R: {cumulative_reward}")
            cumulative_reward = 0
            counter = 0
