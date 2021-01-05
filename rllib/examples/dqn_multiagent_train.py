import argparse
import gym
import ray
from ray import tune
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from typing import Dict
from ray.rllib.examples.env.multi_agent import make_multiagent
import ray.rllib.examples.env.my_envs
import numpy as np
from ray.rllib.models.tf.visionnet import VisionNetwork
from ray.rllib.examples.parser_args import get_config
import copy

# switch between DQN and multiagent DQN for debugging purposes
DQN_MODE = False
if DQN_MODE:
    print(f" !!!!! DQN Mode: !!!!! ")
    from ray.rllib.agents.dqn import DQNTrainer as Trainer
    from ray.rllib.agents.dqn.dqn import DEFAULT_CONFIG
else:
    from ray.rllib.agents.dqn import DQNMATrainer as Trainer
    from ray.rllib.agents.dqn.dqnma import DEFAULT_CONFIG


def base_model_init():
    global BASE_MODEL
    BASE_MODEL = {"main": None, "target": None}


def callback_builder():

    class MyCallbacks(DefaultCallbacks):

        def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv,
                             policies: Dict[str, Policy],
                             episode: MultiAgentEpisode, **kwargs):
            pass

        def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv,
                            episode: MultiAgentEpisode, **kwargs):
            pass

        def on_postprocess_trajectory(
            self, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):
            pass

        def on_sample_end(self, worker: RolloutWorker, samples: SampleBatch, **kwargs):
            pass

        def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                           policies: Dict[str, Policy], episode: MultiAgentEpisode,
                           **kwargs):
            for i in range(worker.env.nagents):
                episode.custom_metrics[f"episodic_return_{i}"] = np.mean(worker.env.reward_queues[i])
                episode.custom_metrics[f"nresets_{i}"] = worker.env.nresets[i]
            episode.custom_metrics["episodic_max"] = np.max([r[-1] for r in worker.env.reward_queues])
    return MyCallbacks


def build_trainer_config(config):
    config = copy.deepcopy(config)
    if isinstance(config["env"], str):
        single_env = gym.make(config["env"])
    elif isinstance(config["env"], dict) and config["env"].get('grid_search', False):  # grid search mode
        single_env = gym.make(config["env"]['grid_search'][0])
    elif callable(config["env"]):
        single_env = config["env"]()
    obs_space = single_env.observation_space
    act_space = single_env.action_space
    if isinstance(config["ensemble_size"], int):
        num_policies = config["ensemble_size"]
    elif config["ensemble_size"].get('grid_search', False):
        num_policies = max(config["ensemble_size"]['grid_search'])
    config["callbacks"] = callback_builder()
    config["env_config"] = {"N": config["ensemble_size"],
                            "env_id": config["env"]}
    config["env"] = make_multiagent()
    config["multiagent"] = {
        "policies": {"policy_{}".format(i): (None, obs_space, act_space, {}) for i in range(num_policies)},
        "policy_mapping_fn": (lambda x: f"policy_{x[0]}")}
    config["compress_observations"] = np.issubdtype(single_env.observation_space.dtype, np.integer)
    config["model"] = {
        # "custom_model": None if isinstance(obs_space, gym.spaces.Discrete) or len(obs_space.shape) <= 2 else VisionNetwork,
        "custom_model_config": {"shared_base_model": config["shared_base_model"]}}

    for key in [key for key in config if key not in DEFAULT_CONFIG]:
        del config[key]
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, default="none")
    args, extra_args = parser.parse_known_args()
    config = get_config(args.f)
    trainer_config = build_trainer_config(config)

    ray.init(num_cpus=config["num_cpus"] or None,
             local_mode=config["local_mode"],)

    if config["debug"]:
        trainer = Trainer(config=trainer_config)
        while True:
            results = trainer.train()
            print(f"Iter: {results['training_iteration']}, R: {results['episode_reward_mean']}")
    else:
        tune.run(Trainer,
                 verbose=config["verbose"],
                 num_samples=config["grid_repeats"],
                 config=trainer_config,
                 stop={"timesteps_total": config["timesteps"]},
                 reuse_actors=True,
                 local_dir=config["local_dir"],
                 checkpoint_freq=config["checkpoint_freq"],
                 checkpoint_at_end=True,
                 global_checkpoint_period=np.inf)
