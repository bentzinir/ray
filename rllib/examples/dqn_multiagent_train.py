import argparse
import gym
import ray
from ray import tune
from ray.rllib.agents.dqn import DQNMATrainer
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from typing import Dict
from ray.rllib.examples.env.multi_agent import make_multiagent
import yaml
from ray.rllib.examples.env.maze_env import MazeEnv
import collections.abc
import numpy as np
from ray.rllib.models.tf.visionnet import VisionNetwork


def base_model_init():
    global BASE_MODEL
    BASE_MODEL = {"main": None, "target": None}


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-iters", type=int, default=200)
    parser.add_argument("--num-cpus", type=int, default=0)
    parser.add_argument("--env", type=str, default="none")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--framework", type=str, default="tf")
    parser.add_argument("--shared_base_model", action="store_true")
    parser.add_argument("--ensemble_size", type=int, default=1)
    parser.add_argument("--timescale", type=int, default=10000)
    parser.add_argument("--timescale_grid_search", action="store_true")
    parser.add_argument("--timesteps", type=int, default=1000000)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--target_entropy", type=float, default=None)
    parser.add_argument("--ensemble_grid_search", action="store_true")
    parser.add_argument("--local_mode", action="store_true")
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--local_dir", type=str, default="none")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--checkpoint_freq", type=int, default=0)
    parser.add_argument("--buffer_size", type=int, default=50000)
    parser.add_argument("--entropy_scale", type=float, default=1.)
    parser.add_argument("--target_div", type=float, default=None)
    parser.add_argument("--entropy_lr", type=float, default=5e-4)
    parser.add_argument("--beta_learning_rate", type=float, default=3e-4)
    parser.add_argument("--shuffle_data", action="store_true")
    parser.add_argument("--divergence_type", type=str, default="none")
    parser.add_argument("--vis_sleep", type=float, default=0.0)
    parser.add_argument("--yaml_config", type=str, default="none")
    parser.add_argument("--initial_beta", type=float, default=1.0)
    parser.add_argument("--object_store_memory", type=int, default=1000000000)
    return parser


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


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

    return MyCallbacks


def get_config(args):
    # Get obs- and action Spaces.
    if isinstance(args.env, str):
        single_env = gym.make(args.env)
    else:
        single_env = args.env({"spatial": args.spatial})
    obs_space = single_env.observation_space
    act_space = single_env.action_space
    assert args.yaml_config != "none"
    tuned_yaml = open(args.yaml_config)
    tuned_config = yaml.load(tuned_yaml, Loader=yaml.FullLoader)

    (k, tuned_config), = tuned_config.items()
    config = tuned_config["config"]

    policies = {
        "policy_{}".format(i): (None, obs_space, act_space, {})
        for i in range(args.ensemble_size)
    }

    override_config = {
        'env': make_multiagent(args.env),
        "env_config": {
            "num_agents": args.ensemble_size,
            "spatial": args.spatial,  # a flag for maze env
            # We normalize observations of any integer type by 255
            "normalize_obs": np.issubdtype(single_env.observation_space.dtype, np.integer),
            },
        'num_workers': args.num_workers,
        'num_gpus': args.num_gpus,
        'framework': args.framework,
        "callbacks": callback_builder(),
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": (lambda x: f"policy_{x[0]}"),
        },
        "beta": args.beta,
        "initial_beta": args.initial_beta,
        'gamma': args.gamma,
        "buffer_size": args.buffer_size,
        "entropy_scale": args.entropy_scale,
        "entropy_lr": args.entropy_lr,
        "target_div": args.target_div,
        "divergence_type": args.divergence_type,
        "compress_observations": np.issubdtype(single_env.observation_space.dtype, np.integer),
        # "learning_starts": args.learning_starts,
        "model": {
            "custom_model": None if isinstance(obs_space, gym.spaces.Discrete) or len(
                obs_space.shape) <= 2 else VisionNetwork,
            "custom_model_config": {
                "shared_base_model": args.shared_base_model,
            }
        }
    }
    config = update(config, override_config)
    return config


def get_args():
    args, extra_args = get_parser().parse_known_args()
    if args.env == 'maze':
        args.env = MazeEnv
        args.spatial = False
    elif args.env == 'maze-spatial':
        args.env = MazeEnv
        args.spatial = True
    else:
        args.spatial = False
    return args, extra_args


if __name__ == "__main__":
    args, extra_args = get_args()

    ray.init(num_cpus=args.num_cpus or None,
             local_mode=args.local_mode,
             object_store_memory=args.object_store_memory)

    if args.debug:
        trainer = DQNMATrainer(config=get_config(args))
        i = 0
        while True:
            results = trainer.train()
            print(f"Iter: {results['training_iteration']}, R: {results['episode_reward_mean']}")
    else:
        tune.run(DQNMATrainer,
                 verbose=args.verbose,
                 config=get_config(args),
                 stop={"timesteps_total": args.timesteps},
                 reuse_actors=True,
                 local_dir=args.local_dir,
                 checkpoint_freq=args.checkpoint_freq,
                 checkpoint_at_end=True,
                 )
