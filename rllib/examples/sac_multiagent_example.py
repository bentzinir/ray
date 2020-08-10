import argparse
import gym
import ray
from ray import tune
from ray.rllib.agents.sac import SACTrainer
from ray.rllib.policy.sample_batch import SampleBatch
import numpy as np
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from typing import Dict


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop-iters", type=int, default=200)
    parser.add_argument("--num-cpus", type=int, default=0)
    parser.add_argument("--env", type=str, default="none")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--tfe", action="store_true")
    parser.add_argument("--shared_actor", action="store_true")
    parser.add_argument("--ensemble_size", type=int, default=1)
    parser.add_argument("--timescale", type=int, default=10000)
    parser.add_argument("--timescale_grid_search", action="store_true")
    parser.add_argument("--timesteps", type=int, default=1000000)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--target_entropy", type=float, default=None)
    parser.add_argument("--ensemble_grid_search", action="store_true")
    parser.add_argument("--asymmetric", action="store_true")
    parser.add_argument("--local_mode", action="store_true")
    parser.add_argument("--experience_masking", action="store_true")
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--alpha_grid_search", action="store_true")
    parser.add_argument("--local_dir", type=str, default="none")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--checkpoint_freq", type=int, default=0)
    return parser


def callback_builder():

    class MyCallbacks(DefaultCallbacks):

        def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv,
                             policies: Dict[str, Policy],
                             episode: MultiAgentEpisode, **kwargs):
            ...
            # episode.custom_metrics[f"max_reward"] = []
            # for i in range(worker.env.ensemble_size):
            #     episode.custom_metrics[f"reward_{i}"] = []

        def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv,
                            episode: MultiAgentEpisode, **kwargs):
            pass

        def on_postprocess_trajectory(
            self, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):

            # postprocessed_batch['members'] = np.array([info['active_member'] for info in postprocessed_batch['infos']],
            #                                           dtype=np.int32)

            return

        # def on_sample_end(self, worker: RolloutWorker, samples: SampleBatch, **kwargs):
        #     samples['members'] = np.array([info['active_member'] for info in samples['infos']], dtype=np.int32)

        def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                           policies: Dict[str, Policy], episode: MultiAgentEpisode,
                           **kwargs):
            ...
            # ensemble_rewards = episode.last_info_for()["ensemble_rewards"]
            # episode.custom_metrics[f"max_reward"].append(np.max(ensemble_rewards))
            # for i, ri in enumerate(ensemble_rewards):
            #     episode.custom_metrics[f"reward_{i}"].append(ri)

    return MyCallbacks


def central_critic_observer(agent_obs, **kw):
    """Rewrites the agent obs to include opponent data for training."""

    # new_obs = {
    #     0: {
    #         "own_obs": agent_obs[0],
    #         "opponent_obs": agent_obs[1],
    #     },
    #     1: {
    #         "own_obs": agent_obs[1],
    #         "opponent_obs": agent_obs[0],
    #     },
    # }
    new_obs = agent_obs
    return new_obs


def make_env_get_spaces(args):
    from gym.spaces import Dict as gym_dict

    if type(args.env) is str:
        import gym
        env = gym.make(args.env)
    else:
        env = args.env()

    # observer_space = gym_dict({
    #     "own_obs": env.observation_space,
    #     "opponent_obs": env.observation_space,
    # })

    return env.observation_space, env.action_space


def get_config(args):
    if args.experience_masking:
        batch_scale = args.ensemble_size
    else:
        batch_scale = 1
    return {
        'env': args.env,
        'num_workers': args.num_workers,
        'num_gpus': args.num_gpus,
        'framework': 'tfe' if args.tfe else 'tf',
        'target_entropy': args.target_entropy,
        "callbacks": callback_builder(),
        'train_batch_size': batch_scale * args.batch_size,
        'gamma': args.gamma,
        # 'alpha': tune.grid_search([0.4, 0.3, 0.2, 0.1]) if args.alpha_grid_search else args.alpha,
        "multiagent": {
            "policies": {
                "pol0": (None, *make_env_get_spaces(args), {}),
                "pol1": (None, *make_env_get_spaces(args), {}),
                # "pol2": (None, *make_env_get_spaces(args), {}),
            },
            "policy_mapping_fn": lambda x: f"pol{x}",
            # "observation_fn": central_critic_observer,
        },
        'ensemble_size': args.ensemble_size
    }


def get_args():
    args, extra_args = get_parser().parse_known_args()
    if args.env == 'maze':
        from ray.rllib.examples.env.maze_env import MazeEnv
        args.env = MazeEnv
    return args, extra_args


if __name__ == "__main__":
    args, extra_args = get_args()

    ray.init(num_cpus=args.num_cpus or None,
             local_mode=args.local_mode,
             # redis_max_memory=int(4e9),
             # object_store_memory=int(10e9),
             # memory=int(16e9)
             )

    if args.debug:
        trainer = SACTrainer(config=get_config(args))
        i = 0
        while True:
            results = trainer.train()  # distributed training step
            print(f"Iter: {results['training_iteration']}, R: {results['episode_reward_mean']}")
    else:
        tune.run(SACTrainer,
                 verbose=args.verbose,
                 config=get_config(args),
                 stop={
                     "timesteps_total": args.timesteps,
                    },
                 # resources_per_trial={'cpu': 2,#config['num_workers'],
                 #                      'gpu': 0.5,#config['num_gpus']
                 #    },
                 reuse_actors=True,
                 local_dir=args.local_dir,
                 checkpoint_freq=args.checkpoint_freq,
                 checkpoint_at_end=True,
                 )
