from ray.rllib.examples.sac_multiagent_train import get_args
import ray
from ray import tune
from ray.rllib.agents.sac import SACTrainer
import gym
from ray.rllib.examples.env.multi_agent import make_multiagent
import numpy as np


def get_config(args):
    single_env = gym.make(args.env)
    policies = {
        "policy_{}".format(i): (None, single_env.observation_space, single_env.action_space, {})
        for i in range(args.ensemble_size)
    }

    return {
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
        'gamma': args.gamma,
        'buffer_size': args.buffer_size,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": (lambda x: f"policy_{x[0]}"),
        },
    }


if __name__ == "__main__":
    args, extra_args = get_args()

    ray.init(num_cpus=args.num_cpus or None,
             local_mode=args.local_mode)

    config = get_config(args)

    if args.debug:
        trainer = SACTrainer(config=config)
        i = 0
        while True:
            results = trainer.train()
            print(f"Iter: {results['training_iteration']}, R: {results['episode_reward_mean']}")
    else:
        tune.run(SACTrainer,
                 verbose=args.verbose,
                 config=config,
                 stop={"timesteps_total": args.timesteps},
                 reuse_actors=True,
                 local_dir=args.local_dir,
                 checkpoint_freq=args.checkpoint_freq,
                 checkpoint_at_end=True,
                 )
