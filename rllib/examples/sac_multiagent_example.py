import argparse
import gym
import ray
from ray import tune
from ray.rllib.agents.sac import SACTrainer
from ray.rllib.agents.sac.sac_tf_policy import SACTFPolicy

parser = argparse.ArgumentParser()
parser.add_argument("--stop-iters", type=int, default=200)
parser.add_argument("--num-cpus", type=int, default=0)
parser.add_argument("--env", type=str, default="none")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--tfe", action="store_true")
parser.add_argument("--shared_actor_body", action="store_true")
parser.add_argument("--ensemble_size", type=int, default=1)
parser.add_argument("--verbose", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--num_gpus", type=int, default=1)
parser.add_argument("--target_entropy", type=float, default=None)
parser.add_argument("--constant_alpha", action="store_true")
parser.add_argument("--ensemble_grid_search", action="store_true")
parser.add_argument("--asymmetric", action="store_true")
parser.add_argument("--local_mode", action="store_true")


def policy_mapping_func(x):
    return f"worker_p{x}"


if __name__ == "__main__":
    args, extra_args = parser.parse_known_args()
    # args.env = 'CartPole-v0'
    # args.env = 'HalfCheetah-v3'
    # args.debug = True
    env = gym.make(args.env)

    policies = dict()
    for e in range(args.ensemble_size):
        policies[f"worker_p{e}"] = (SACTFPolicy, env.observation_space, env.action_space, {})

    config = {
            'env': args.env,
            'num_workers': args.num_workers,
            'num_gpus': args.num_gpus,
            # 'constant_alpha': args.constant_alpha,
            # 'ensemble_size': args.ensemble_size,
            'framework': 'tfe' if args.tfe else 'tf',
            # 'target_entropy': args.target_entropy,
            # 'asymmetric': args.asymmetric,
            # "multiagent": {
            #     "policy_mapping_fn": lambda x: f"worker_p{x}", # if x == 0 else "worker_p2",
            #     'policies': policies,
            #     'policies_to_train': [f"worker_p{e}" for e in range(args.ensemble_size)]
            # },

    }
    ray.init(num_cpus=args.num_cpus or None,
             local_mode=args.local_mode,
             # redis_max_memory=int(4e9),
             # object_store_memory=int(10e9),
             # memory=int(16e9)
             )
    if args.debug:
        trainer = SACTrainer(config=config)
        # trainer = PPOTrainer(config=config)
        while True:
            results = trainer.train()  # distributed training step
            print(f"Iter: {results['training_iteration']}, R: {results['episode_reward_mean']}")
    else:
        tune.run(
                SACTrainer,
                # PPOTrainer,
                verbose=args.verbose,
                config=config)
