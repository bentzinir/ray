import argparse
import ray
from ray import tune
from ray.rllib.agents.sac import SACEnsembleTrainer
from ray.rllib.agents.sac import SACTrainer


parser = argparse.ArgumentParser()
parser.add_argument("--stop-iters", type=int, default=200)
parser.add_argument("--num-cpus", type=int, default=0)
parser.add_argument("--env", type=str, default="none")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--eager", action="store_true")
parser.add_argument("--shared_actor_body", action="store_true")
parser.add_argument("--ensemble_size", type=int, default=1)
parser.add_argument("--verbose", type=int, default=1)
parser.add_argument("--num_workers", type=int, default=1)
parser.add_argument("--num_gpus", type=int, default=1)
parser.add_argument("--target_entropy", type=float, default=None)
parser.add_argument("--constant_alpha", action="store_true")

if __name__ == "__main__":
    args, extra_args = parser.parse_known_args()
    # args.env = 'CartPole-v0'
    # args.env = 'HalfCheetah-v3'
    config = {
            'env': args.env,
            'num_workers': args.num_workers,
            'num_gpus': args.num_gpus,
            'partial_ensemble_size': args.ensemble_size,
            'shared_actor_body': args.shared_actor_body,
            'eager': args.eager,
            'target_entropy': args.target_entropy,
            'constant_alpha': args.constant_alpha,
    }

    ray.init(num_cpus=args.num_cpus or None,
             # local_mode=True,
             # redis_max_memory=int(4e9),
             # object_store_memory=int(10e9),
             # memory=int(16e9)
             )
    if args.debug:
        trainer = SACEnsembleTrainer(config=config)
        i = 0
        while True:
            results = trainer.train()  # distributed training step
            print(f"Iter: {results['training_iteration']}, R: {results['episode_reward_mean']}")
    else:
        tune.run(SACEnsembleTrainer, config=config, verbose=args.verbose)
        # tune.run(SACTrainer, config=config, verbose=args.verbose)