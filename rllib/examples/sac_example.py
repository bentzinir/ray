import argparse
import ray
from ray import tune
from ray.rllib.agents.sac import SACTrainer


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


def get_config(args):

    return {
        'env': args.env,
        'num_workers': args.num_workers,
        'num_gpus': args.num_gpus,
        'framework': 'tfe' if args.tfe else 'tf',
        'gamma': args.gamma,
        # 'alpha': tune.grid_search([0.4, 0.3, 0.2, 0.1]) if args.alpha_grid_search else args.alpha,
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
             local_mode=args.local_mode)

    if args.debug:
        trainer = SACTrainer(config=get_config(args))
        i = 0
        while True:
            results = trainer.train()
            print(f"Iter: {results['training_iteration']}, R: {results['episode_reward_mean']}")
    else:
        tune.run(SACTrainer,
                 verbose=args.verbose,
                 config=get_config(args),
                 stop={"timesteps_total": args.timesteps},
                 reuse_actors=True,
                 local_dir=args.local_dir,
                 checkpoint_freq=args.checkpoint_freq,
                 checkpoint_at_end=True,
                 )
