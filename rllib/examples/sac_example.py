from ray.rllib.examples.sac_multiagent_train import get_parser, get_config
import ray
from ray import tune
from ray.rllib.agents.sac import SACTrainer


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
