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
parser.add_argument("--eager", action="store_true")


def policy_mapper(agent_id):
    import random
    if agent_id.startswith("supervisor_"):
        return "supervisor_policy"
    else:
        return random.choice(["worker_p1", "worker_p2"])


if __name__ == "__main__":
    args, extra_args = parser.parse_known_args()
    args.env = 'CartPole-v0'
    # args.debug = True
    env = gym.make(args.env)

    config = {
            'env': args.env,
            "multiagent": {
                "policy_mapping_fn": lambda x: "worker_p1" if x == 0 else "worker_p2",
                "policy_graphs": {
                    "worker_p1": (SACTFPolicy, env.observation_space, env.action_space, {}),
                    "worker_p2": (SACTFPolicy, env.observation_space, env.action_space, {}),
                },
                "policies_to_train": ["worker_p1", "worker_p2"],
            },
            'ensemble_size': 1,
            'eager': args.eager
    }

    if args.debug:
        trainer = SACTrainer(config=config)
        while True:
            print(trainer.train())  # distributed training step
    else:
        ray.init(num_cpus=args.num_cpus or None)
        tune.run(SACTrainer, config=config,)
