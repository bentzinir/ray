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
from ray.rllib.examples.env.multi_agent import make_multiagent
import random


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


def get_q_value(policy, batch):
    if policy.loss_initialized():
        model_out_t, _ = policy.model({
                "obs": batch[SampleBatch.CUR_OBS],
                "is_training": policy._get_is_training_placeholder(),
            }, [], None)
        actions = batch[SampleBatch.ACTIONS]
        if policy.model.discrete:
            qvec = policy.model.get_q_values(model_out_t)
            if policy.config['framework'] in ['tf2', 'tfe']:
                return [q[a] for q, a in zip(qvec.numpy(), actions)]
            else:  # 'tensorflow'
                qvec_np = qvec.eval(session=policy.get_session())
                return [q[a] for q, a in zip(qvec_np, actions)]

        else:
            return policy.model.get_q_values(model_out_t, actions)
    else:
        return -np.inf * np.ones_like(batch[SampleBatch.REWARDS])


def callback_builder():

    class MyCallbacks(DefaultCallbacks):

        def on_episode_start(self, worker: RolloutWorker, base_env: BaseEnv,
                             policies: Dict[str, Policy],
                             episode: MultiAgentEpisode, **kwargs):
            pass

        def on_episode_step(self, worker: RolloutWorker, base_env: BaseEnv,
                            episode: MultiAgentEpisode, **kwargs):
            pass

        @staticmethod
        def average_performance(batches):
            return {k: batches[k][1]['infos'][0]['R'] for k in batches.keys()}

        def on_postprocess_trajectory(
            self, worker: RolloutWorker, episode: MultiAgentEpisode,
            agent_id: str, policy_id: str, policies: Dict[str, Policy],
            postprocessed_batch: SampleBatch,
            original_batches: Dict[str, SampleBatch], **kwargs):

            policy, _ = original_batches[agent_id]

            average_performance = self.average_performance(original_batches)

            masters = [okey for okey in original_batches.keys() if okey < agent_id]
            if len(masters) > 0:
                opp_index = random.choice(masters)
                opolicy, obatch = original_batches[opp_index]
                for i, (oobs, oinfo) in enumerate(zip(obatch["obs"], obatch["infos"])):
                    if oinfo['my_id'] == opp_index:
                        postprocessed_batch["opponent_obs"][i] = oobs.copy()
                        postprocessed_batch["valid_opp_obs"][i] = True

            slaves = [okey for okey in original_batches.keys() if
                      okey > agent_id and average_performance[okey] > average_performance[agent_id]]
            if len(slaves) > 0:
                opp_index = random.choice(slaves)
                opolicy, obatch = original_batches[opp_index]
                # assert that original_batches keys are indeed agent_id
                # TODO: this assertion may fail. replace with a logic that extracts individual samples
                assert np.all([oinfo['my_id'] == opp_index for oinfo in obatch["infos"]])
                # own_qs = get_q_value(policy, obatch)
                # slave_qs = get_q_value(opolicy, obatch)
                for i in range(obatch.count):
                    for key in obatch.keys():
                        val = obatch[key][i].copy()
                        if key == 'valid_opp_obs':
                            val = False
                        postprocessed_batch[key] = np.append(postprocessed_batch[key], [val], axis=0)
            return

        def on_sample_end(self, worker: RolloutWorker, samples: SampleBatch, **kwargs):
            pass

        def on_episode_end(self, worker: RolloutWorker, base_env: BaseEnv,
                           policies: Dict[str, Policy], episode: MultiAgentEpisode,
                           **kwargs):
            pass

    return MyCallbacks


def central_critic_observer(agent_obs, **kw):
    """Rewrites the agent obs to include opponent data for training."""
    return agent_obs


def get_config(args):
    if args.experience_masking:
        batch_scale = args.ensemble_size
    else:
        batch_scale = 1

    # Get obs- and action Spaces.
    if isinstance(args.env, str):
        single_env = gym.make(args.env)
    else:
        single_env = args.env()
    # from gym.spaces import Dict
    # obs_space = Dict({"own": single_env.observation_space})
    obs_space = single_env.observation_space
    act_space = single_env.action_space

    policies = {
        "policy_{}".format(i): (None, obs_space, act_space, {})
        for i in range(args.ensemble_size)
    }

    return {
        'env': make_multiagent(args.env),
        "env_config": {
            "num_agents": args.ensemble_size,
            "normalize_actions": True,
            },
        'num_workers': args.num_workers,
        'num_gpus': args.num_gpus,
        'framework': 'tfe' if args.tfe else 'tf',
        # 'framework': 'torch',
        'target_entropy': args.target_entropy,
        "callbacks": callback_builder(),
        'train_batch_size': batch_scale * args.batch_size,
        'gamma': args.gamma,
        # 'alpha': tune.grid_search([0.4, 0.3, 0.2, 0.1]) if args.alpha_grid_search else args.alpha,
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": (lambda x: f"policy_{x}"),
            "observation_fn": central_critic_observer,
        },
        "normalize_actions": False,
        "alpha": args.alpha,
        "beta": args.beta,
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
