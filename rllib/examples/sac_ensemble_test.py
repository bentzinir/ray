import ray
from ray.rllib.env.action_mux import ActionMux
from ray.rllib.agents.sac import SACEnsembleTrainer
from ray.rllib.examples.sac_ensemble_train import get_config
from ray.rllib.examples.sac_ensemble_train import get_args
import time


if __name__ == "__main__":
    args, extra_args = get_args()

    ray.init()

    config = get_config(args)
    tester = SACEnsembleTrainer(config=config)
    tester.restore(args.checkpoint_dir)

    if type(args.env) is not str:
        env = args.env()
        if 'partial_ensemble_size' in config:
            env = ActionMux(env, ensemble_size=config['partial_ensemble_size'])

    state = env.reset()
    env.render()
    done = False
    cumulative_reward = 0

    while True:
        action = tester.compute_action(state)
        state, reward, done, _ = env.step(action)
        cumulative_reward += reward
        env.render()
        time.sleep(0.001)
        if done:
            # print(cumulative_reward)
            cumulative_reward = 0
            state = env.reset()
