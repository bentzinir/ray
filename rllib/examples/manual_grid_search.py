from multiprocessing import Process, Lock, Manager
import argparse
import random
from ray.rllib.examples.parser_args import get_config
import itertools
import ray
from ray import tune
import copy
from ray.rllib.examples.dqn_multiagent_train import build_trainer_config

# switch between DQN and multi-agent DQN for debugging purposes
DQN_MODE = False
if DQN_MODE:
    print(f" !!!!! DQN Mode: !!!!! ")
    from ray.rllib.agents.dqn import DQNTrainer as Trainer
else:
    from ray.rllib.agents.dqn import DQNMATrainer as Trainer


class MyWorker:
    def __init__(self, lock, driver, i):
        self.idx = i
        self.driver = driver
        self.lock = lock

    def run(self):
        self.driver.mark_tasks(self.idx)


class Driver:
    def __init__(self, combinations, keys, config):
        manager = Manager()
        self.lock = Lock()
        self.lst = manager.list()
        self.keys = keys
        self.config = config
        self.dict = manager.dict({combination: False for combination in combinations})
        self.lst.append(self.dict)

    def workerrun(self, lock, i):
        worker1 = MyWorker(lock, self, i)
        worker1.run()

    def run(self):
        D = [Process(target=self.workerrun, args=(self.lock, i)) for i in range(5)]
        for d in D:
            d.start()
        for d in D:
            d.join()

    def mark_tasks(self, i):
        done = False
        while not done:
            done = True
            d = self.lst[0]
            unprocessed = [key for key, val in d.items() if not val]
            if unprocessed:
                done = False
                r = random.choice(unprocessed)
                print(f"thread id: {i}, is processing job: {r}")
                config = copy.deepcopy(self.config)
                for key, val in zip(self.keys, r):
                    config[key] = val
                trainer_config = build_trainer_config(config)
                # execute the "de-serialized" config
                self.execute_task(trainer_config, idx=i)
                d[r] = True

    def execute_task(self, trainer_config, idx):
        ray.init(num_cpus=config["num_cpus"] or None,
                 local_mode=config["local_mode"],
                 dashboard_host=f'127.0.0.{idx}')

        if config["debug"]:
            trainer = Trainer(config=trainer_config)
            while True:
                results = trainer.train()
                print(f"Iter: {results['training_iteration']}, R: {results['episode_reward_mean']}")
        else:
            tune.run(Trainer,
                     verbose=config["verbose"],
                     num_samples=config["grid_repeats"],
                     config=trainer_config,
                     stop={"timesteps_total": config["timesteps"]},
                     reuse_actors=True,
                     local_dir=config["local_dir"],
                     checkpoint_freq=config["checkpoint_freq"],
                     checkpoint_at_end=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, default="none")
    args, extra_args = parser.parse_known_args()
    config = get_config(args.f)

    grid_dict = dict()
    for key, val in config.items():
        if isinstance(val, dict) and val.get("grid_search", False):
            grid_dict[key] = val['grid_search']

    allNames = sorted(grid_dict)
    combinations = itertools.product(*(grid_dict[Name] for Name in allNames))

    dr = Driver(combinations, allNames, config)
    dr.run()
