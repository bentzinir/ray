from gym.envs.registration import register

register(
    id='maze-v0',
    entry_point='ray.rllib.examples.env.my_envs.maze.maze_env:MazeEnv',
    max_episode_steps=100000,
    kwargs={}
)

