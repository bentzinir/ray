import gym
from gym.spaces import Box, Discrete, Tuple
import logging
import random
import numpy as np

logger = logging.getLogger(__name__)

# Agent has to traverse the maze from the starting position S -> F
# Observation space [x_pos, y_pos, wind_direction]
# Action space: stay still OR move in current wind direction
MAP_DATA = """
###########################
#            S            #
#           ###           #
#           ###           #
#           ###           #
#           ###           #
#           ###           #
#           ###           #
#           ###           #
#           ###           #
#           ###           #
#           ###           #
#           ###           #
#            F            #
###########################"""


class MazeEnv(gym.Env):
    def __init__(self, env_config={}):
        self.map = [m for m in MAP_DATA.split("\n") if m]
        self.x_dim = len(self.map)
        self.y_dim = len(self.map[0])
        logger.info("Loaded map {} {}".format(self.x_dim, self.y_dim))
        for x in range(self.x_dim):
            for y in range(self.y_dim):
                if self.map[x][y] == "S":
                    self.start_pos = (x, y)
                elif self.map[x][y] == "F":
                    self.end_pos = (x, y)
        logger.info("Start pos {} end pos {}".format(self.start_pos,
                                                     self.end_pos))
        # self.observation_space = Tuple([
        #     Box(0, 100, shape=(2, )),  # (x, y)
        #     Discrete(4),  # wind direction (N, E, S, W)
        # ])
        self.observation_space = Box(0, 100, shape=(2,))
        self.action_space = Discrete(5)  # whether to move or not
        self.viewer = None
        self.h = len(self.map)
        self.w = len(self.map[0])
        self.bg = 255 * np.ones((self.h, self.w, 3), dtype=np.uint8)
        for ridx in range(self.h):
            for cidx in range(self.w):
                if self.map[ridx][cidx] == "#":
                    self.bg[ridx, cidx, :] = [255, 0, 0]
        self.bg[self.end_pos] = [0, 0, 255]
        self.member = None
        self.member_list = []
        self.frames = dict()

    def reset(self):
        # self.wind_direction = random.choice([0, 1, 2, 3])
        self.pos = self.start_pos
        self.num_steps = 0
        return np.array(self.pos)

    def step(self, action, verbose=False):
        # if action == 1:
        #     self.pos = self._get_new_pos(self.pos, self.wind_direction)

        # self.wind_direction = random.choice([0, 1, 2, 3])
        self.pos = self._get_new_pos(self.pos, action)
        self.num_steps += 1
        at_goal = self.pos == self.end_pos
        done = at_goal or self.num_steps >= 100
        if verbose:
            print(f"step: {self.num_steps}, pos: {self.pos}")
        # return (np.array(self.pos), float(10 * int(at_goal)), done, {})
        reward = 100. if at_goal else 0.
        return np.array(self.pos), reward, done, {}

    def _get_new_pos(self, pos, direction):
        new_pos = pos
        if direction == 0:
            new_pos = (pos[0] - 1, pos[1])
        elif direction == 1:
            new_pos = (pos[0], pos[1] + 1)
        elif direction == 2:
            new_pos = (pos[0] + 1, pos[1])
        elif direction == 3:
            new_pos = (pos[0], pos[1] - 1)
        if (new_pos[0] >= 0 and new_pos[0] < self.x_dim and new_pos[1] >= 0 and new_pos[1] < self.y_dim
                and self.map[new_pos[0]][new_pos[1]] != "#"):
            return new_pos
        else:
            return pos  # did not move

    def set_member(self, member):
        self.member = member
        self.member_list = list(set(self.member_list + [member]))
        self.member_list.sort()

    def member_color(self):
        if self.member == 0:
            return [255, 0, 0]  # red
        elif self.member == 1:
            return [255, 128, 0]  # orange
        elif self.member == 2:
            return [255, 255, 0]  # yellow
        elif self.member == 3:
            return [128, 255, 0]  # green
        else:
            raise ValueError

    def _get_image(self, alpha=0.995):
        frame_t = self.bg.copy()
        frame_t[self.pos] =  [0, 255, 0] #self.member_color()
        # frame[self.end_pos] = [0, 0, 255]
        if not self.member in self.frames:
            self.frames[self.member] = self.bg.copy()

        self.frames[self.member] = (alpha * self.frames[self.member] + (1 - alpha) * frame_t).astype(np.uint8)
        # self.frame[self.pos] = self.member_color()
        frames = [frame_t] + [self.frames[i] for i in self.member_list]
        return np.concatenate(frames, axis=0)

    def render(self, mode='human'):
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen
