import gym
from gym.spaces import Box, Discrete, Tuple
import logging
import random
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Agent has to traverse the maze from the starting position S -> F
# Observation space [x_pos, y_pos, wind_direction]
# Action space: stay still OR move in current wind direction
MAP_DATA = """
###############
#      S      #
#             #
#             #
#             #
#             #
# # # ### # # #
#             #
#             #
#             #
#             #
#             #
#             #
#      F      #
###############"""


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
        self.im_size = 42
        if not env_config.get("compact", False):
            self.observation_space = Box(0, 255, shape=(self.im_size, self.im_size, 3), dtype=np.uint8)
            self.spatial = True
        else:
            self.observation_space = Box(0, 1, shape=(2,), dtype=np.float32)
            self.spatial = False
        self.action_space = Discrete(5)  # whether to move or not
        self.viewer = None

        self.bg = 255 * np.ones((self.x_dim, self.y_dim, 3), dtype=np.uint8)
        for ridx in range(self.x_dim):
            for cidx in range(self.y_dim):
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
        if self.spatial:
            return self._get_image(resize=True)
        return np.array(self.pos) / np.maximum(self.x_dim, self.y_dim)

    def step(self, action, verbose=False):
        self.pos = self._get_new_pos(self.pos, action)
        self.num_steps += 1
        at_goal = self.pos == self.end_pos
        done = at_goal or self.num_steps >= 100
        if verbose:
            print(f"step: {self.num_steps}, pos: {self.pos}")
        # return (np.array(self.pos), float(10 * int(at_goal)), done, {})
        reward = 1. if at_goal else 0.
        if self.spatial:
            obs = self._get_image(resize=True)
        else:
            obs = np.array(self.pos) / np.maximum(self.x_dim, self.y_dim)
        return obs, reward, done, {}

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

    def member_color(self, color_idx):
        if color_idx == 0:
            return [0, 255, 0]  # green
        elif color_idx == 1:
            return [255, 0, 0]  # red
        elif color_idx == 2:
            return [255, 128, 0]  # orange
        elif color_idx == 3:
            return [255, 255, 0]  # yellow
        elif color_idx == 4:
            return [128, 255, 0]  # green
        else:
            raise ValueError

    def _get_image(self, resize=False, color_idx=0, s=None):
        s = s or self.im_size
        frame_t = self.bg.copy()
        color = self.member_color(color_idx)
        frame_t[self.pos] = color
        if resize:
            frame_t = cv2.resize(frame_t, dsize=(s, s), interpolation=cv2.INTER_NEAREST)
        return frame_t

    def _get_dynamic_mask(self):
        frame = np.zeros_like(self.bg.copy())
        frame[self.pos] = [1, 1, 1]
        return frame

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
