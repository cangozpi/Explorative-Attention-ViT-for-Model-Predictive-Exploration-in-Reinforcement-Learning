import gym
import cv2

import numpy as np

from abc import abstractmethod
from collections import deque
from copy import copy

import gym_super_mario_bros
# from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv # line below is the new way to import: https://github.com/uvipen/Super-mario-bros-A3C-pytorch/issues/3
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

from torch.multiprocessing import Process

from model import *
from config import *
from PIL import Image
import matplotlib.pyplot as plt

from utils import Logger
import time

import torch.distributed as dist
from dist_utils import get_dist_info

train_method = default_config['TrainMethod']
max_step_per_episode = int(default_config['MaxStepPerEpisode'])


class Environment(Process):
    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def reset(self, seed):
        pass

    @abstractmethod
    def pre_proc(self, x):
        pass

    @abstractmethod
    def get_init_state(self, x):
        pass


def unwrap(env):
    if hasattr(env, "unwrapped"):
        return env.unwrapped
    elif hasattr(env, "env"):
        return unwrap(env.env)
    elif hasattr(env, "leg_env"):
        return unwrap(env.leg_env)
    else:
        return env


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, is_render, skip=4):
        """
        It returns only every skip-th frame. Action are repeated and rewards are sum for the skipped frames.
        It also takes element-wise maximum over the last two consecutive frames, which helps algorithm deal with the 
        problem of how certain Atari games only render their sprites every other game frame.
        """
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip
        self.is_render = is_render

        # custom rendering:
        if self.is_render:
            self.env.reset()
            plt.ion()
            rendering_view = self.env.render()
            self.ax = plt.imshow(rendering_view)

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        for i in range(self._skip):
            obs, reward, done, trun, info = self.env.step(action)
            if self.is_render:
                # custom rendering:
                rendering_view = self.env.render()
                self.ax.set_data(rendering_view)
                plt.xticks([])
                plt.yticks([])
                plt.axis
                plt.pause(1/60)


            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done or trun:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, trun, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if self.is_render:
            # custom rendering:
            rendering_view = self.env.render()
            self.ax.set_data(rendering_view)
            plt.xticks([])
            plt.yticks([])
            plt.axis
            plt.pause(1/60)

        return obs, info


class MaxStepPerEpisodeWrapper(gym.Wrapper):
    def __init__(self, env, max_step_per_episode):
        """
        Let's you call env.step() max_step_per_episode many times before returning done=(False | True), truncated=True
        """
        super(MaxStepPerEpisodeWrapper, self).__init__(env)
        self.max_step_per_episode = max_step_per_episode
        self.steps = 0

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.steps += 1
        if self.max_step_per_episode <= self.steps:
            # done = False
            truncated = True

        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        self.steps = 0
        return self.env.reset(**kwargs)

class FrameStackWrapper(gym.Wrapper):
    def __init__(self, env, history_size):
        super().__init__(env)
        assert history_size > 1, "history size must be higher than 1"
        self.history_size = history_size
        self.history = np.zeros((history_size, ) + self.env.observation_space.shape)
        h, w = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(h, w, history_size), dtype=np.uint8)

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)

        self.history[:(self.history_size - 1)] = self.history[1:, :, :]
        self.history[(self.history_size - 1)] = state

        return self.history, reward, done, truncated, info

    def reset(self, **kwargs):
        state, info =  self.env.reset(**kwargs)
        for i in range(self.history_size):
            self.history[i] = state
        return self.history, info


class StickyActionWrapper(gym.Wrapper):
    def __init__(self, env, p):
        super().__init__(env)
        self.last_action = 0
        self.p = p

    def step(self, action):
        if np.random.rand() <= self.p:
            action = self.last_action
        self.last_action = action
        return self.env.step(action)

    def reset(self, **kwargs):
        self.last_action = 0 
        return self.env.reset(**kwargs)

class ResizeAndGrayScaleWrapper(gym.Wrapper):
    """
    Resize image and Convert to Grayscale from RGB.
    """
    def __init__(self, env, h, w):
        super().__init__(env)
        self.h = h
        self.w = w
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.h, self.w), dtype=np.uint8)

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        state = self.pre_proc(state)
        return state, reward, done, truncated, info

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        state = self.pre_proc(state)
        return state, info

    def pre_proc(self, X):
        """
        convert to grayscale and rescale the image
        """
        assert X.shape[-1] == 3 # [H, W, 3]
        # x = np.array(Image.fromarray(X).convert('L').resize((self.w, self.h))).astype('float32')
        X = np.array(Image.fromarray(X).convert('L')).astype('float32')
        x = cv2.resize(X, (self.h, self.w))
        return x


class MontezumaInfoWrapper(gym.Wrapper):
    def __init__(self, env, room_address):
        super(MontezumaInfoWrapper, self).__init__(env)
        self.room_address = room_address
        self.visited_rooms = set()

    def get_current_room(self):
        ram = unwrap(self.env).ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def step(self, action):
        obs, rew, done, trun, info = self.env.step(action)
        self.visited_rooms.add(self.get_current_room())

        if done or trun:
            if 'episode' not in info:
                info['episode'] = {}
            info['episode'].update(visited_rooms=copy(self.visited_rooms))
            self.visited_rooms.clear()
        return obs, rew, done, trun, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class AtariEnvironment(Environment):
    def __init__(
            self,
            env_id,
            is_render,
            env_idx,
            history_size=4,
            h=84,
            w=84,
            life_done=True,
            sticky_action=True,
            p=0.25,
            seed=42,
            logger:Logger=None,
            child_conn=None):
        super(AtariEnvironment, self).__init__()
        assert child_conn is not None
        self.child_conn = child_conn
        self.daemon = True

        self.GLOBAL_WORLD_SIZE, self.GLOBAL_RANK, self.LOCAL_WORLD_SIZE, self.LOCAL_RANK = get_dist_info()

        self.env = gym.make(env_id, render_mode="rgb_array" if is_render else None)
        # if sticky_action:
        #     self.env = StickyActionWrapper(self.env, p)
        self.env = MaxAndSkipEnv(self.env, is_render, skip=4)
        if sticky_action:
            self.env = StickyActionWrapper(self.env, p)
        self.env = ResizeAndGrayScaleWrapper(self.env, h, w)
        # self.env = MaxAndSkipEnv(self.env, is_render, skip=4)
        self.env = FrameStackWrapper(self.env, history_size)
        self.env = MaxStepPerEpisodeWrapper(self.env, max_step_per_episode)
        self.env = Monitor(self.env)
        if 'Montezuma' in env_id:
            self.env = MontezumaInfoWrapper(self.env, room_address=3 if 'Montezuma' in env_id else 1)
        assert self.env.observation_space.shape == (h, w, history_size)

        self.logger = logger
        self.env_id = env_id
        self.env_idx = env_idx
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)

        self.action_dim = self.env.action_space.shape

        self.seed = seed
        # self.env.seed = seed
        # self.reset()
        self.reset(seed=seed)
        # self.env.seed(seed)

    # from torch.distributed.elastic.multiprocessing.errors import record
    # @record
    def run(self):
        super(AtariEnvironment, self).run()
        while True:
            action = self.child_conn.recv()
            # assert ...

            if 'Breakout' in self.env_id: # TODO: not sure why other implementations do this. Find it out
                action += 1

            state, reward, done, trun, info = self.env.step(action)

            self.rall += reward

            if done or trun:
                self.recent_rlist.append(self.rall)


                if self.logger is not None:
                    if 'Montezuma' in self.env_id:
                        self.logger.log_msg_to_both_console_and_file(f'[Rank: {self.GLOBAL_RANK}] episode: {self.episode}, step: {self.env.steps}, undiscounted_return: {self.rall}, moving_average_undiscounted_return: {np.mean(self.recent_rlist)}, visited_rooms: {info.get("episode", {}).get("visited_rooms", {})}')
                    else:
                        self.logger.log_msg_to_both_console_and_file(f'[Rank: {self.GLOBAL_RANK}] episode: {self.episode}, step: {self.env.steps}, undiscounted_return: {self.rall}, moving_average_undiscounted_return: {np.mean(self.recent_rlist)}')

                # state, _info = self.reset()
                state, _info = self.reset(seed=self.seed)

            self.child_conn.send([state, reward, done, trun, info.get("episode", {}).get("visited_rooms", {})])

            if done or trun:
                if 'Montezuma' in self.env_id:
                    self.child_conn.send([len(info['episode']['visited_rooms']), info.get("episode", {}).get("visited_rooms", {})])
                self.child_conn.send([info['episode']['undiscounted_episode_return'], info['episode']['l'], info['episode']['num_finished_episodes']])


            # dist.gather_object(info, None, dst=0)

    def reset(self, **kwargs):
        self.episode += 1
        self.rall = 0
        state, info = self.env.reset(**kwargs)
        # state, info = self.env.reset()
        # self.env.seed(self.seed)
        return state, info


class MarioEnvironment(Environment):
    def __init__(
            self,
            env_id,
            is_render,
            env_idx,
            history_size=4,
            life_done=False,
            h=84,
            w=84, movement=COMPLEX_MOVEMENT, 
            sticky_action=True,
            p=0.25,
            seed=42,
            logger:Logger=None,
            child_conn=None):
        super(MarioEnvironment, self).__init__()
        self.child_conn = child_conn
        self.daemon = True

        self.GLOBAL_WORLD_SIZE, self.GLOBAL_RANK, self.LOCAL_WORLD_SIZE, self.LOCAL_RANK = get_dist_info()

        self.env = gym_super_mario_bros.make(env_id, apply_api_compatibility=True, render_mode = "rgb_array" if is_render else None)
        JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs) # See: https://stackoverflow.com/questions/76509663/typeerror-joypadspace-reset-got-an-unexpected-keyword-argument-seed-when-i
        self.env = JoypadSpace(self.env, COMPLEX_MOVEMENT)
        if sticky_action:
            self.env = StickyActionWrapper(self.env, p)
        self.env = ResizeAndGrayScaleWrapper(self.env, h, w)
        self.env = MaxAndSkipEnv(self.env, is_render, skip=4)
        self.env = FrameStackWrapper(self.env, history_size)
        self.env = MaxStepPerEpisodeWrapper(self.env, max_step_per_episode)
        self.env = Monitor(self.env)
        assert self.env.observation_space.shape == (h, w, history_size)

        self.logger = logger
        self.env_idx = env_idx
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)

        self.life_done = life_done
        self.h = h
        self.w = w

        self.action_dim = self.env.action_space.shape

        self.seed = seed
        # self.env.seed = seed
        # self.reset()
        self.reset(seed=seed)
        # self.env.seed(seed)


    def run(self):
        super(MarioEnvironment, self).run()
        while True:
            action = self.child_conn.recv()
            # assert ...

            state, reward, done, trun, info = self.env.step(action)

            # reward range -15 ~ 15
            reward /= 15
            self.rall += reward

            # when Mario loses life, changes the state to the terminal
            # state.
            if self.life_done:
                if self.lives is None:
                    self.lives = info['life']
                elif self.lives > info['life'] and info['life'] > 0:
                    done = True
                    self.lives = info['life']


            # r_ = int(info.get('flag_get', False)) #TODO: not sure how to use this

            if done or trun:
                self.recent_rlist.append(self.rall)

                if self.logger is not None:
                    self.logger.log_msg_to_both_console_and_file(f'[Rank: {self.GLOBAL_RANK}] episode: {self.episode}, step: {self.env.steps}, undiscounted_return: {self.rall}, moving_average_undiscounted_return: {np.mean(self.recent_rlist)}, visited_rooms: {info.get("episode", {}).get("visited_rooms", {})}, stage: {info["stage"]}, current_x: {info["x_pos"]}, max_x: {self.max_pos}')

                # state, _info = self.reset()
                state, _info = self.reset(seed=self.seed)

            self.child_conn.send([state, reward, done, trun])

            if done or trun:
                self.child_conn.send([info['episode']['undiscounted_episode_return'], info['episode']['l'], info['episode']['num_finished_episodes']])


    def reset(self, **kwargs):
        self.episode += 1
        self.rall = 0
        self.lives = None
        self.stage = 1
        self.max_pos = 0
        state, info = self.env.reset(**kwargs)
        # state, info = self.env.reset()
        # self.env.seed(self.seed)
        return state, info


    
class RGBArrayAsObservationWrapper(gym.Wrapper):
    """
    Uses env.render(rgb_array) as observation (image observations)
    rather than the observation environment provides.
    """
    def __init__(self, env):
        super(RGBArrayAsObservationWrapper, self).__init__(env)
        self.env.reset()
        dummy_obs = env.render()
        self.observation_space  = gym.spaces.Box(low=0, high=255, shape=dummy_obs.shape, dtype=dummy_obs.dtype)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self.env.render()
        assert obs.shape[-1] == 3
        return obs, info

    def step(self, action):
        obs, reward, done, _, info = self.env.step(action)
        obs = self.env.render()
        assert obs.shape[-1] == 3
        return obs, reward, done, _, info


class ClassicControlEnvironment(Environment):
    def __init__(
            self,
            env_id,
            is_render,
            env_idx,
            history_size=4,
            h=84,
            w=84,
            life_done=True,
            sticky_action=True,
            p=0.25,
            seed=42,
            logger:Logger=None,
            child_conn=None):
        super(ClassicControlEnvironment, self).__init__()
        self.child_conn = child_conn
        self.daemon = True

        self.GLOBAL_WORLD_SIZE, self.GLOBAL_RANK, self.LOCAL_WORLD_SIZE, self.LOCAL_RANK = get_dist_info()

        self.env = gym.make(env_id, render_mode="rgb_array")
        self.env = RGBArrayAsObservationWrapper(self.env)
        # if sticky_action:
        #     self.env = StickyActionWrapper(self.env, p)
        self.env = ResizeAndGrayScaleWrapper(self.env, h, w)
        # self.env = MaxAndSkipEnv(self.env, is_render, skip=4)
        self.env = FrameStackWrapper(self.env, history_size)
        self.env = MaxStepPerEpisodeWrapper(self.env, max_step_per_episode)
        self.env = Monitor(self.env)
        assert self.env.observation_space.shape == (h, w, history_size)

        self.logger = logger
        self.env_id = env_id
        self.env_idx = env_idx
        self.episode = 0
        self.rall = 0
        self.recent_rlist = deque(maxlen=100)

        self.h = h
        self.w = w

        self.action_dim = self.env.action_space.shape

        self.seed = seed
        # self.env.seed = seed
        # self.reset()
        self.reset(seed=seed)
        # self.env.seed(seed)

    def run(self):
        super(ClassicControlEnvironment, self).run()
        while True:
            action = self.child_conn.recv()
            # assert ...


            state, reward, done, trun, info = self.env.step(action)

            self.rall += reward

            if done or trun:
                self.recent_rlist.append(self.rall)

                if self.logger is not None:
                    self.logger.log_msg_to_both_console_and_file(f'[Rank: {self.GLOBAL_RANK}] episode: {self.episode}, step: {self.env.steps}, undiscounted_return: {self.rall}, moving_average_undiscounted_return: {np.mean(self.recent_rlist)}')

                # state, _info = self.reset()
                state, _info = self.reset(seed=self.seed)

            self.child_conn.send([state, reward, done, trun])

            if done or trun:
                self.child_conn.send([info['episode']['undiscounted_episode_return'], info['episode']['l'], info['episode']['num_finished_episodes']])



    def reset(self, **kwargs):
        self.episode += 1
        self.rall = 0
        state, info = self.env.reset(**kwargs)
        # state, info = self.env.reset()
        # self.env.seed(self.seed)
        return state, info



class Monitor(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env=env)
        self.tstart = time.time()

        self.rewards = []
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0

    def reset(self, **kwargs):
        self.rewards = []
        return self.env.reset(**kwargs)

    def step(self, action):
        ob, rew, done, trun, info = self.env.step(action)
        self.rewards.append(rew)
        if done or trun:
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.tstart)
            epinfo = {"undiscounted_episode_return": round(eprew, 6), "l": eplen, "t": round(time.time() - self.tstart, 6), "num_finished_episodes": len(self.episode_rewards)}
            if "episode" not in info:
                info["episode"] = {}
            info['episode'].update(epinfo)
        self.total_steps += 1
        return (ob, rew, done, trun, info)

    def get_total_steps(self):
        return self.total_steps

    def get_episode_rewards(self):
        return self.episode_rewards

    def get_episode_lengths(self):
        return self.episode_lengths

    def get_episode_times(self):
        return self.episode_times