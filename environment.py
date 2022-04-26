import time

import cv2 as cv
import numpy
import numpy as np
from matplotlib import pyplot as plt
# from pynput.mouse import Button, Controller
import ctypes
import sys
import time

import pywintypes
import win32con
import win32gui
from PIL import ImageGrab
import pygetwindow
import pyautogui as auto

import mouse

import mss
from ctypes import windll
import win32api

def imageGrab(x=0, y=0, w=0, h=0, grabber=None):

    image = numpy.array(grabber.grab({"top": y, "left": x, "width": w, "height": h}))
    return image

def char2key(c):
    # https://msdn.microsoft.com/en-us/library/windows/desktop/ms646329(v=vs.85).aspx
    # print(f"The ord of {c}: {ord(c)}")
    result = 0
    if c == '[SPACE]':
        result = windll.User32.VkKeyScanW(32)
    else:
        result = windll.User32.VkKeyScanW(ord(c))
    shift_state = (result & 0xFF00) >> 8
    vk_key = result & 0xFF

    return vk_key

KEY_W = char2key('w')
KEY_A = char2key('a')
KEY_S = char2key('s')
KEY_D = char2key('d')

KEY_SPACE = char2key('[SPACE]')

KEY_H = char2key('h')
KEY_J = char2key('j')
KEY_K = char2key('k')
KEY_L = char2key('l')

def keyHold(key):

    win32api.keybd_event(key, win32api.MapVirtualKey(key, 0), 0, 0)

def keyRelease(key):

    win32api.keybd_event(key, win32api.MapVirtualKey(key, 0), win32con.KEYEVENTF_KEYUP, 0)





import os
from tkinter import Tk
import numpy as np
from gym import spaces
from PIL import ImageTk, Image
import gym

# from tkinter import Frame, Tk, Label

# from mttkinter import mtTkinter


from ray.rllib.agents.ppo import DDPPOTrainer, ppo
from ray.tune import register_env

from ray import tune

from ray.rllib.utils.typing import EnvActionType, EnvObsType, EnvInfoDict
import threading
import uuid
from typing import Optional

from six.moves import queue
import ray
from ray import serve

import requests
import logging
import threading
import time
from typing import Union, Optional

import ray.cloudpickle as pickle
from ray.rllib.env import ExternalEnv, MultiAgentEnv, ExternalMultiAgentEnv
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import MultiAgentDict, EnvInfoDict, EnvObsType, EnvActionType


class BrawlEnv(ExternalEnv):

    def __init__(self, config=None):

        threading.Thread.__init__(self)

        print(config)
        if config is not None:
            if 'sleep' in config:
                self.sleep = config['sleep']
            else:
                self.sleep = False
        else:
            self.sleep = False

        self.daemon = True

        self.observation_space = spaces.Box(low=0, high=255,
                                                               shape=(480, 640, 3), dtype=np.uint8)

        self.action_space = spaces.MultiDiscrete(
            [
                2, # W
                2, # A
                2, # S
                2, # D
                2, # Space
                2, # H
                2, # J
                2, # K
                2 # L
            ]
        )
        self._episodes = {}
        self._finished = set()
        self._results_avail_condition = threading.Condition()
        self._max_concurrent_episodes = 1  # maybe maybe not, no clue lmao

        windows = pygetwindow.getWindowsWithTitle('Brawlhalla')
        win = None

        for window in windows:
            if window.title == 'Brawlhalla':
                win = window

        width = 640
        height = 480

        win.size = (width, height)
        win.moveTo(0, 0)
        win.activate()
        ctypes.windll.user32.SetProcessDPIAware()

        hwnd = win32gui.FindWindow(None, 'Brawlhalla')
        win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0,
                              win32con.SWP_SHOWWINDOW | win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)

        sct = mss.mss()
        time.sleep(1)

        self.sct = sct
        self.width = width
        self.height = height



        print('got past main loop')

    def getObservatioN(self):
        full_screen_all = imageGrab(x=0, w=self.width, y=0, h=self.height, grabber=self.sct)
        return full_screen_all[:, :, :3]

    def act(self, actions):
        print("Got following actions:")
        print(actions)
        return 0

    def run(self):  # if I can't get this to work, try not overriding it in the first place?
        """Override this to implement the run loop.
        Your loop should continuously:
            1. Call self.start_episode(episode_id)
            2. Call self.get_action(episode_id, obs)
                    -or-
                    self.log_action(episode_id, obs, action)
            3. Call self.log_returns(episode_id, reward)
            4. Call self.end_episode(episode_id, obs)
            5. Wait if nothing to do.
        Multiple episodes may be started at the same time.
        """
        while True:
            time.sleep(0.25)
        #
        # if self.sleep:
        #     time.sleep(999999)
        #
        # episode_id = None
        # episode_id = self.start_episode(episode_id=episode_id)
        # while True:  # not sure if it should be a literal loop..........?
        #     gameObservation = self.underlord.getObservation()
        #     self.root.update()
        #
        #     action = self.get_action(episode_id=episode_id, observation=gameObservation)
        #
        #     reward = self.underlord.act(action=action[0], x=action[1], y=action[2], selection=action[3])
        #     self.log_returns(episode_id=episode_id, reward=reward)
        #
        #     if self.underlord.finished() != -1:
        #         self.end_episode(episode_id=episode_id, observation=gameObservation)
        #         episode_id = self.start_episode(episode_id=None)

    def start_episode(self,
                      episode_id: Optional[str] = None,
                      training_enabled: bool = True) -> str:
        """Record the start of an episode.
        Args:
            episode_id (Optional[str]): Unique string id for the episode or
                None for it to be auto-assigned and returned.
            training_enabled (bool): Whether to use experiences for this
                episode to improve the policy.
        Returns:
            episode_id (str): Unique string id for the episode.
        """

        print('start episode?')
        if episode_id is None:
            episode_id = uuid.uuid4().hex
            print('trying to call new game')
            # self.underlord.startNewGame()
            print('got past new game')

        print('got past is none episode')

        if episode_id in self._finished:
            raise ValueError(
                "Episode {} has already completed.".format(episode_id))

        if episode_id in self._episodes:
            raise ValueError(
                "Episode {} is already started".format(episode_id))

        self._episodes[episode_id] = _ExternalEnvEpisode(
            episode_id, self._results_avail_condition, training_enabled)

        return episode_id

    def get_action(self, episode_id: str,
                   observation: EnvObsType) -> EnvActionType:
        """Record an observation and get the on-policy action.
        Args:
            episode_id (str): Episode id returned from start_episode().
            observation (obj): Current environment observation.
        Returns:
            action (obj): Action from the env action space.
        """

        episode = self._get(episode_id)
        return episode.wait_for_action(observation)

    def log_returns(self,
                    episode_id: str,
                    reward: float,
                    info: EnvInfoDict = None) -> None:
        """Record returns from the environment.
        The reward will be attributed to the previous action taken by the
        episode. Rewards accumulate until the next action. If no reward is
        logged before the next action, a reward of 0.0 is assumed.
        Args:
            episode_id (str): Episode id returned from start_episode().
            reward (float): Reward from the environment.
            info (dict): Optional info dict.
        """

        episode = self._get(episode_id)
        episode.cur_reward += reward

        if info:
            episode.cur_info = info or {}

    def end_episode(self, episode_id: str, observation: EnvObsType) -> None:
        """Record the end of an episode.
        Args:
            episode_id (str): Episode id returned from start_episode().
            observation (obj): Current environment observation.
        """

        # self.underlord.returnToMainScreen()
        episode = self._get(episode_id)
        self._finished.add(episode.episode_id)
        episode.done(observation)

    def _get(self, episode_id: str) -> "_ExternalEnvEpisode":
        """Get a started episode or raise an error."""

        if episode_id in self._finished:
            raise ValueError(
                "Episode {} has already completed.".format(episode_id))

        if episode_id not in self._episodes:
            raise ValueError("Episode {} not found.".format(episode_id))

        return self._episodes[episode_id]


class _ExternalEnvEpisode:
    """Tracked state for each active episode."""

    def __init__(self,
                 episode_id: str,
                 results_avail_condition: threading.Condition,
                 training_enabled: bool,
                 multiagent: bool = False):

        self.episode_id = episode_id
        self.results_avail_condition = results_avail_condition
        self.training_enabled = training_enabled
        self.multiagent = multiagent
        self.data_queue = queue.Queue()
        self.action_queue = queue.Queue()
        if multiagent:
            self.new_observation_dict = None
            self.new_action_dict = None
            self.cur_reward_dict = {}
            self.cur_done_dict = {"__all__": False}
            self.cur_info_dict = {}
        else:
            self.new_observation = None
            self.new_action = None
            self.cur_reward = 0.0
            self.cur_done = False
            self.cur_info = {}

    def get_data(self):
        if self.data_queue.empty():
            return None
        return self.data_queue.get_nowait()

    def log_action(self, observation, action):
        if self.multiagent:
            self.new_observation_dict = observation
            self.new_action_dict = action
        else:
            self.new_observation = observation
            self.new_action = action
        self._send()
        self.action_queue.get(True, timeout=60.0)

    def wait_for_action(self, observation):
        if self.multiagent:
            self.new_observation_dict = observation
        else:
            self.new_observation = observation
        self._send()
        return self.action_queue.get(True, timeout=500.0)

    def done(self, observation):
        if self.multiagent:
            self.new_observation_dict = observation
            self.cur_done_dict = {"__all__": True}
        else:
            self.new_observation = observation
            self.cur_done = True
        self._send()

    def _send(self):
        if self.multiagent:
            if not self.training_enabled:
                for agent_id in self.cur_info_dict:
                    self.cur_info_dict[agent_id]["training_enabled"] = False
            item = {
                "obs": self.new_observation_dict,
                "reward": self.cur_reward_dict,
                "done": self.cur_done_dict,
                "info": self.cur_info_dict,
            }
            if self.new_action_dict is not None:
                item["off_policy_action"] = self.new_action_dict
            self.new_observation_dict = None
            self.new_action_dict = None
            self.cur_reward_dict = {}
        else:
            item = {
                "obs": self.new_observation,
                "reward": self.cur_reward,
                "done": self.cur_done,
                "info": self.cur_info,
            }
            if self.new_action is not None:
                item["off_policy_action"] = self.new_action
            self.new_observation = None
            self.new_action = None
            self.cur_reward = 0.0
            if not self.training_enabled:
                item["info"]["training_enabled"] = False

        with self.results_avail_condition:
            self.data_queue.put_nowait(item)
            self.results_avail_condition.notify()



#
#
# if __name__ == '__main__':
#     windows = pygetwindow.getWindowsWithTitle('Brawlhalla')
#     win = None
#
#     for window in windows:
#         if window.title == 'Brawlhalla':
#             win = window
#
#     width = 640
#     height = 480
#
#     win.size = (width, height)
#     win.moveTo(0, 0)
#     win.activate()
#     ctypes.windll.user32.SetProcessDPIAware()
#     hwnd = win32gui.FindWindow(None, 'Brawlhalla')
#     dimensions = win32gui.GetWindowRect(hwnd)
#
#
#     win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
#     win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
#     win32gui.SetWindowPos(hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0,
#                           win32con.SWP_SHOWWINDOW | win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
#
#     sct = mss.mss()
#     time.sleep(1)
#
#     while True:
#         time_start = time.time()
#         time_end = time.time()
#         imgs = []
#         counter = 0
#
#         while time_end-time_start < 1:
#
#             time_end=time.time()
#             full_screen_all = imageGrab(x=0, w=width, y=0, h=height, grabber=sct)
#             full_screen = full_screen_all[:, :, :3]
#             # print(full_screen)
#             # print(full_screen.shape)
#
#             my_stock = full_screen[55:55+10, 548:548+10]
#             enemy_stock = full_screen[55:55+10, 587:587+10]
#
#             # plt.subplot(1, 1, 1), plt.imshow(my_stock, 'gray', vmin=0, vmax=255)
#             # plt.show()
#
#             # keyHold(KEY_W)
#             # keyHold(KEY_H)
#             # time.sleep(0.00025)
#             # keyRelease(KEY_W)
#             # keyRelease(KEY_H)
#
#             counter += 1
#
#         print(f"FPS: {counter}")