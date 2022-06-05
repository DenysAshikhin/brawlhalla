import MTM
import numpy
import ctypes
import win32con
import win32gui
import pygetwindow

import time
import mss
from ctypes import windll
import win32api
import os
import cv2
from PIL.Image import Image
from matplotlib import pyplot as plt

gearX = 211
gearY = 67


def loadDigits():
    root = os.path.join(os.getcwd(), "digits")
    print(os.getcwd())
    print(root)
    digitsList = []

    for i in range(len(os.listdir(root))):
        print(os.path.join(root, str(i) + ".png"))
        if os.path.isfile(os.path.join(root, str(i) + ".png")):
            img = cv2.imread(os.path.join(root, str(i) + ".png"), cv2.IMREAD_GRAYSCALE)
            digitsList.append((str(i), img))
    return digitsList


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
KEY_C = char2key('c')


def keyHold(key):
    win32api.keybd_event(key, win32api.MapVirtualKey(key, 0), 0, 0)


def keyRelease(key):
    win32api.keybd_event(key, win32api.MapVirtualKey(key, 0), win32con.KEYEVENTF_KEYUP, 0)


def countLife(img, templates):
    hits = MTM.matchTemplates(templates,
                              img,
                              method=cv2.TM_CCOEFF_NORMED,
                              N_object=float("inf"),
                              score_threshold=0.6,
                              maxOverlap=0,
                              searchBox=None)

    if len(hits['TemplateName']) == 0:
        return -1

    for index, row in hits.iterrows():
        # Multiple templates to search are used, each with different threshold value. These
        # values were determined by experimentation to give best false positive and false
        # negative values
        if row['TemplateName'] == "2" and row['Score'] >= 0.81:
            return int(row['TemplateName'])
        elif row['TemplateName'] != "0" and row['Score'] >= 0.89:
            return int(row['TemplateName'])
        elif row['TemplateName'] == "0" and row['Score'] >= 0.65:
            return int(row['TemplateName'])

    return -1


def findOffset(image):
    root = os.path.join(os.getcwd(), "offsetGear.png")
    offsetTemplate = cv2.imread(root)
    offsetTemplate = offsetTemplate[:, :, 1]

    searchImage = image[0:100, 400:640, 1]
    hits = MTM.matchTemplates([("Offset", offsetTemplate)],
                              searchImage,
                              method=cv2.TM_CCOEFF_NORMED,
                              N_object=float("inf"),
                              score_threshold=0.8,
                              maxOverlap=0,
                              searchBox=None)

    if len(hits['TemplateName']) == 0:
        print("Gear Icon Used for Template not found")
        sys.exit()

    return hits['BBox'].iloc[0]


import numpy as np
from gym import spaces

import uuid

from six.moves import queue

import threading
import time
from typing import Union, Optional

from ray.rllib.env import ExternalEnv, MultiAgentEnv, ExternalMultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict, EnvInfoDict, EnvObsType, EnvActionType

from skimage.transform import resize

x = 320
y = 240


# x = 640
# y = 480

class BrawlEnv(ExternalEnv):

    def __init__(self, config=None):
        self.templates = loadDigits()
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

        self.observation_space = spaces.Box(low=0, high=1, shape=(y, x, 1), dtype=np.float32)

        self.action_space = spaces.MultiDiscrete(
            [
                2,  # W
                2,  # A
                2,  # S
                2,  # D
                2,  # Space
                2,  # H
                2,  # J
                2,  # K
                2  # L
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

        win32gui.MoveWindow(hwnd, 0, 0, width, height, True)

        sct = mss.mss()
        time.sleep(1)

        self.sct = sct
        self.width = width
        self.height = height

        self.currentStock = 3
        self.enemyStock = 3
        self.actionsTaken = 0
        self.failedStocks = 0
        self.pressedKeys = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.lastAction = time.time()
        self.actionRewards = 0

        full_screen_all = imageGrab(x=0, w=self.width, y=0, h=self.height, grabber=self.sct)[:, :, :3]
        offSet = findOffset(imageGrab(x=0, w=self.width, y=0, h=self.height, grabber=self.sct)[:, :, :3])

        print(f"my x: {gearX} - my Y: {gearY}")
        print(f"my offset: {offSet}")

        if offSet[0] > gearX:
            self.xOffset = offSet[0] - gearX
        else:
            self.xOffset = gearX - offSet[0]

        if offSet[1] > gearY:
            self.yOffset = gearY - offSet[1]
        else:
            self.yOffset = offSet[1] - gearY

        self.yOffset *= 2
        self.xOffset = 0

        print(f"Final offsetX: {self.xOffset}")
        print(f"Final offsetY: {self.yOffset}")

        self.myStockX = 547 + self.xOffset
        self.enemyStockX = 585 + self.xOffset

        self.stockY = 63 + self.yOffset

        self.gameOver = False

        print('got past main loop')

    def releaseAllKeys(self, Force=False):

        # print('reseting actions: ')
        #
        # actions = np.array(self.pressedKeys)
        # print(actions)
        #
        # if actions[0] == 1 or Force:
        #     keyRelease(KEY_W)
        #     actions[0] = 0
        #     print('releasing W')
        #
        # if actions[1] == 1 or Force:
        #     keyRelease(KEY_A)
        #     print('releasing A')
        #     actions[1] = 0
        #
        # if actions[2] == 1 or Force:
        #     keyRelease(KEY_S)
        #     print('releasing S')
        #     actions[2] = 0
        #
        # if actions[3] == 1 or Force:
        #     keyRelease(KEY_D)
        #     print('releasing D')
        #     actions[3] = 0
        #
        # if actions[4] == 1 or Force:
        #     keyRelease(KEY_SPACE)
        #     actions[4] = 0
        #     print('releasing SPACE')
        #
        # if actions[5] == 1 or Force:
        #     keyRelease(KEY_H)
        #     actions[5] = 0
        #     print('releasing H')
        #
        # if actions[6] == 1 or Force:
        #     keyRelease(KEY_J)
        #     actions[6] = 0
        #     print('releasing J')
        #
        # if actions[7] == 1 or Force:
        #     keyRelease(KEY_K)
        #     actions[7] = 0
        #     print('releasing K')
        #
        # if actions[8] == 1 or Force:
        #     keyRelease(KEY_L)
        #     actions[8] = 0
        #     print('releasing L')
        #
        # self.pressedKeys = actions
        #
        # return

        keyHold(KEY_SPACE)
        time.sleep(0.01)
        keyRelease(KEY_SPACE)

        keyHold(KEY_W)
        time.sleep(0.01)
        keyRelease(KEY_W)

        time.sleep(0.01)
        keyHold(KEY_A)
        time.sleep(0.01)
        keyRelease(KEY_A)

        time.sleep(0.01)
        keyHold(KEY_S)
        time.sleep(0.01)
        keyRelease(KEY_S)

        time.sleep(0.01)
        keyHold(KEY_D)
        time.sleep(0.01)
        keyRelease(KEY_D)

        time.sleep(0.01)
        keyHold(KEY_H)
        time.sleep(0.01)
        keyRelease(KEY_H)

        time.sleep(0.01)
        keyHold(KEY_J)
        time.sleep(0.01)
        keyRelease(KEY_J)

        time.sleep(0.01)
        keyHold(KEY_K)
        time.sleep(0.01)
        keyRelease(KEY_K)

        time.sleep(0.01)
        keyHold(KEY_L)
        time.sleep(0.01)
        keyRelease(KEY_L)

        # time.sleep(0.01)
        # keyHold(KEY_C)
        # time.sleep(0.01)
        # keyRelease(KEY_C)

    def startInitialGame(self):

        return None

    def resetValues(self):
        self.enemyStock = 3
        self.currentStock = 3
        self.actionsTaken = 0
        self.gameOver = False
        self.failedStocks = 0
        self.pressedKeys = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.lastAction = time.time()
        self.actionRewards = 0

    def restartMatch(self):

        for i in range(8):
            keyHold(KEY_C)
            time.sleep(0.1)
            keyRelease(KEY_C)
            time.sleep(1.75)

    def restartRound(self):

        # self.releaseAllKeys()
        self.resetValues()
        self.restartMatch()

        print('finished reseting the game')

    def getObservation(self):

        rgb_weights = [0.1140, 0.5870, 0.2989]

        full_screen_all = imageGrab(x=0, w=self.width, y=0, h=self.height, grabber=self.sct)[:, :, :3]

        grayscale_image = np.int_(np.dot(full_screen_all[..., :3], rgb_weights))

        # Crop out stocks from full screen, format: top y cord, bot y cord, left x cord, right x cord
        my_stock_img = grayscale_image[self.stockY:self.stockY + 12, self.myStockX:self.myStockX + 10]
        enemy_stock_img = grayscale_image[self.stockY:self.stockY + 12, self.enemyStockX:self.enemyStockX + 10]

        # # Extract one channel green channel, screen capture goes BGR from stocks
        # my_stock_img = my_stock_img
        # # plt.subplot(1, 1, 1), plt.imshow(my_stock_img, 'gray', vmin=0, vmax=255)
        # # plt.show()
        # enemy_stock_img = enemy_stock_img
        # # plt.subplot(1, 1, 1) jl, plt.imshow(enemy_stock_img, 'gray', vmin=0, vmax=255)
        # # plt.show()

        my_stock = countLife(my_stock_img, self.templates)
        enemy_stock = countLife(enemy_stock_img, self.templates)

        grayscale_image = grayscale_image / 255.0
        grayscale_image = resize(grayscale_image, (y, x))
        # print(grayscale_image.shape)
        grayscale_image = numpy.reshape(grayscale_image, grayscale_image.shape + (1,))
        # print(grayscale_image.shape)
        print(f"my stock: {my_stock} - enemy stock: {enemy_stock}")

        reward = 0

        gameOver = False
        forceEnd = False

        if my_stock != -1 and enemy_stock != -1:

            if my_stock < self.currentStock:
                reward -= 0.33
                self.currentStock = my_stock

            if enemy_stock < self.enemyStock:
                reward += 0.33
                self.enemyStock = enemy_stock

            self.failedStocks = 0
        elif my_stock == -1 and enemy_stock == -1:
            forceEnd = True
        else:
            self.failedStocks = self.failedStocks + 1

        if self.gameOver == False:
            if enemy_stock == 0:
                self.gameOver = True
                reward += 1

            if my_stock == 0:
                self.gameOver = True
                reward -= 1

        # Emergency breaker to just kill the game
        elif self.failedStocks > 13 or forceEnd == True:
            self.gameOver = True

            if self.enemyStock < self.currentStock:
                reward += 1
                reward += 0.33 * self.currentStock
            elif self.currentStock < self.enemyStock:
                reward -= 1
                reward -= 0.33 * self.enemyStock

        modifier = 1
        actionRewardMax = 1.51
        actionPerSecond = 0.0125  # 120 seconds * 0.0125 = 1.5. Max negative is -3
        elapsedTime = time.time() - self.lastAction

        # rewardAmount = 0.003
        # actions_per_second = 5
        # actionRewards = elapsedTime * rewardAmount * actions_per_second
        actionRewards = elapsedTime * actionPerSecond

        if self.actionRewards < actionRewardMax:
            reward += actionRewards
            self.actionRewards += actionRewards

        self.lastAction = time.time()
        # if self.actionsTaken < (500 * modifier):
        #     reward += (rewardAmount / modifier)
        #     self.actionsTaken = self.actionsTaken + 1

        return grayscale_image, reward, self.gameOver

    def act(self, actions):

        self.pressedKeys = actions

        delay = 0.005

        if actions[0] == 0:
            keyRelease(KEY_W)
        else:
            keyHold(KEY_W)
        time.sleep(delay)
        if actions[1] == 0:
            keyRelease(KEY_A)
        else:
            keyHold(KEY_A)
        time.sleep(delay)
        if actions[2] == 0:
            keyRelease(KEY_S)
        else:
            keyHold(KEY_S)
        time.sleep(delay)
        if actions[3] == 0:
            keyRelease(KEY_D)
        else:
            keyHold(KEY_D)
        time.sleep(delay)
        if actions[4] == 0:
            keyRelease(KEY_SPACE)
        else:
            keyHold(KEY_SPACE)
        time.sleep(delay)
        if actions[5] == 0:
            keyRelease(KEY_H)
        else:
            keyHold(KEY_H)
        time.sleep(delay)
        if actions[6] == 0:
            keyRelease(KEY_J)
        else:
            keyHold(KEY_J)
        time.sleep(delay)
        if actions[7] == 0:
            keyRelease(KEY_K)
        else:
            keyHold(KEY_K)
        time.sleep(delay)
        if actions[8] == 0:
            keyRelease(KEY_L)
        else:
            keyHold(KEY_L)
        time.sleep(delay)
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
