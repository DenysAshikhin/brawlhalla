import time

import numpy
import numpy as np
from matplotlib import pyplot as plt
from environment import BrawlEnv
import cv2 as cv
from gym import spaces


# x = 4
# y = 2
# spacy = spaces.Box(low=0, high=1, shape=(y, x), dtype=np.float32)
#
# print(spacy)
#
# testy = numpy.array([[0.5,0.5,0.5,0.5],
#                      [0.554534643,0.5,0.5,0.5]])
# print(testy)
#
# print(spacy.contains(testy))


env = BrawlEnv()
#

# print(obs[120][120])
#
#
#
# from PIL import ImageGrab
#
#
# x0 = 0
# y0 = 0
# xoffset = 0
# yoffset = 0
# w = 640
# h = 480
# image = ImageGrab.grab(bbox=(x0 + xoffset, y0 + yoffset, x0 + w + xoffset, y0 + h + yoffset))
# image_arr = numpy.array(image)
#
# # print(image)
# print(image_arr[120][120])
#
#
#
# cv.imwrite('mss.png', obs)


modifier = 1
actionRewardMax = 1
rewardAmount = 0.003
actions_per_second = 5
lastAction = time.time()

initial = time.time()
reward = 0

while time.time() - initial < 100000:
    elapsedTime = time.time() - lastAction

    env.getObservation()[0]
    # actionRewards = elapsedTime * rewardAmount * actions_per_second
    #
    # if actionRewards < actionRewardMax:
    #     reward += actionRewards
    lastAction = time.time()
    time.sleep(0.1)


print(f"final reward: {reward}")
