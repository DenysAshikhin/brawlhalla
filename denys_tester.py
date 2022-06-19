import time

import numpy
import numpy as np
from matplotlib import pyplot as plt
from environment import BrawlEnv
import environment
import cv2 as cv
from gym import spaces

plt.style.use('seaborn-whitegrid')


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


# env = BrawlEnv()
# print(f" 1.13: ${environment.sigmoidHP(1.13)}")#
# print(f" 1.0: ${environment.sigmoidHP(1.0)}")
# print(f" 0.9: ${environment.sigmoidHP(0.9)}")
# print(f" 0.8: ${environment.sigmoidHP(0.8)}")
# print(f" 0.7: ${environment.sigmoidHP(0.7)}")
# print(f" 0.6: ${environment.sigmoidHP(0.6)}")
# print(f" 0.5: ${environment.sigmoidHP(0.5)}")
# print(f" 0.2: ${environment.sigmoidHP(0.2)}")

health = np.linspace(0.0, 1.0, num=25)
health = np.flip(health)
print(health)
punishment = np.linspace(0,1,num=25)

for i in range(25):
    punishment[i] = environment.sigmoidHP(health[i])

print(punishment)

fig = plt.figure()
ax = plt.axes()

x = np.linspace(0, 10, 1000)
plt.plot(health, punishment)
plt.show()




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

#
# modifier = 1
# actionRewardMax = 1
# rewardAmount = 0.003
# actions_per_second = 5
# lastAction = time.time()
#
# initial = time.time()
# reward = 0
#
# while time.time() - initial < 100000:
#     elapsedTime = time.time() - lastAction
#
#     env.getObservation()[0]
#     # actionRewards = elapsedTime * rewardAmount * actions_per_second
#     #
#     # if actionRewards < actionRewardMax:
#     #     reward += actionRewards
#     lastAction = time.time()
#     time.sleep(0.1)
#
#
# print(f"final reward: {reward}")
