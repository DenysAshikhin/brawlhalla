import os
import sys
import time

import numpy
import numpy as np
from matplotlib import pyplot as plt
from environment import BrawlEnv

import environment
import cv2 as cv
from gym import spaces
from pathlib import Path
from PIL import Image

plt.style.use('seaborn-whitegrid')


runningReward = 2.3212
epochNum = 23
runningCounter = 586

folderString = f"reward-{round(runningReward,4)}-{epochNum}-{runningCounter}"
fullString = os.getcwd() + "/replays/" + folderString
Path(fullString).mkdir(parents=True, exist_ok=True)

env = BrawlEnv()

modifier = 1
actionRewardMax = 1
rewardAmount = 0.003
actions_per_second = 5
lastAction = time.time()

initial = time.time()
reward = 0

x = 320
y = 240

while time.time() - initial < 20:
    elapsedTime = time.time() - lastAction
    env.getObservation()
    # actionRewards = elapsedTime * rewardAmount * actions_per_second
    #
    # if actionRewards < actionRewardMax:
    #     reward += actionRewards
    lastAction = time.time()
    time.sleep(0.2)

index = 1
# print(f"Size in mem: {len(env.images) * sys.getsizeof(env.images[0])/(10**6)}")
# for image in env.images:
#     print(image.shape)
#     # image = numpy.reshape(y, x)
#     start = time.time()
#
#     cv.imwrite(fullString +  f"/{index}.png", image*255)
#     print(f"Elapsed Time {time.time() - start}")
#     # im = Image.fromarray(image[0])
#     # print(fullString + f"/{index}.png")
#     index += 1
#     # im.save(fullString + f"/{index}.png")

# clip = ImageSequenceClip(env.images, fps = 6)
# clip.write_videofile(fullString + "/video.mp4")


# start = time.time()
# fourcc = cv.VideoWriter_fourcc('M','J','P','G')
# video = cv.VideoWriter(fullString + '/video.avi', fourcc, 4.4, (x,y),False)
#
#
#
# for img in env.images:
#     img = img * 255.0
#     video.write(img.astype('uint8'))
# video.release()
# print(f"Elapsed Time {time.time() - start}")
#
#
#




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

# health = np.linspace(0.0, 1.0, num=25)
# health = np.flip(health)
# print(health)
# punishment = np.linspace(0,1,num=25)
#
# for i in range(25):
#     punishment[i] = environment.sigmoidHP(health[i])
#
# print(punishment)
#
# fig = plt.figure()
# ax = plt.axes()
#
# x = np.linspace(0, 10, 1000)
# plt.plot(health, punishment)
# plt.show()




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


