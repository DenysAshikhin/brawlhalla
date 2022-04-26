from queue import Queue
from tkinter import Tk

import gym
from ray.rllib.env import PolicyClient
from ray.tune.registry import register_env

from environment import BrawlEnv
import logging
import time
import argparse

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('-ip', type=str,
                    help='IP of this device')

parser.add_argument('-speed', type=float,
                    help='gameFactor, default 1.0')

parser.add_argument('-update', type=float,
                    help='seconds how often to update from main process')

parser.add_argument('-local', type=str,
                    help='Whether to create and update a local copy of the AI (adds delay) or query server for each action.'
                         'possible values: "local" or "remote"')

# parser.add_argument('-data', type=str,
#                     help='Whether or not to log data')

args = parser.parse_args()

update = 3600.0

local = 'local'

remoteee = False

if args.update:
    update = args.update
    # remoteee = True

if args.local:
    local = args.local

if local == 'remote':
    remoteee = True

print(f"Going to update {local}-y  at {update} seconds interval")

print('trying to launch policy client')
print(f"http://{args.ip}:55556")
client = PolicyClient(address=f"http://{args.ip}:55556", update_interval=60, inference_mode=local)


forced = True
root = None

env = BrawlEnv({'sleep': True})

print('trying to get initial eid')
episode_id = client.start_episode()

# if local == 'remote':
#     env.underlord.startNewGame()

# gameObservation = env.underlord.getObservation()
reward = 0
print('starting main loop')
replayList = []

if args.speed is not None:
    print(f"multiply by {args.speed}")

    env.underlord.mouseSleepTime *= args.speed
    env.underlord.shopSleepTime *= args.speed

update = True

runningReward = 0


while True:

    gameObservation = env.getObservation()


    if not env.observation_space.contains(gameObservation):
        print(gameObservation)
        print("Not lined up 1")
        print(env.underlord.heroAlliances)
        sys.exit()


    action = None

    action = client.get_action(episode_id=episode_id, observation=gameObservation)

    reward = env.act(action)

    runningReward += reward
    # act_time = time.time() - act_time
    # print("--- %s seconds to get do action ---" % (time.time() - start_time))
    # print(f"running reward: {reward}")
    client.log_returns(episode_id=episode_id, reward=reward)
    # print('finished logging step')

    # print("--- %s seconds to get finish logging return ---" % (time.time() - start_time))

    # replayList.append((gameObservation, action, reward))

    # print(
    #     f"Round: {gameObservation[5]} - Time Left: {gameObservation[12]} - Obs duration: {obs_time} - Act duration: {act_time} - Overall duration: {time.time() - start_time}")






    # if finalPosition != 0:
    #     print(env.underlord.rewardSummary)
    #     print(
    #         f"GAME OVER! final position: {finalPosition} - final reward: {runningReward} - bought: {env.underlord.localHeroID} heroes!")
    #     runningReward = 0
    #     reward = 0
    #     # need to call a reset of env here
    #     finalObs = env.underlord.getObservation()
    #
    #     if not env.observation_space.contains(finalObs):
    #         print(gameObservation)
    #         print("Not lined up 3")
    #         sys.exit()
    #
    #     if not env.observation_space.contains(finalObs):
    #         print(gameObservation)
    #         print("Not lined up 4")
    #
    #
    #     client.end_episode(episode_id=episode_id, observation=finalObs)
    #     env.underlord.resetEnv()
    #     # fileWriter = logger(episode_id)
    #     # fileWriter.createLog()
    #     # fileWriter.writeLog(replayList)
    #     # replayList.clear()
    #
    #     # if forced:
    #     #     # print("Updating policy weights")
    #     #     client.update_policy_weights()
    #     #     print('Updated policy weights')
    #
    #     episode_id = client.start_episode(episode_id=None)
    #
    #


