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

# Setting update_interval to false, so it doesn't update in middle of games, will be manually updating it between games
client = PolicyClient(address=f"http://{args.ip}:55556", update_interval=False, inference_mode=local)
# client = PolicyClient(address=f"http://{args.ip}:55556", update_interval=60, inference_mode=local)


forced = True
root = None

env = BrawlEnv({'sleep': True})

print('trying to get initial eid')
episode_id = client.start_episode()

# if local == 'remote':
#     env.underlord.startNewGame()c

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

counter = 0
runningCounter = 0

startTime = time.time()
endTime = time.time()

env.restartRound()

actionTimeOut = 0.2
actionTime = time.time()

epochActions = 4096
actionsUntilEpoch = 4096
epochNum = 0

while True:

    elapsed_time = time.time() - actionTime
    if elapsed_time < actionTimeOut:
        continue



    actionTime = time.time()

    # average out to ~30actions a second
    counter = counter + 1
    runningCounter = runningCounter + 1
    endTime = time.time()
    if (endTime - startTime) > 1:
        print(f"actions per second: {counter}")
        startTime = time.time()
        counter = 0

    gameObservation, reward, gameOver = env.getObservation()
    print('got observation')
    # print(gameObservation)
    # print(env.observation_space.contains(gameObservation))
    # print(reward, gameOver)

    if not env.observation_space.contains(gameObservation):
        print(gameObservation)
        print("Not lined up 1")
        print(env.underlord.heroAlliances)
        sys.exit()

    action = None

    action = client.get_action(episode_id=episode_id, observation=gameObservation)
    print('got action')
    env.act(action)
    print('took action')

    runningReward += reward
    # act_time = time.time() - act_time
    # print("--- %s seconds to get do action ---" % (time.time() - start_time))
    # print(f"running reward: {reward}")

    client.log_returns(episode_id=episode_id, reward=reward)
    print('logged returns')
    # Updating the model after every game in case there is a new one

    if gameOver or elapsed_time > 20:

        if elapsed_time > 20:
            print("restarting due to elapsed time")

        if reward <= -1:
            print(f"GAME OVER! WE Lost final reward: {runningReward}! Number of actions: {runningCounter}")

        else:
            print(f"GAME OVER! WE Won final reward: {runningReward}! Number of actions: {runningCounter}")

        actionsUntilEpoch = actionsUntilEpoch - runningCounter

        if actionsUntilEpoch < 0:
            epochNum = epochNum + 1

        print(f"Actions until epoch: {actionsUntilEpoch}, current epoch: {epochNum}")

        if actionsUntilEpoch < 0:
            actionsUntilEpoch = epochActions

        runningReward = 0
        runningCounter = 0
        reward = 0
        # need to call a reset of env here
        finalObs, reward, gameOver = env.getObservation()

        if not env.observation_space.contains(finalObs):
            print(gameObservation)
            print("Not lined up 3")
            sys.exit()

        client.end_episode(episode_id=episode_id, observation=finalObs)
        print('ended episode')
        episode_id = client.start_episode(episode_id=None)
        print('started new episode')

        if local == 'local':
            print("updating weights")
            client.update_policy_weights()
        print("restarting round")
        env.restartRound()
        print('round restarted')

    # print('finished logging step')

    # print("--- %s seconds to get finish logging return ---" % (time.time() - start_time))

    # replayList.append((gameObservation, action, reward))

    # print(
    #     f"Round: {gameObservation[5]} - Time Left: {gameObservation[12]} - Obs duration: {obs_time} - Act duration: {act_time} - Overall duration: {time.time() - start_time}")
