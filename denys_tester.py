from environment import BrawlEnv

env = BrawlEnv()

obs = env.getObservation()
print(obs)


# env.restartRound()
