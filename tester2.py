from environment import BrawlEnv



env = BrawlEnv({'sleep': True})


while True:

    env.releaseAllKeys(Force=True)