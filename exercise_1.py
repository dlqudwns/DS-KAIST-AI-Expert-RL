import time

import gym
import gym_maze
import numpy as np

env = gym.make('maze-v0')

state = np.array(env.reset())
for t in range(1000):
    action = env.action_space.sample()
    state1, reward, done, _ = env.step(action)
    print('[%d] state=%s / action=%s / reward=%f / state1=%s' % (t, state, action, reward, state1))
    if done:
        print('done')
        break

    env.render()
    state = np.array(state1)

    time.sleep(1)
time.sleep(10000)
