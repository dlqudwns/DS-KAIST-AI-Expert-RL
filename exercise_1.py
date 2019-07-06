import os
import matplotlib.pyplot as plt
import sys
import numpy as np
import math
import random
import gym
import gym_maze
import time

env = gym.make('maze-v0')

state = env.reset()
for t in range(1000):
    action = env.action_space.sample()
    state1, reward, done, _ = env.step(action)
    print('[%d] state=%s / action=%s / reward=%f / state1=%s' % (t, state, action, reward, state1))
    if done:
        print('done')
        break

    env.render()
    state = state1

    time.sleep(1)
time.sleep(10000)