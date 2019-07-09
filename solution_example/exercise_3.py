import time

import gym
import envs
import numpy as np

np.set_printoptions(precision=3, suppress=True, threshold=10000, linewidth=250)

""" Load environment """
env_name = 'MazeSample3x3-v0'
# env_name = 'MazeSample5x5-v0'
# env_name = 'MazeSample10x10-v0'
# env_name = 'MazeRandom10x10-v0'
# env_name = 'MazeRandom10x10-plus-v0'
# env_name = 'MazeRandom20x20-v0'
# env_name = 'MazeRandom20x20-plus-v0'
# env_name = 'MyCartPole-v0'
# env_name = 'MyMountainCar-v0'

env = gym.make(env_name)
env.T = env.R = None

"""
env.S: the number of states (integer)
env.A: the number of actions (integer)
gamma: discount factor (0 ~ 1)
"""


def epsilon_greedy(Q, s, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.randint(0, env.A)
    else:
        return np.argmax(Q[s, :])


step_size = 0.2

Q = np.zeros((env.S, env.A))
epsilon = 1.0
epsilon_min = 0.1

for episode in range(1000):
    state = env.reset()
    env.render()

    episode_reward = 0.
    for t in range(10000):
        action = epsilon_greedy(Q, state, epsilon)
        next_state, reward, done, info = env.step(action)

        # Update Q-table
        target_Q = reward if done else reward + env.gamma * np.max(Q[next_state, :])
        Q[state, action] = Q[state, action] + step_size * (target_Q - Q[state, action])

        episode_reward += reward
        print("[epi=%4d,t=%4d] state=%4s / action=%d / reward=%7.4f / next_state=%4s / info=%s / Q[s]=%s" % (episode, t, state, action, reward, next_state, info, Q[state, :]))

        env.draw_policy_evaluation(Q)
        env.render()
        time.sleep(0.01)

        if done:
            break
        state = next_state

    epsilon = np.max([epsilon * 0.99, epsilon_min])
    print('[%4d] Episode reward=%.4f / epsilon=%f' % (episode, episode_reward, epsilon))

    time.sleep(0.1)
time.sleep(10)
