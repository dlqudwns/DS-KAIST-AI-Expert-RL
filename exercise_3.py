import time

import gym
import envs
import numpy as np

np.set_printoptions(precision=3, suppress=True, threshold=10000, linewidth=250)

""" Load environment """
# env_name = 'MazeSample5x5-v0'
# env_name = 'MazeSample10x10-v0'
# env_name = 'MazeRandom10x10-v0'
# env_name = 'MazeRandom10x10-plus-v0'
# env_name = 'MazeRandom20x20-v0'
# env_name = 'MazeRandom20x20-plus-v0'
# env_name = 'MyMountainCar-v0'
env_name = 'MyCartPole-v0'
env = gym.make(env_name)
S, A, gamma = env.env.S, env.env.A, env.env.gamma
env.draw_policy_evaluation = env.env.draw_policy_evaluation

"""
S: the number of states (integer)
A: the number of actions (integer)
gamma: discount factor (0 ~ 1)
"""

def epsilon_greedy(Q, s, epsilon=0.1):
    if np.random.rand() < epsilon:
        return np.random.randint(0, A)
    else:
        return np.argmax(Q[s, :])

Q = np.zeros((S, A))
alpha = 0.2

for episode in range(100):
    state = env.reset()
    env.render()

    episode_reward = 0.
    for t in range(1000):
        action = epsilon_greedy(Q, state, epsilon=0.2)
        state1, reward, done, info = env.step(action)

        # Update Q-table
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[state1, :]) - Q[state, action])
        env.draw_policy_evaluation(Q)

        episode_reward += reward
        print("[%4d] state=%4s / action=%d / reward=%7.4f / state1=%4s / info=%s" % (t, state, action, reward, state1, info))

        env.render()
        time.sleep(0.01)

        if done:
            break
        state = state1
    print('Episode reward: %.4f' % episode_reward)

    time.sleep(1)
time.sleep(10)
