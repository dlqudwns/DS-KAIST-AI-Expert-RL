import time

import gym
import envs
import numpy as np

np.set_printoptions(precision=3, suppress=True)

""" Load environment """
# env_name = 'MazeSample5x5-v0'
# env_name = 'MazeSample10x10-v0'
# env_name = 'MazeRandom10x10-v0'
# env_name = 'MazeRandom30x30-plus-v0'
env_name = 'MyMountainCar-v0'
env = gym.make(env_name)
env.S, env.A, env.T, env.R, env.gamma = env.env.S, env.env.A, env.env.T, env.env.R, env.env.gamma

"""
env.S: the number of states (integer)
env.A: the number of actions (integer)
env.T: transition matrix (S x A x S)-sized array
env.R: reward matrix (S x A)-sized array
env.gamma: discount factor (0 ~ 1)
"""


def policy_evaluation(env, pi):
    """
    :param env: MDP(S, A, T, R, gamma)
    :param pi: behavior policy (S x A)-sized array
    :return: V, Q where V is (S)-sized array and Q is (S x A)-sized array
    """
    V = np.zeros(env.S)
    Q = np.zeros((env.S, env.A))

    r = np.sum(env.R * pi, axis=1)
    P = np.tensordot(pi, env.T, axes=([1], [1]))[np.arange(env.S), np.arange(env.S), :]
    V = np.linalg.inv(np.eye(env.S) - env.gamma * P).dot(r)
    Q = env.R + env.gamma * env.T.dot(V)

    return V, Q


def policy_improvement(env, Q):
    pi = np.zeros((env.S, env.A))
    pi[np.arange(env.S), np.argmax(Q, axis=1)] = 1.
    return pi


pi = np.ones((env.S, env.A)) / env.A
for i in range(1000):
    V, Q = policy_evaluation(env, pi)
    new_pi = policy_improvement(env, Q)
    if np.all(pi == new_pi):
        break
    pi = new_pi
    print(i, pi)
print(Q)
print(pi)

for episode in range(10):
    state = env.reset()
    env.render()
    for t in range(1000):
        action = int(np.random.choice(np.arange(env.A), p=pi[state, :]))
        state1, reward, done, info = env.step(action)
        print("state=%s / action=%d / reward=%f / state1=%s / info=%s" % (state, action, reward, state1, info))

        env.render()
        # time.sleep(0.3)
        time.sleep(0.03)

        if done:
            break
        state = state1

    time.sleep(1)
time.sleep(10)
