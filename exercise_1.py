import time

import gym
import envs
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, suppress=True, threshold=10000, linewidth=250)

""" Load environment """
# env_name = 'MazeSample3x3-v0'
# env_name = 'MazeSample5x5-v0'
# env_name = 'MazeSample10x10-v0'
# env_name = 'MazeRandom10x10-v0'
env_name = 'MazeRandom10x10-plus-v0'
env = gym.make(env_name)
env.S, env.A, env.T, env.R, env.gamma = env.env.S, env.env.A, env.env.T, env.env.R, env.env.gamma
env.draw_policy_evaluation = env.env.draw_policy_evaluation

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

pi = np.ones((env.S, env.A)) / env.A
# pi = np.random.dirichlet(np.ones(env.A), env.S)
# pi = np.zeros((env.S, env.A))
# pi[:, 0] = 1
# pi = np.array([[0, 1, 0, 0],
#                [0, 0, 0, 1],
#                [0, 0, 0, 1],
#                [0, 1, 0, 0],
#                [0, 0, 1, 0],
#                [1, 0, 0, 0],
#                [0, 0, 1, 0],
#                [0, 0, 1, 0],
#                [1, 0, 0, 0],
#                [0.25, 0.25, 0.25, 0.25]])

V, Q = policy_evaluation(env, pi)
print(Q)
# print(pi)

# env.reset()
# print(env.step(2))
env.draw_policy_evaluation(Q, pi)
for i in range(600):
    env.render()
    time.sleep(0.1)
