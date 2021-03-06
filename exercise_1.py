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

env = gym.make(env_name)

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

    ###################
    # TODO: V와 Q를 계산하는 코드를 여기에 작성하세요.
    # ...
    ###################

    return V, Q


pi = np.ones((env.S, env.A)) / env.A

V, Q = policy_evaluation(env, pi)

env.draw_policy_evaluation(Q, pi)
env.render()
time.sleep(10)
