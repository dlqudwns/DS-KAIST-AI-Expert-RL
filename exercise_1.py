import time

import gym
import gym_maze
import numpy as np

np.set_printoptions(precision=3, suppress=True)

""" Load environment """
# env = gym.make('maze-sample-5x5-v0')
# env = gym.make('maze-sample-10x10-v0')
# env = gym.make('maze-random-10x10-v0')
env = gym.make('maze-random-20x20-plus-v0')
env.S, env.A, env.T, env.R, env.gamma = env.unwrapped.S, env.unwrapped.A, env.unwrapped.T, env.unwrapped.R, env.unwrapped.gamma

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
for i in range(100):
    V, Q = policy_evaluation(env, pi)
    new_pi = policy_improvement(env, Q)
    if np.all(pi == new_pi):
        break
    pi = new_pi
    print(i, pi)
print(V)
print(Q)

for episode in range(10):
    state = env.reset()
    env.render()
    for t in range(1000):
        action = int(np.random.choice(np.arange(env.A), p=pi[state, :]))
        state1, reward, done, info = env.step(action)

        env.render()
        time.sleep(0.3)

        if done:
            break
        state = state1

    time.sleep(1)
time.sleep(10)
