import random
import time
from collections import deque

import gym
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential

np.set_printoptions(precision=3, suppress=True, threshold=10000, linewidth=250)

class DQNAgent:
    def __init__(self, state_dim, action_size, gamma=0.99):
        self.state_dim = state_dim
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Deep-Q Network Model
        model = Sequential([
            Dense(24, input_dim=self.state_dim, activation='relu'),
            Dense(24, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.train.AdamOptimizer(learning_rate=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


""" Load environment """
# env_name = 'MountainCar-v0'
env_name = 'CartPole-v0'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_size = env.action_space.n
gamma = 0.99

agent = DQNAgent(state_dim, action_size)

for episode in range(100):
    state = env.reset()
    env.render()

    episode_reward = 0.
    for t in range(1000):
        action = agent.act(state)
        state1, reward, done, info = env.step(action)

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
