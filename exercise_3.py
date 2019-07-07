import random
import time
from collections import deque

import gym
import numpy as np
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam

np.set_printoptions(precision=3, suppress=True, threshold=10000, linewidth=250)


class DQNAgent:

    def __init__(self, state_dim, action_size, gamma=0.99):
        self.state_dim = state_dim
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.q_model = self._build_model()
        self.q_model.compile(loss='mse', optimizer=Adam(0.001))
        self.target_q_model = self._build_model()
        self.update_target_q_weights()  # copy Q network params to target Q network

        self.replay_counter = 0

    def _build_model(self):
        # Deep-Q Network Model
        model = Sequential([
            Dense(64, input_dim=self.state_dim, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        return model

    def update_target_q_weights(self):
        self.target_q_model.set_weights(self.q_model.get_weights())

    def act(self, state):
        # epsilon-greedy policy
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.action_size)
        q_values = self.q_model.predict(np.array([state]))
        return np.argmax(q_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((np.array([state]), action, reward, np.array([next_state]), done))

    def get_target_q_value(self, next_state):
        return reward + self.gamma * np.max(self.target_q_model.predict(next_state)[0])

    def replay(self, batch_size):
        sars_batch = random.sample(self.memory, batch_size)
        state_batch, q_values_batch = [], []

        # fixme: for speedup, this could be done on the tensor level
        # but easier to understand using a loop
        for state, action, reward, next_state, done in sars_batch:
            # policy prediction for a given state
            q_values = self.q_model.predict(state)

            # get Q_max
            q_value = self.get_target_q_value(next_state)

            # correction on the Q value for the action used
            q_values[0][action] = reward if done else q_value

            # collect batch state-q_value mapping
            state_batch.append(state[0])
            q_values_batch.append(q_values[0])

        # train the Q-network
        self.q_model.fit(np.array(state_batch), np.array(q_values_batch), batch_size=batch_size, epochs=1, verbose=1)

        # update exploration-exploitation probability
        self.epsilon = np.max([self.epsilon * self.epsilon_decay, self.epsilon_min])

        # copy new params on old target after every 10 training updates
        if self.replay_counter % 10 == 0:
            self.update_target_q_weights()

        self.replay_counter += 1

""" Load environment """
# env_name = 'MountainCar-v0'
env_name = 'CartPole-v0'
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_size = env.action_space.n
gamma = 0.99
batch_size = 64

agent = DQNAgent(state_dim, action_size)

for episode in range(5000):
    state = env.reset()
    env.render()

    episode_reward = 0.
    for t in range(1000):
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        agent.remember(state, action, reward, next_state, done)

        episode_reward += reward
        print("[%4d] state=%4s / action=%d / reward=%7.4f / next_state=%4s / info=%s" % (t, state, action, reward, next_state, info))
        #
        # env.render()
        # time.sleep(0.01)

        if done:
            break
        state = next_state

    # call experience relay
    if len(agent.memory) >= batch_size:
        agent.replay(batch_size)

    print('Episode reward: %.4f' % episode_reward)
time.sleep(10)
