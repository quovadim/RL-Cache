from collections import deque
import numpy as np
import random


class Agent:
    def __init__(self, model, config):
        self.model = model
        self.state_size = config['state size']
        self.action_size = config['action size']
        self.memory = deque(maxlen=config['max len'])
        self.gamma = config['gamma']
        self.epsilon = config['epsilon']
        self.epsilon_min = config['epsilon min']
        self.epsilon_decay = config['epsilon decay']
        self.learning_rate = config['learning rate']

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay