import random
import numpy as np
import matplotlib.pyplot as plt
import discreteaction_pendulum
import torch.nn as nn
import torch.optim as optim

#Neural Network for approximating Q-Function
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        q_value = self.fc3(out)
        return q_value

#Deep Q-Network Agent
class DQN_agent:
    def __init__(self, env, learning_rate, epsilon, num_episodes, gamma=0.95):
        self.env = env
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.log = {
        't': [0],
        's': [],
        'a': [],
        'r': []
    }
        
    def qnetwork(self):
        model = NeuralNet(input_size = self.env.num_states, hidden_size = 64, output_size = self.env.num_actions, learning_rate = self.learning_rate)
        return model
    

