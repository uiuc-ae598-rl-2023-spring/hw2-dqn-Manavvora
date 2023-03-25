import random
import numpy as np
import matplotlib.pyplot as plt
import discreteaction_pendulum
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

#Neural Network for approximating Q-Function
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
        # self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate)
        # self.loss = nn.MSELoss()
        self.float()

    def forward(self, x):
        x = torch.from_numpy(x).float()
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        q_values = self.fc3(out)
        return q_values

#Deep Q-Network Agent
class DQN_agent:
    def __init__(self, env, learning_rate, epsilon, num_episodes, gamma=0.95, replay_buffer = 2000):
        self.env = env
        self.learning_rate = learning_rate
        self.evaluate_net = self.qnetwork()
        self.target_net = self.qnetwork()
        self.epsilon_initial = epsilon
        self.epsilon = self.epsilon_initial
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.num_episodes = num_episodes
        self.gamma = gamma
        self.loss = torch.nn.MSELoss()
        self.optimizer = optim.RMSprop(self.evaluate_net.parameters(), lr=learning_rate)
        self.replay_buffer = replay_buffer
        self.learn_step_counter = 0
        self.target_replace_freq = 10
        self.memory = deque(maxlen = self.replay_buffer) #each element of the memory is [s,a,r,s',done]
        self.log = {
        't': [0],
        's': [],
        'a': [],
        'r': [],
        'G': [],
        'episodes': []
    }
        
    def qnetwork(self):
        model = NeuralNet(input_size = self.env.num_states, hidden_size = 64, output_size = self.env.num_actions, learning_rate = self.learning_rate)
        return model
    
    def store_experience(self, s, a, r, s_new, done):
        self.memory.append((s,a,r,s_new,done))

    def epsilon_greedy(self, s):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.env.num_actions)
        q_values = self.evaluate_net.forward(s)
        greedy_action = torch.argmax(q_values).item()
        return greedy_action
    
    def experience_replay(self, batch_size):
        if self.learn_step_counter % self.target_replace_freq == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())

        experience_sample = random.sample(self.memory, batch_size)
        for s,a,r,s_new,done in experience_sample:
            eval_output = self.evaluate_net(s)[a]
            # target_output = r
            # if not done:
            q_values = self.target_net(s_new).detach()
            target_output = r + self.gamma*(q_values.max(0)[0])
            # target_output = r + self.target_net.forward(s)
            # target_output[a] = target
            # output = self.qnetwork().forward(s)
            loss = self.loss(eval_output, target_output)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            # self.epsilon *= self.epsilon_decay
            self.epsilon -= (self.epsilon_initial - self.epsilon_min)/self.replay_buffer
    
    def load(self, name):
        self.qnetwork().load_state_dict(name)

    def save(self, name):
        self.qnetwork()._save_to_state_dict(name)

    def DQN(self):
        batch_size = 32
        for episode in range(self.num_episodes):
            s = self.env.reset()
            # self.log['s'].append(s)
            G = 0
            iters = 0
            done = False
            while not done:
                a = self.epsilon_greedy(s)
                (s_new,r,done) = self.env.step(a)
                # self.log['t'].append(self.log['t'][-1] + 1)
                # self.log['s'].append(s_new)
                # self.log['a'].append(a)
                # self.log['r'].append(r)
                iters += 1
                G += r*self.gamma**(iters-1)
                self.store_experience(s,a,r,s_new,done)
                s = s_new
                if len(self.memory) > batch_size:
                    self.experience_replay(batch_size)
            self.log['G'].append(G)
            self.log['episodes'].append(episode)
            print("Episode:", episode, "Return:", G, "Epsilon:", self.epsilon)
        return self.log

def main():
    env = discreteaction_pendulum.Pendulum()
    agent = DQN_agent(env=env, learning_rate=0.001, epsilon=1.0, num_episodes=100, gamma=0.95, replay_buffer=1000)
    log = agent.DQN()
    plt.plot(log['episodes'], log['G'])
    plt.show()
    # s = env.reset()
    # log['s'].append(s)
    # while not done:
    #     a

if __name__ == "__main__":
    main()

