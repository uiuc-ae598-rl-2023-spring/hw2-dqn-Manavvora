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
    def __init__(self, env, learning_rate, epsilon, num_episodes, target_replace_freq, batch_size, gamma=0.95, replay_buffer = 2000):
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
        self.optimizer = optim.RMSprop(self.evaluate_net.parameters(), lr=learning_rate, alpha=0.95, eps=0.01, momentum=0.95)
        self.replay_buffer = replay_buffer
        self.init_memory_size = 500
        self.learn_step_counter = 0
        self.target_replace_freq = target_replace_freq
        self.batch_size = batch_size
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

    def epsilon_greedy(self, s, epsilon):
        if np.random.rand() <= epsilon:
            return random.randrange(self.env.num_actions)
        q_values = self.evaluate_net.forward(s)
        greedy_action = torch.argmax(q_values).item()
        return greedy_action
    
    def experience_replay(self):
        if self.learn_step_counter % self.target_replace_freq == 0:
            print("Updating Target Network")
            self.target_net.load_state_dict(self.evaluate_net.state_dict())
        self.learn_step_counter += 1

        experience_sample = random.sample(self.memory, self.batch_size)
        # print(experience_sample[0])
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

            for para in self.evaluate_net.parameters():
                para.grad.data.clamp_(-1,1)
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            # self.epsilon *= self.epsilon_decay
            self.epsilon -= (self.epsilon_initial - self.epsilon_min)/10000
        
        return self.evaluate_net, self.target_net
    
    # def load(self, name):
    #     self.qnetwork().load_state_dict(name)

    # def save(self, name):
    #     self.qnetwork()._save_to_state_dict(name)

    def DQN(self):
        # batch_size = 32
        # pi = lambda s: self.env.num_actions // 2
        for episode in range(self.num_episodes):
            s = self.env.reset()
            # self.log['s'].append(s)
            G = 0
            iters = 0
            done = False
            while not done:
                if len(self.memory) < self.init_memory_size:
                    a = random.randrange(self.env.num_actions)
                    (s_new, r, done) = self.env.step(a)
                    self.store_experience(s,a,r,s_new,done)
                    s = s_new
                    continue
                else:
                    a = self.epsilon_greedy(s, self.epsilon)
                    (s_new,r,done) = self.env.step(a)
                    iters += 1
                    G += r*self.gamma**(iters-1)
                    self.store_experience(s,a,r,s_new,done)
                    s = s_new
                    if len(self.memory) > self.batch_size:
                        evaluate_net, target_net = self.experience_replay()
            pi = lambda s: torch.argmax(evaluate_net(s)).item()
            self.log['G'].append(G)
            self.log['episodes'].append(episode)
            print("Episode:", episode+1, "Return:", G, "Epsilon:", self.epsilon)
        return evaluate_net, pi, self.log


# def wrap_pi(x):
#     theta = ((x + np.pi) % (2 * np.pi)) - np.pi
#     return theta


def main():
    env = discreteaction_pendulum.Pendulum()
    agent = DQN_agent(env=env, learning_rate=0.00025, epsilon=1.0, num_episodes=100, target_replace_freq=1000, batch_size=32, gamma=0.95, replay_buffer=100000)
    evaluate_net, pi, log = agent.DQN()
    # s = env.reset()
    # print(pi(s))
    s = env.reset()
    log['s'].append(s)
    done = False
    opt_policy = []
    while not done:
        a = pi(s)
        opt_policy.append(a)
        (s,r,done) = env.step(a)
        log['t'].append(log['t'][-1] + 1)
        log['s'].append(s)
        log['a'].append(a)
        log['r'].append(r)
    print(opt_policy)
    log['s'] = np.array(log['s'])
    theta = log['s'][:,0]
    print(theta)
    thetadot = log['s'][:,1]
    tau = [env._a_to_u(a) for a in log['a']]
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    ax[0].plot(log['t'], theta, label='theta')
    ax[0].plot(log['t'], thetadot, label='thetadot')
    ax[0].axhline(y=np.pi, color='r', linestyle='-', label='Theta = pi')
    ax[0].axhline(y=-np.pi, color='r', linestyle='-', label='Theta = -pi')
    ax[0].axhline(y = 0, color = 'r', linestyle='--', label='Theta = 0')
    ax[0].legend()
    ax[1].plot(log['t'][:-1], tau, label='tau')
    ax[1].legend()
    ax[2].plot(log['t'][:-1], log['r'], label='r')
    ax[2].legend()
    ax[2].set_xlabel('time step')
    plt.tight_layout()
    plt.savefig('figures/test_discreteaction_pendulum.png')

    plt.figure()
    plt.plot(log['episodes'], log['G'])
    plt.ylabel('Return')
    plt.xlabel('Episodes')
    plt.title('Learning Curve')
    plt.savefig('figures/learning_curve.png')

    # env.video(pi, filename='figures/test_discreteaction_pendulum.gif')
if __name__ == "__main__":
    main()

