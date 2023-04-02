import random
import numpy as np
import matplotlib.pyplot as plt
import discreteaction_pendulum
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from collections import deque


#Neural Network for approximating Q-Function
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()
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
    def __init__(self, env, learning_rate, epsilon, num_episodes, target_replace_freq, batch_size, replay_buffer, gamma=0.95):
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
        #memory is defined using a deque for efficient replacement of experiences when memory is full
        self.memory = deque(maxlen = self.replay_buffer) #each element of the memory is [s,a,r,s',done]
        self.init_memory_size = 500
        self.batch_size = batch_size
        if self.replay_buffer > self.batch_size:
            self.initialize_memory()
        self.learn_step_counter = 0
        self.target_replace_freq = target_replace_freq
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
    
    #Populating memory with 500 experiences generated using a random policy
    def initialize_memory(self):
        while len(self.memory) < self.init_memory_size:
            s = self.env.reset()
            done = False
            while not done:
                a = random.randrange(self.env.num_actions)
                (s_new, r, done) = self.env.step(a)
                self.store_experience(s,a,r,s_new,done)
                s = s_new
    
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
            if self.target_replace_freq > 1: #to prevent printing at every step for the no target cases
                print("Updating Target Network")
            self.target_net.load_state_dict(self.evaluate_net.state_dict())
        self.learn_step_counter += 1

        experience_sample = random.sample(self.memory, self.batch_size)
        for s,a,r,s_new,done in experience_sample:
            eval_output = self.evaluate_net(s)[a]
            q_values = self.target_net(s_new).detach()
            target_output = r + (1-done)*self.gamma*(q_values.max(0)[0])
            loss = self.loss(eval_output, target_output)

            self.optimizer.zero_grad()
            loss.backward()

            for para in self.evaluate_net.parameters():
                para.grad.data.clamp_(-1,1)
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon -= (self.epsilon_initial - self.epsilon_min)/18000
        else:
            self.epsilon = self.epsilon_min

        return self.evaluate_net, self.target_net
    
    # def load(self, name):
    #     self.qnetwork().load_state_dict(name)

    # def save(self, name):
    #     self.qnetwork()._save_to_state_dict(name)

    def DQN(self):
        for episode in range(self.num_episodes):
            s = self.env.reset()
            G = 0
            iters = 0
            done = False
            while not done:
                if len(self.memory) > self.batch_size:
                    a = self.epsilon_greedy(s, self.epsilon)
                    (s_new,r,done) = self.env.step(a)
                    iters += 1
                    G += r*self.gamma**(iters-1)
                    self.store_experience(s,a,r,s_new,done)
                    evaluate_net, target_net = self.experience_replay()

                elif len(self.memory) <= self.batch_size:
                    a = self.epsilon_greedy(s, self.epsilon)
                    (s_new,r,done) = self.env.step(a)
                    iters += 1
                    G += r*self.gamma**(iters-1)
                    self.store_experience(s,a,r,s_new,done)
                    if self.learn_step_counter % self.target_replace_freq == 0:
                        if self.target_replace_freq > 1:
                            print("Updating Target Network")
                        self.target_net.load_state_dict(self.evaluate_net.state_dict())
                    self.learn_step_counter += 1
                    evaluate_net = self.evaluate_net
                    eval_output = self.evaluate_net(s)[a]
                    q_values = self.target_net(s_new).detach()
                    target_output = r + self.gamma*(q_values.max(0)[0])
                    loss = self.loss(eval_output, target_output)

                    self.optimizer.zero_grad()
                    loss.backward()

                    for para in self.evaluate_net.parameters():
                        para.grad.data.clamp_(-1,1)
                    self.optimizer.step()
                    if self.epsilon > self.epsilon_min:
                        self.epsilon -= (self.epsilon_initial - self.epsilon_min)/18000

                s = s_new
        
            pi = lambda s: torch.argmax(evaluate_net(s)).item() #could also output a stochastic policy like self.epsilon_greedy(s, epsilon=0.1) 
            value_function = lambda s: torch.max(evaluate_net(s)).item()
            self.log['G'].append(G)
            self.log['episodes'].append(episode)
            print("Episode:", episode+1, "Return:", G, "Epsilon:", self.epsilon)
        return evaluate_net, pi, value_function, self.log
    

def main():
    env = discreteaction_pendulum.Pendulum()
    
    num_runs = 10
    num_episodes = 200
    return_multiple_runs_standard = np.zeros((num_runs,num_episodes))
    return_multiple_runs_no_target = np.zeros((num_runs,num_episodes))
    return_multiple_runs_no_replay = np.zeros((num_runs,num_episodes))
    return_multiple_runs_no_replay_no_target = np.zeros((num_runs,num_episodes))
    for i in range(num_runs):
        print("Standard DQN")
        print("-----------------------")
        agent_1 = DQN_agent(env=env, learning_rate=0.00025, epsilon=1.0, num_episodes=num_episodes, target_replace_freq=1000, batch_size=32, replay_buffer=100000)
        evaluate_net_1, pi_1, value_function_1, log_1 = agent_1.DQN()
        return_multiple_runs_standard[i,:] = log_1['G']

        print("DQN Without Target : Target Network updated at every step")
        print("-----------------------")
        agent_2 = DQN_agent(env=env, learning_rate=0.00025, epsilon=1.0, num_episodes=num_episodes, target_replace_freq=1, batch_size=32, replay_buffer=100000)
        evaluate_net_2, pi_2, value_function_2, log_2 = agent_2.DQN()
        return_multiple_runs_no_target[i,:] = log_2['G']

        print("DQN Without Replay")
        print("-----------------------")
        agent_3 = DQN_agent(env=env, learning_rate=0.00025, epsilon=1.0, num_episodes=num_episodes, target_replace_freq=1000, batch_size=32, replay_buffer=32)
        evaluate_net_3, pi_3, value_function_3, log_3 = agent_3.DQN()
        return_multiple_runs_no_replay[i,:] = log_3['G']

        print("DQN Without Target and Replay : Target Network updated at every step")
        print("-----------------------")
        agent_4 = DQN_agent(env=env, learning_rate=0.00025, epsilon=1.0, num_episodes=num_episodes, target_replace_freq=1, batch_size=32, replay_buffer=32)
        evaluate_net_4, pi_4, value_function_4, log_4 = agent_4.DQN()
        return_multiple_runs_no_replay_no_target[i,:] = log_4['G']

    # conf_int_standard = 1.96*np.std(return_multiple_runs_standard, axis=0)/np.sqrt(num_episodes)
    # conf_int_no_target = 1.96*np.std(return_multiple_runs_no_target, axis=0)/np.sqrt(num_episodes)
    # conf_int_no_replay = 1.96*np.std(return_multiple_runs_no_replay, axis=0)/np.sqrt(num_episodes)
    # conf_int_no_replay_no_target = 1.96*np.std(return_multiple_runs_no_replay_no_target, axis=0)/np.sqrt(num_episodes)
    episode_array = log_1['episodes']

    plt.figure()
    plt.plot(episode_array, np.mean(return_multiple_runs_standard, axis=0), color='b', label='Average Return')
    plt.plot(episode_array, np.mean(return_multiple_runs_no_target, axis=0), color='r', label='Average Return (no target)')
    plt.plot(episode_array, np.mean(return_multiple_runs_no_replay, axis=0), color='g', label='Average Return (no replay)')
    plt.plot(episode_array, np.mean(return_multiple_runs_no_replay_no_target, axis=0), color='c', label='Average Return (no replay, no target)')
    plt.fill_between(episode_array, (np.mean(return_multiple_runs_standard, axis=0)-np.std(return_multiple_runs_standard, axis=0)), (np.mean(return_multiple_runs_standard, axis=0)+np.std(return_multiple_runs_standard, axis=0)), color='b', alpha=.2)
    plt.fill_between(episode_array, (np.mean(return_multiple_runs_no_target, axis=0)-np.std(return_multiple_runs_no_target, axis=0)), (np.mean(return_multiple_runs_no_target, axis=0)+np.std(return_multiple_runs_no_target, axis=0)), color='r', alpha=.2)
    plt.fill_between(episode_array, (np.mean(return_multiple_runs_no_replay, axis=0)-np.std(return_multiple_runs_no_replay, axis=0)), (np.mean(return_multiple_runs_no_replay, axis=0)+np.std(return_multiple_runs_no_replay, axis=0)), color='g', alpha=.2)
    plt.fill_between(episode_array, (np.mean(return_multiple_runs_no_replay_no_target, axis=0)-np.std(return_multiple_runs_no_replay_no_target, axis=0)), (np.mean(return_multiple_runs_no_replay_no_target, axis=0)+np.std(return_multiple_runs_no_replay_no_target, axis=0)), color='c', alpha=.2)
    plt.legend()
    plt.ylim(0,20)
    plt.xlabel('Epsiodes')
    plt.ylabel('Return')
    plt.title('Ablation Study')
    plt.savefig('figures/ablation_study.png')

    theta = np.linspace(-np.pi, np.pi, 500)
    thetadot = np.linspace(-15,15,500)
    theta_array, thetadot_array = np.meshgrid(theta, thetadot)
    pi_array = np.zeros_like(theta_array)
    value_array = np.zeros_like(theta_array)
    for i in range(len(theta)):
        for j in range(len(thetadot)):
            s = np.array((theta_array[i,j],thetadot_array[i,j]))
            pi_array[i,j] = pi_1(s)
            value_array[i,j] = value_function_1(s)
    s = env.reset()
    log_1['s'].append(s)
    done = False
    while not done:
        a = pi_1(s)
        (s,r,done) = env.step(a)
        log_1['t'].append(log_1['t'][-1] + 1)
        log_1['s'].append(s)
        log_1['a'].append(a)
        log_1['r'].append(r)

    log_1['s'] = np.array(log_1['s'])
    theta = log_1['s'][:,0]
    thetadot = log_1['s'][:,1]
    tau = [env._a_to_u(a) for a in log_1['a']]
    fig, ax = plt.subplots(3, 1, figsize=(10, 10))
    ax[0].plot(log_1['t'], theta, label=r'$\theta$')
    ax[0].plot(log_1['t'], thetadot, label=r'$\dot{\theta}$')
    ax[0].axhline(y=np.pi, color='r', linestyle='-', label=r'$\theta = \pm\pi$')
    ax[0].axhline(y=-np.pi, color='r', linestyle='-')
    ax[0].axhline(y=0.1*np.pi, color='g', linestyle='--', label=r'$\theta = \pm 0.1*\pi$')
    ax[0].axhline(y=-0.1*np.pi, color='g', linestyle='--')
    ax[0].axhline(y = 0, color = 'r', linestyle='--', label=r'$\theta = 0$')
    ax[0].legend()
    ax[1].plot(log_1['t'][:-1], tau, label='tau')
    ax[1].legend()
    ax[2].plot(log_1['t'][:-1], log_1['r'], label='r')
    ax[2].legend()
    ax[2].set_xlabel('time step')
    plt.tight_layout()
    plt.savefig('figures/trajectory_pendulum.png')

    plt.figure()
    plt.plot(episode_array, np.mean(return_multiple_runs_standard, axis=0), color='b', label='Average Return')
    plt.fill_between(episode_array, (np.mean(return_multiple_runs_standard, axis=0)-np.std(return_multiple_runs_standard, axis=0)), (np.mean(return_multiple_runs_standard, axis=0)+np.std(return_multiple_runs_standard, axis=0)), color='b', alpha=.2, label='95% Confidence Interval')
    plt.ylabel('Return')
    plt.xlabel('Episodes')
    plt.title('Learning Curve for Standard DQN')
    plt.savefig('figures/learning_curve.png')

    plt.figure()
    plt.contourf(theta_array, thetadot_array, pi_array, levels=30)
    plt.colorbar()
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\dot{\theta}$")
    plt.title("Plot of Optimal Policy for Standard DQN")
    plt.savefig('figures/policy.png')

    plt.figure()
    plt.contourf(theta_array, thetadot_array, value_array, levels=30)
    plt.colorbar()
    plt.xlabel(r"$\theta$")
    plt.ylabel(r"$\dot{\theta}$")
    plt.title("Plot of Value Function for Standard DQN")
    plt.savefig('figures/value_function.png')

    env.video(pi_1, filename='figures/optimal_discreteaction_pendulum.gif')


if __name__ == "__main__":
    main()

