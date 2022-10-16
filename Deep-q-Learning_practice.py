import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.optim import AdamW
from torch import nn as nn
from tqdm import tqdm
import random
import torch
import copy
import gym


env = gym.make('CartPole-v1')


state_dims = env.observation_space.shape[0]
num_actions = env.action_space.n

class PreprocessEnv(gym.Wrapper):
    
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
    
    def reset(self):
        obs = self.env.reset()
        return torch.from_numpy(obs).unsqueeze(dim=0).float()
    
    def step(self, action):
        action = action.item()
        next_state, reward, done, info = self.env.step(action)
        next_state = torch.from_numpy(next_state).unsqueeze(dim=0).float()
        reward = torch.tensor(reward).view(1, -1).float()
        done = torch.tensor(done).view(1, -1)
        return next_state, reward, done, info

env = PreprocessEnv(env)

def policy(state, epsilon=0.):
    if torch.rand(1) < epsilon:
        return torch.randint(num_actions, (1, 1))
    else:
        av = q_network(state).detach() ###
        return torch.argmax(av, dim=-1, keepdim=True)


class ReplayMemory:
    
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def insert(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        assert self.can_sample(batch_size)

        batch = random.sample(self.memory, batch_size)
        batch = zip(*batch)
        return [torch.cat(items) for items in batch]

    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size * 10

    def __len__(self):
        return len(self.memory)

q_network = nn.Sequential(
    nn.Linear(state_dims, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, num_actions)
)

target_q_network = copy.deepcopy(q_network).eval()

def deep_q_learning(q_network, policy, episodes, lr = 0.0001, batch_size = 32, gamma = 0.99, epsilon = 1, eps_dec = 0.05, eps_end = 0.2):
    
    optim = AdamW(q_network.parameters(), lr = lr)
    memory = ReplayMemory()
    stats = {'MSE Loss' : [], 'Returns' : [], 'Epsilon' : []}
    
    for episode in tqdm(range(1, episodes + 1)):
        state = env.reset()
        done = False
        ep_return = 0
        
        while not done:
            action = policy(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            
            memory.insert([state, action, reward, done, next_state])

            if memory.can_sample(batch_size):
                state_b, action_b, reward_b, done_b, next_state_b = memory.sample(batch_size)
                qsa_b = q_network(state_b).gather(1, action_b)

                next_qsa_b = target_q_network(next_state_b)
                next_qsa_b = torch.max(next_qsa_b, dim = -1, keepdim = True)[0]

                target_b = reward_b + ~done_b * gamma * next_qsa_b
                loss = F.mse_loss(qsa_b, target_b)

                q_network.zero_grad()
                loss.backward()
                optim.step()

                stats['MSE Loss'].append(loss.item())
            state = next_state
            ep_return += reward.item()
        stats['Returns'].append(ep_return)
        stats['Epsilon'].append(epsilon)
        epsilon = epsilon - eps_dec if epsilon != eps_end else epsilon
        
        
        if episode % 10 == 0:
            target_q_network.load_state_dict(q_network.state_dict())
    return stats


stats = deep_q_learning(q_network, policy, 700)

plt.plot(stats['Returns'])
plt.show()
