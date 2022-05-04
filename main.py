import gym
import torch
from torch.optim import AdamW
from PreprocessEnv import PreprocessEnv
from ReplayBuffer import replayBuffer
from ActorCritic import policy

sample_size = 32
max_mem_size = 500
num_episodes = 100
lr = 0.001

env = PreprocessEnv(gym.make('CartPole-v1'))
memory = replayBuffer(sample_size, max_mem_size)

optim = AdamW(policy.parameters(), lr)

for episode in range(1, num_episodes + 1):
    #play out full episode
    state = env.reset()
    done = torch.tensor(False).view(1, -1)
    transition = []
    ep_return = 0

    while not done.item():
        action = policy(state).multinomial(1).detach()
        next_state, reward, done = env.step(action)
        
