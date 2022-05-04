import gym
import torch
from tqdm import tqdm
from torch.optim import AdamW
from PreprocessEnv import PreprocessEnv
from ReplayBuffer import replayBuffer
from ActorCritic import policy
import matplotlib.pyplot as plt

sample_size = 32
max_mem_size = 500
num_episodes = 1000
lr = 0.001
gamma = 0.99

env = PreprocessEnv(gym.make('CartPole-v1'))
memory = replayBuffer(sample_size, max_mem_size)

optim = AdamW(policy.parameters(), lr)

returns = []

for episode in tqdm(range(1, num_episodes + 1)):
    #play out full episode
    state = env.reset()
    done = torch.tensor(False).view(1, -1)
    transitions = []
    ep_return = 0

    while not done.item():
        action = policy(state).multinomial(1).detach()
        next_state, reward, done = env.step(action)
        transitions.append([state, action, ~done * reward])
        ep_return += reward.item()
        state = next_state
    #go backwards and update
    G = 0
    for t, (state_t, action_t, reward_t) in reversed(list(enumerate(transitions))):
        G = gamma * G + reward_t ###
        probabilities = policy(state_t)
        log_probabilities = torch.log(probabilities + 1e-6)
        action_t_log_prob = log_probabilities[0, action_t]
        entropy_regularization = torch.sum(probabilities * log_probabilities, dim = -1, keepdim=True)
        gamma_t = gamma ** t
        policy_performance = -gamma_t * G * action_t_log_prob - entropy_regularization
        optim.zero_grad()
        policy_performance.backward()
        optim.step()
    returns.append(ep_return)

plt.plot(returns)
plt.show()