import torch
import torch.nn as nn
import torch.nn.functional as F

policy = nn.Sequential(
    nn.Linear(4, 128),
    nn.ReLU(),
    nn.Linear(128, 128),
    nn.ReLU(),
    nn.Linear(128, 2),
    nn.Softmax(dim = -1)
)

if __name__ == '__main__':
    import gym
    from ReplayBuffer import replayBuffer
    from PreprocessEnv import PreprocessEnv
    env = PreprocessEnv(gym.make('CartPole-v1'))
    memory = replayBuffer(10, 1000)
    for i in range(10):
        state = env.reset()
        done = False
        while not done:
            action = torch.tensor(env.action_space.sample()).view(1, -1)
            next_state, reward, done = env.step(action)
            memory.insert([state, action, reward, next_state])
            state = next_state

    state_b, action_b, reward_b, next_state_b = memory.sample()
    print(state_b)
    print(policy(state_b))
