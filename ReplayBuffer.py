import random
import torch

class replayBuffer:
    def __init__(self, sample_size, max_mem_size):
        self.sample_size = sample_size
        self.max_mem_size = max_mem_size
        self.mem_counter = 0

        self.memory = []
    
    def insert(self, transition):

        if len(self.memory) < self.max_mem_size:
            self.memory.append(None)
        index = self.mem_counter % self.max_mem_size
        self.memory[index] = transition
        self.mem_counter += 1

    def sample(self):
        batch = random.sample(self.memory, self.sample_size)
        batch = zip(*batch)
        return [torch.cat(item) for item in batch]

    def can_sample(self):
        if len(self.memory) <= self.sample_size * 8:
            return False
        else:
            return True

    def clear(self):
        self.memory = []

if __name__ == '__main__':
    import gym
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

    print(memory.sample())