import torch
import gym

class PreprocessEnv(gym.Wrapper):
   
    def __init__(self, env):
        super().__init__(env)    
    
    def step(self, action):
        action = action.item()
        next_state, reward, done, _ = self.env.step(action)
        next_state = torch.from_numpy(next_state).view(1, -1).float()
        reward = torch.tensor(reward).view(1, -1)
        done = torch.tensor(done).view(1, -1)
        return next_state, reward, done 

    def reset(self):
        state = self.env.reset()
        state = torch.from_numpy(state).view(1, -1).float()
        return state


if __name__ == '__main__':
    env = PreprocessEnv(gym.make('CartPole-v1'))
    state = env.reset()
    print(state)
    done = False
    action = torch.tensor(env.action_space.sample()).view(1, -1)
    print(env.step(action))