import gymnasium as gym
from gymnasium import spaces
import numpy as np

class MyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(5) 
        self.state = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        observation = self.state
        info = {} 
        return observation, info
        
    def step(self, action):
        if action == 1:
            self.state = min(self.state + 1, 4)
        elif action == 0: 
            self.state = max(self.state - 1, 0)
        
        terminated = (self.state == 4)
        reward = 10 if terminated else -1
        truncated = False
        
        observation = self.state 
        info = {}
        return observation, reward, terminated, truncated, info
    
if __name__ == "__main__":   
    env = MyEnv()
    obs, info = env.reset()
    print("reset -> obs: ", obs, "info: ", info)
    done = False
    step_i = 0
    while not done:
        action = env.action_space.sample() 
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_i += 1
        print(f"STEP {step_i}: action={action} -> obs={obs}, reward={reward}, term={terminated}, trunc={truncated}")

    print("Episode finished.")
