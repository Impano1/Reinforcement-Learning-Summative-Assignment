import numpy as np
import gymnasium as gym

class TrafficEnv(gym.Env):
    def __init__(self):
        super(TrafficEnv, self).__init__()
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(4)  
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        # Seed the environment
        super().reset(seed=seed)
        np.random.seed(seed)

        # Reset the environment to an initial state
        self.state = np.zeros(self.observation_space.shape, dtype=np.float32)

        # Return the initial observation
        return self.state, {}

    def step(self, action):
        # Example step logic
        reward = -1  
        done = False  
        self.state = np.random.rand(*self.observation_space.shape) * 100  # Example state update
        return self.state, reward, done, False, {}