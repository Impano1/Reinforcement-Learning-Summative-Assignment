import numpy as np
import gymnasium as gym

class TrafficEnv(gym.Env):
    def __init__(self):
        super(TrafficEnv, self).__init__()
        
        # Define the action space (e.g., 0: keep current light, 1: change light)
        self.action_space = gym.spaces.Discrete(2)
        
        # Define the state space (e.g., traffic density at 4 intersections)
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(4,), dtype=np.float32)
        
        # Initialize the state (traffic density at intersections)
        self.state = np.zeros(self.observation_space.shape, dtype=np.float32)

    def reset(self, seed=None, options=None):
        # Reset the environment to an initial state
        self.state = np.random.randint(0, 100, size=self.observation_space.shape).astype(np.float32)
        return self.state, {}

    def step(self, action):
        
        # Simulate traffic density changes based on the action
        if action == 1:  # Change traffic light
            self.state = np.maximum(self.state - np.random.randint(10, 30, size=self.state.shape), 0)
        else:  
            self.state = np.minimum(self.state + np.random.randint(5, 15, size=self.state.shape), 100)
        
        # Calculate reward (negative of total traffic density)
        reward = -np.sum(self.state)
        
        # Check if the episode is done (e.g., after a fixed number of steps)
        done = np.all(self.state < 10)  
        
        return self.state, reward, done, False, {}