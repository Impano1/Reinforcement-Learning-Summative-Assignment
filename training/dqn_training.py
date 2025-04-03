import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from environment.custom_env import TrafficEnv
from stable_baselines3 import DQN

# Initialize the custom environment
env = TrafficEnv()

# Initialize the DQN model
model = DQN(
    "MlpPolicy",  
    env,          
    learning_rate=0.0005,  
    gamma=0.99,            
    verbose=1              
)

# Train the model
model.learn(total_timesteps=10000)

# Save the trained model
model.save("models/dqn/traffic_dqn_model")

# Close the environment
env.close()