import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from environment.custom_env import TrafficEnv
from stable_baselines3 import PPO

# Initialize the custom environment
env = TrafficEnv()

# Initialize the PPO model
model = PPO(
    "MlpPolicy",  
    env,          
    learning_rate=0.0003,  
    gamma=0.98,            
    n_steps=2048,          
    verbose=1              
)

# Train the model
model.learn(total_timesteps=10000)

# Save the trained model
model.save("models/pg/traffic_ppo_model")

# Close the environment
env.close()