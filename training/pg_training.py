import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from environment.custom_env import TrafficEnv

# Initialize the custom environment
env = TrafficEnv()

# Initialize the PPO model with custom hyperparameters
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,  
    gamma=0.98,            
    n_steps=2048,          
    verbose=1
)

# Set up TensorBoard logging
log_dir = "./training/ppo_tensorboard/"  
os.makedirs(log_dir, exist_ok=True)  
new_logger = configure(log_dir, ["stdout", "tensorboard"])
model.set_logger(new_logger)

# Train the model
model.learn(total_timesteps=10000)

# Save the trained model
model.save("models/pg/traffic_ppo_model")