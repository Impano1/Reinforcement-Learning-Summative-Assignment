import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure
from environment.custom_env import TrafficEnv

# Initialize the custom environment
env = TrafficEnv()

# Set up TensorBoard logging
log_dir = "./training/dqn_tensorboard/"  
os.makedirs(log_dir, exist_ok=True)  
new_logger = configure(log_dir, ["stdout", "tensorboard"])

# Initialize the DQN model with custom hyperparameters
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=0.0005,  
    gamma=0.99,            
    batch_size=64,         
    verbose=1
)

# Set the custom logger
model.set_logger(new_logger)

# Train the model
model.learn(total_timesteps=10000)

# Save the trained model
model.save("models/dqn/traffic_dqn_model")