
from training.dqn_training import model as dqn_model
from training.pg_training import model as pg_model
from environment.rendering import visualize_traffic

if __name__ == "__main__":
    print("Running DQN Model...")
    dqn_model.learn(total_timesteps=1000)
    print("Running PPO Model...")
    pg_model.learn(total_timesteps=1000)

    traffic_density = [50, 80, 20, 60] 
    print("Visualizing Traffic...")
    visualize_traffic(traffic_density)