from environment.custom_env import TrafficEnv
from environment.rendering import visualize_traffic

if __name__ == "__main__":
    # Initialize the custom environment
    env = TrafficEnv()

    # Reset the environment
    state, _ = env.reset()

    
    print("Visualizing Traffic...")
    visualize_traffic(state)