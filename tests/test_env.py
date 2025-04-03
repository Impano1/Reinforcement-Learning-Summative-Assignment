from environment.custom_env import TrafficEnv

env = TrafficEnv()

# Reset the environment
state, _ = env.reset()
print("Initial State:", state)

# Take random actions
for _ in range(10):
    action = env.action_space.sample()  # Random action
    state, reward, done, _, _ = env.step(action)
    print(f"Action: {action}, State: {state}, Reward: {reward}, Done: {done}")
    if done:
        break