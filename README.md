# Reinforcement Learning for Traffic Optimization

This project implements reinforcement learning methods to optimize traffic flow at intersections. The goal is to minimize congestion and improve traffic efficiency using two approaches: **Deep Q-Network (DQN)** and **Proximal Policy Optimization (PPO)**.

## **Project Overview**
The project simulates a traffic control system where an agent manages traffic lights at intersections. The agent learns to reduce congestion by interacting with a custom environment and receiving rewards based on traffic density.

## **Environment Description**
- **Agent**: Represents a traffic controller that changes traffic light states.
- **Action Space**: Discrete actions:
  - `0`: Keep the current traffic light state.
  - `1`: Change the traffic light state.
- **State Space**: A vector representing traffic density at intersections (e.g., `[50, 80, 20, 60]`).
- **Reward Structure**: Negative of total traffic density (`-sum(traffic_density)`), encouraging the agent to reduce congestion.

## **Implemented Methods**
1. **Deep Q-Network (DQN)**:
   - Uses experience replay and a target network for stable training.
   - Optimized for long-term rewards using a discount factor (`gamma`).

2. **Proximal Policy Optimization (PPO)**:
   - Implements a clipped objective for stable policy updates.
   - Includes entropy regularization to encourage exploration.

## **Visualization**
The project includes a dynamic bar chart that visualizes traffic density at intersections in real-time. The animation updates as the agent interacts with the environment.

## **How to Run the Project**
1. **Set Up the Environment**:
   - Clone the repository:
     ```bash
     git clone <repository-link>
     cd Reinforcement-Learning-Summative-Assignment
     ```
   - Create a virtual environment and install dependencies:
     ```bash
     python -m venv venv
     source venv/Scripts/activate  # On Windows
     pip install -r requirements.txt
     ```

2. **Train the Models**:
   - Train the DQN model:
     ```bash
     python training/dqn_training.py
     ```
   - Train the PPO model:
     ```bash
     python training/pg_training.py
     ```

3. **Visualize Traffic**:
   - Run the main script to visualize traffic density:
     ```bash
     python main.py
     ```

4. **Monitor Training**:
   - Start TensorBoard to monitor training logs:
     ```bash
     tensorboard --logdir=training/dqn_tensorboard
     tensorboard --logdir=training/ppo_tensorboard
     ```

## **Project Structure**