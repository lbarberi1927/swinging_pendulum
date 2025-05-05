# Swinging Pendulum Control with Soft Actor-Critic (SAC)
**Author**: Leonardo Barberi

## Description
This repository implements a reinforcement learning solution for the classic control problem of swinging up and balancing a pendulum, using the Soft Actor-Critic (SAC) algorithm. This project was completed as part of ETH Zurich's Probabilistic Artificial Intelligence course and demonstrates the application of maximum-entropy RL in continuous action spaces.

## Learning Problem
In the OpenAI Gym `Pendulum-v1` environment, the agent observes the pendulum's angle and angular velocity and must output a continuous torque to swing the pendulum upright and keep it balanced. The objective is to maximize the cumulative reward, which penalizes deviation from the upright position and excessive torque usage.

## SAC
Soft Actor-Critic (SAC) is an off-policy actor-critic algorithm that optimizes a stochastic policy by maximizing a trade‑off between expected return and policy entropy. Key features include:
- **Entropy Regularization**: Encourages exploration by adding an entropy term to the objective.
- **Off‑Policy Updates**: Reuses experiences from a replay buffer for sample efficiency.
- **Stability**: Employs soft updates of the target networks and separate Q‑function estimates.

### Visualizations
The video below shows the pendulum being swung up and stabilized by the trained SAC agent:


## Installation
To set up the environment and install dependencies, run:

```bash
# Clone the repository
git clone https://github.com/lbarberi1927/swinging-pendulum.git
cd swinging-pendulum

# (Optional) Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

## Usage
The `SAC.py` script contains multiple classes to implement the SAC algorithm and a toy `main` function to run a training session and visualize results.
