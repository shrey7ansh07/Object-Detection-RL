# Object Detection using Deep Reinforcement Learning

This project implements an object detection system using a Deep Q-Network (DQN), a reinforcement learning algorithm. The agent learns to find objects in an image by adjusting a bounding box to maximize the Intersection over Union (IoU) with a ground truth object.

## How it Works

The project is structured around a custom environment and a DQN agent.

### Environment (`env.py`)

The `DQNObjectDetectionEnv` class defines the environment for the object detection task.

-   **State:** The state is represented by the normalized coordinates of the current bounding box `[x_min, y_min, x_max, y_max]` and the current IoU.
-   **Actions:** The agent can take one of 9 discrete actions to modify the bounding box:
    -   Move (left, right, up, down)
    -   Change width (grow, shrink)
    -   Change height (grow, shrink)
    -   Stop
-   **Reward:** The reward is calculated based on the change in IoU. A positive reward is given for an increase in IoU, a negative reward for a decrease, and a small penalty for no change. There are also bonuses for achieving a high IoU and for using the `STOP` action when the IoU is high.

### DQN Agent (`dqn.py`)

The `DQNAgent` class implements the Deep Q-Network algorithm.

-   **Network:** A multi-layer perceptron (MLP) is used to approximate the Q-values for each action.
-   **Experience Replay:** A replay buffer stores past transitions (state, action, reward, next_state, done) to break the correlation between consecutive samples and stabilize training.
-   **Target Network:** A separate target network is used to generate the target Q-values, which helps to stabilize training. The target network's weights are periodically updated with the policy network's weights.
-   **Epsilon-Greedy Policy:** The agent uses an epsilon-greedy policy to balance exploration and exploitation. The value of epsilon decays over time.

## File Descriptions

-   `dqn.py`: Contains the implementation of the DQN agent, including the Q-network, replay buffer, and training/evaluation loops.
-   `env.py`: Defines the custom object detection environment.
-   `dimension.py`: A utility script to get the dimensions of an image.
-   `requirements.txt`: A list of the Python dependencies for this project.

## Requirements

The project requires the following Python libraries:

-   numpy
-   torch
-   matplotlib
-   Pillow

You can install them using pip:

```bash
pip install -r requirements.txt
```

## Usage

To train the DQN agent, you can run the `dqn.py` script. This will:

1.  Create a sample environment with a random image and a ground truth bounding box.
2.  Initialize the DQN agent.
3.  Train the agent for a specified number of episodes.
4.  Save the trained model to `dqn_object_detection.pth`.
5.  Plot the training statistics and save them to `training_stats.png`.
6.  Evaluate the trained agent.

```bash
python dqn.py
```

You can also use the `env.py` script to see an example of the environment running with random actions.

```bash
python env.py
```
