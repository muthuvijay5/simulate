import torch
import torch.nn as nn
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt

r = []
r1 = pd.read_csv("train.csv")["speed"][:500]
r2 = []
spd = []


def update_transmission(dataset):
    # Get number of nodes
    num_nodes = dataset["num_vehicles"]

    # Create nodes
    nodes = range(num_nodes)

    # Create edges
    edges = [(0, i) for i in nodes if i != 0]

    # Node features
    node_features = tf.constant(
        [[dataset["current_vehicle"]] + dataset["vehicle_positions"]]
    )

    # Graph layer
    graph_layer = Dense(16, input_shape=(1, num_nodes - 1, 1))

    # Node embeddings
    node_embeddings = [graph_layer(node_features[:, i, :]) for i in nodes]

    # Edge features
    # edge_features = tf.constant(dataset["transmission_quality"])
    # edge_features = tf.reshape(edge_features, (1, -1, 2))

    # # Edge embeddings
    # edge_embeddings = graph_layer(edge_features)

    # Update transmission
    for i, edge in enumerate(edges):
        u, v = edge
        dist = tf.norm(node_embeddings[u] - node_embeddings[v])
        dataset["transmission_quality"][i] = 1 / (1 + dist)

    correct_quality = []
    for p in dataset["transmission_quality"]:
        correct_quality.append(float(p))

    return correct_quality


def flatten_state(state):
    state_tensor = torch.zeros(
        4 + (state["num_vehicles"] - 1) * 4 + len(state["obstacle_positions"]) * 2,
        dtype=torch.long,
    )

    state_tensor[0] = state["current_vehicle"][0]
    state_tensor[1] = state["current_vehicle"][1]
    state_tensor[2] = state["current_vehicle_speed"]
    state_tensor[3] = state["num_lane"]
    idx = 4
    i = 0
    for x_pos, y_pos in state["vehicle_positions"]:
        state_tensor[idx] = x_pos
        idx += 1
        state_tensor[idx] = y_pos
        idx += 1
        state_tensor[idx] = state["speed"][i]
        idx += 1
        state_tensor[idx] = int(state["transmission_quality"][i] * 10000)
        idx += 1
        i += 1

    for x_pos, y_pos in state["obstacle_positions"]:
        state_tensor[idx] = x_pos
        idx += 1
        state_tensor[idx] = y_pos
        idx += 1

    return state_tensor


class QNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_actions)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class SARSAAgent:
    def __init__(
        self,
        input_size,
        num_actions,
        learning_rate,
        discount_factor,
        exploration_prob,
        hidden_size,
    ):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob

        # Create the Q-network
        self.q_network = QNetwork(input_size, hidden_size, num_actions)

    def select_action(self, state_data, state_tensor, transmission_features):
        transmission_quality = state_data["transmission_quality"]
        transmission_features = torch.tensor([transmission_quality], dtype=torch.long)
        transmission_features = transmission_features.view(1, -1)

        if random.uniform(0, 1) < self.exploration_prob:
            return random.choice(range(self.num_actions))
        else:
            q_values = self.q_network(state_tensor)
            action = torch.argmax(q_values).item()
            return action % self.num_actions

    def update_q_network(self, state, action, reward, next_state, next_action):
        state_tensor = state.clone().detach()
        next_state_tensor = next_state.clone().detach()

        # Use the Q-network to predict Q-values for the current and next states
        q_values = self.q_network(state_tensor)
        next_q_values = self.q_network(next_state_tensor)

        # Compute the target Q-value using the next action
        target = reward + self.discount_factor * next_q_values[0, next_action]

        # Compute the loss and update the Q-network
        loss = nn.MSELoss()(q_values[0, action], target)
        self.q_network.zero_grad()
        loss.backward()
        with torch.no_grad():
            for param in self.q_network.parameters():
                param.data -= self.learning_rate * param.grad

        return loss.item()


class Environment:
    def __init__(self, dataset):
        self.dataset = dataset

    def step(self, dataset, action):
        reward = self.calculate_reward(dataset, action)
        return reward

    def calculate_reward(self, dataset, action):
        reward1 = 0
        reward2 = 0
        reward3 = 0

        if dataset["current_vehicle"][0] == 0 and action == 2:
            reward1 -= 1
        elif dataset["current_vehicle"][1] == dataset["num_lane"] and action == 3:
            reward1 -= 1
        else:
            reward1 += 1

        next_position = dataset["current_vehicle"]
        if action == 1:
            next_position[1] += 1
        if action == 2:
            next_position[0] -= 1
        if action == 3:
            next_position[0] += 1

        if (
            next_position in dataset["vehicle_positions"]
            or next_position in dataset["obstacle_positions"]
        ):
            reward2 -= 1
        else:
            reward2 += 1

        for x, y in dataset["vehicle_positions"]:
            if (
                x == next_position[0]
                and ((x - next_position[0]) * 2 + (y - next_position[1]) * 2) ** 0.5
                < safety_distance
            ):
                reward3 -= 2

        return reward1 * lambda1 + reward2 * lambda2 + reward3 * lambda3


lambda1 = 0.7
lambda2 = 0.7
lambda3 = 0.4
safety_distance = 10
learning_rate = 0.1
discount_factor = 0.9
exploration_prob = 0.3

input_features = 5
hidden_features = 64
output_features = 5

models = {}
ds = pd.read_csv("dataset.csv")

train = pd.DataFrame()
data = []
speed = []

for i in range(ds.shape[0]):
    state_data = {
        "num_lane": ds["num_lane"][i],
        "num_vehicles": ds["num_vehicles"][i],
        "current_vehicle": eval(ds["current_vehicle"][i]),
        "current_vehicle_speed": ds["current_vehicle_speed"][i],
        "vehicle_positions": eval(ds["vehicle_positions"][i]),
        "speed": eval(ds["speed"][i]),
        "obstacle_positions": eval(ds["obstacle_positions"][i]),
        "transmission_quality": eval(ds["transmission_quality"][i]),
    }

    input_size = (
        4
        + (state_data["num_vehicles"] - 1) * 4
        + len(state_data["obstacle_positions"]) * 2
    )
    num_actions = 4
    hidden_size = 64

    sarsa_agent = None
    if input_size in models:
        sarsa_agent = models[input_size]
    else:
        sarsa_agent = SARSAAgent(
            input_size,
            num_actions,
            learning_rate,
            discount_factor,
            exploration_prob,
            hidden_size,
        )

    env = Environment(state_data)
    transmission_quality = state_data["transmission_quality"]
    transmission_features = torch.tensor([transmission_quality], dtype=torch.long)

    transmission_features = transmission_features.view(1, -1)
    action = sarsa_agent.select_action(
        state_data,
        flatten_state(state_data),
        transmission_features,
    )
    reward = env.step(state_data, action)
    next_state = state_data

    next_action = sarsa_agent.select_action(
        next_state,
        flatten_state(next_state),
        torch.tensor([next_state["transmission_quality"]], dtype=torch.long).view(
            1, -1
        ),
    )

    # Update the Q-network using SARSA update
    sarsa_agent.update_q_network(
        flatten_state(state_data),
        action,
        reward,
        flatten_state(next_state),
        next_action,
    )

    speed_2d = [
        [
            state_data["current_vehicle"][0],
            state_data["current_vehicle"][1],
            state_data["current_vehicle_speed"],
            action,
        ]
    ] + [
        [
            state_data["vehicle_positions"][i][0],
            state_data["vehicle_positions"][i][1],
            state_data["speed"][i],
            state_data["transmission_quality"][i],
        ]
        for i in range(state_data["num_vehicles"] - 1)
    ]

    speed_vector = []
    for p in speed_2d:
        for q in p:
            speed_vector.append(q)

    data.append(str(speed_vector))

    if action == 0:
        speed.append(
            state_data["current_vehicle_speed"]
            + random.choice([i for i in range(-15, -9)])
        )
    elif action == 1:
        speed.append(
            state_data["current_vehicle_speed"]
            + random.choice([i for i in range(10, 16)])
        )
    else:
        speed.append(
            state_data["current_vehicle_speed"]
            + random.choice([i for i in range(-5, 6)])
        )

    models[input_size] = sarsa_agent
    next_transmission_quality = update_transmission(state_data)
    print(f"Action = {action}")
    print(f"Communication = {next_transmission_quality}\n")

train["data"] = data
train["speed"] = speed
train.to_csv("train.csv")
