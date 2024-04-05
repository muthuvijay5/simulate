import torch
import torch.nn as nn
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


def ic(r1x1, r1y1, r1x2, r1y2, r2x1, r2y1, r2x2, r2y2):
    if r1x2 < r2x1 or r2x2 < r1x1:
        return False
    if r1y2 < r2y1 or r2y2 < r1y1:
        return False
    return True


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
        6 + (state["num_vehicles"] - 1) * 6 + len(state["obstacle_positions"]) * 2,
        dtype=torch.long,
    )

    state_tensor[0] = state["current_vehicle"][0]
    state_tensor[1] = state["current_vehicle"][1]
    state_tensor[2] = state["current_vehicle"][2]
    state_tensor[3] = state["current_vehicle"][3]
    state_tensor[4] = state["current_vehicle_speed"]
    state_tensor[5] = state["num_lane"]
    idx = 6
    i = 0
    for x1, y1, x2, y2 in state["vehicle_positions"]:
        state_tensor[idx] = x1
        idx += 1
        state_tensor[idx] = y1
        idx += 1
        state_tensor[idx] = x2
        idx += 1
        state_tensor[idx] = y2
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


class QLearningAgent:
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

    def update_q_network(self, state, action, reward, next_state):
        state_tensor = state.clone().detach()
        next_state_tensor = next_state.clone().detach()

        # Use the Q-network to predict Q-values for the current and next states
        q_values = self.q_network(state_tensor)
        next_q_values = self.q_network(next_state_tensor)

        # Compute the target Q-value
        target = reward + self.discount_factor * torch.max(next_q_values)

        # Compute the loss and update the Q-network
        loss = nn.MSELoss()(q_values[action], target)
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
        elif dataset["current_vehicle"][3] == dataset["num_lane"] and action == 3:
            reward1 -= 1
        else:
            reward1 += 1

        next_position = dataset["current_vehicle"]
        if action == 1:
            next_position[1] += 1
            next_position[3] += 1
        if action == 2:
            next_position[0] -= 1
            next_position[2] -= 1
        if action == 3:
            next_position[0] += 1
            next_position[2] += 1

        for x, y in dataset["obstacle_positions"]:
            if ic(
                next_position[0],
                next_position[1],
                next_position[2],
                next_position[3],
                x,
                y,
                x,
                y,
            ):
                reward2 -= 1

        if next_position in dataset["vehicle_positions"]:
            reward2 -= 1
        else:
            reward2 += 1

        for x1, y1, x2, y2 in dataset["vehicle_positions"]:
            if x1 <= next_position[0] <= x2 or x1 <= next_position[2] <= x2:
                if (
                    abs(next_position[1] - y1) < safety_distance
                    or abs(next_position[1] - y2) < safety_distance
                    or abs(next_position[3] - y1) < safety_distance
                    or abs(next_position[3] - y2) < safety_distance
                ):
                    reward3 -= 1
        else:
            flag = False
            for x, y in dataset["obstacle_positions"]:
                if next_position[0] <= x <= next_position[2]:
                    if abs(y - next_position[1]) < safety_distance:
                        flag = True
                        break
            if not flag:
                if action != 1:
                    reward3 -= 1
            else:
                reward3 += 1

        return reward1 * lambda1 + reward2 * lambda2 + reward3 * lambda3


lambda1 = 1  # random.random()
lambda2 = 1  # random.random()
lambda3 = 1  # random.random()
# lambda1 = 0.7  # random.random()
# lambda2 = 0.7  # random.random()
# lambda3 = 0.4  # random.random()
safety_distance = 15
learning_rate = 0.1
discount_factor = 0.9
exploration_prob = 0.3

input_features = 5
hidden_features = 64
output_features = 5

models = {}

# cnn = {i: load_model(f"{i}.h5") for i in range(4, 121, 4)}

ds = pd.read_csv("dataset.csv")

train = pd.DataFrame()
data = []
speed = []
res = pd.DataFrame()
actions = []
rewards = []

# for i in range(100):
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

    next_transmission_quality = update_transmission(state_data)
    state_data["transmission_quality"] = next_transmission_quality

    input_size = (
        6
        + (state_data["num_vehicles"] - 1) * 6
        + len(state_data["obstacle_positions"]) * 2
    )
    num_actions = 4
    hidden_size = 64

    q_agent = None
    if input_size in models:
        q_agent = models[input_size]
    else:
        q_agent = QLearningAgent(
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
    action = q_agent.select_action(
        state_data,
        flatten_state(state_data),
        transmission_features,
    )
    reward = env.step(state_data, action)
    next_state = state_data  # Update this with your next state logic if needed

    # Update the Q-table
    q_agent.update_q_network(
        flatten_state(state_data),
        action,
        reward,
        flatten_state(next_state),
    )

    actions.append(action)
    rewards.append(reward)

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

    # r1.append(state_data["current_vehicle_speed"])
    # r2.append(cnn[len(speed_vector)].predict([speed_vector])[0][0])

    # data.append(str(speed_vector))

    # if action == 0:
    #     speed.append(
    #         state_data["current_vehicle_speed"]
    #         + random.choice([i for i in range(-15, -9)])
    #     )
    # elif action == 1:
    #     speed.append(
    #         state_data["current_vehicle_speed"]
    #         + random.choice([i for i in range(10, 16)])
    #     )
    # else:
    #     speed.append(
    #         state_data["current_vehicle_speed"]
    #         + random.choice([i for i in range(-5, 6)])
    #     )

    models[input_size] = q_agent

    # next_speed = cnn[len(speed_vector)].predict([speed_vector])
    # print(f"Action = {action}")
    # print(f"Speed = {next_speed[0][0]}")
    # print(f"Communication = {next_transmission_quality}\n")

    # r.append(reward)
    # spd.append(next_speed[0][0])

    # if len(r2) == 500:
    # plt.figure(figsize=(18, 6))
    # plt.plot([(i + 1) for i in range(len(r))], r)
    # plt.xlabel("Episode")
    # plt.ylabel("Reward")
    # plt.title("Rewards")
    # plt.show()
    # r = []
    # plt.plot([i + 1 for i in range(500)], r1, label='Expected', color='blue')
    # plt.plot([i + 1 for i in range(500)], r2, label='Predicted', color='red')
    # plt.xlabel('Episodes')
    # plt.ylabel('Speed')
    # plt.legend()
    # Show the plot
    # plt.show()

# train["data"] = data
# train["speed"] = speed
# train.to_csv("train.csv")


# plt.plot([(i + 1) for i in range(len(r))], r)
# plt.xlabel("Episode")
# plt.ylabel("Reward")
# plt.title("Rewards")
# plt.show()


# res = pd.DataFrame()
# res["speed"] = spd
# res["reward"] = r
# res.to_csv("result.csv")
res["action"] = actions
res["reward"] = rewards
res.to_csv("res1.csv")


"""
Library code changes:
    C:/Users/Cheems/AppData/Local/Programs/Python/Python311/Lib/site-packages/torch/nn/modules/linear.py", line 114
    Add:
        input = input.to(self.weight.dtype)
"""
