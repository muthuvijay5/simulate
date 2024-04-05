import torch
import torch.nn as nn
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


def ic(x11, y11, x12, y12, x21, y21, x22, y22):
    if x12 < x21 or x22 < x11:
        return False
    if y12 < y21 or y22 < y11:
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


class ActorNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_actions)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.tanh(self.fc2(x))  # For continuous action spaces
        return x


class CriticNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size + num_actions, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)  # Q value

    def forward(self, state, action):
        t = None
        try:
            t = list(action[0])
        except:
            t = list(action)
        x = torch.Tensor([list(state[0]) + t])
        # x = torch.cat([state, action], 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DDPGAgent:
    def __init__(
        self,
        input_size,
        num_actions,
        learning_rate,
        discount_factor,
        tau,  # for soft update
        hidden_size,
    ):
        self.num_actions = num_actions

        # Create Actor and Critic Networks
        self.actor_network = ActorNetwork(input_size, hidden_size, num_actions)
        self.actor_target = ActorNetwork(input_size, hidden_size, num_actions)
        self.actor_optimizer = optim.Adam(
            self.actor_network.parameters(), lr=learning_rate
        )

        self.critic_network = CriticNetwork(input_size, hidden_size, num_actions)
        self.critic_target = CriticNetwork(input_size, hidden_size, num_actions)
        self.critic_optimizer = optim.Adam(
            self.critic_network.parameters(), lr=learning_rate
        )

        self.discount_factor = discount_factor
        self.tau = tau

    def select_action(self, state):
        state_tensor = torch.from_numpy(state.clone().numpy()).float().unsqueeze(0)
        action = self.actor_network(state_tensor)
        return action.detach().numpy()[0]

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).float()
        rewards = torch.from_numpy(rewards).float().unsqueeze(1)
        next_states = torch.from_numpy(next_states).float()
        dones = torch.from_numpy(dones.astype(np.uint8)).float().unsqueeze(1)

        # Critic loss
        Qvals = self.critic_network(states, actions)
        next_actions = actions
        next_Q = self.critic_target(next_states, next_actions)
        Qprime = rewards + (1 - dones) * self.discount_factor * next_Q
        critic_loss = F.mse_loss(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic_network(states, self.actor_network(states)).mean()

        # Update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update target networks
        for target_param, param in zip(
            self.actor_target.parameters(), self.actor_network.parameters()
        ):
            target_param.data.copy_(
                param.data * self.tau + target_param.data * (1.0 - self.tau)
            )

        for target_param, param in zip(
            self.critic_target.parameters(), self.critic_network.parameters()
        ):
            target_param.data.copy_(
                param.data * self.tau + target_param.data * (1.0 - self.tau)
            )


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

        slow = False
        for x1, y1, x2, y2 in dataset["vehicle_positions"]:
            if x1 <= next_position[0] <= x2 or x1 <= next_position[2] <= x2:
                if (
                    abs(next_position[1] - y1) < safety_distance
                    or abs(next_position[1] - y2) < safety_distance
                    or abs(next_position[3] - y1) < safety_distance
                    or abs(next_position[3] - y2) < safety_distance
                ):
                    reward3 -= 1
                    slow = True
                    break
        else:
            flag = False
            for x, y in dataset["obstacle_positions"]:
                if next_position[0] <= x <= next_position[2]:
                    if abs(y - next_position[1]) < safety_distance:
                        flag = True
                        break
            if not flag:
                if action != 1 and slow == False:
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
tau = 0.3

input_features = 5
hidden_features = 64
output_features = 5

models = {}

# cnn = {i: load_model(f"{i}.h5") for i in range(4, 121, 4)}

# ds = pd.read_csv("ds.csv")
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
        q_agent = DDPGAgent(
            input_size,
            num_actions,
            learning_rate,
            discount_factor,
            tau,
            hidden_size,
        )

    env = Environment(state_data)
    transmission_quality = state_data["transmission_quality"]
    transmission_features = torch.tensor([transmission_quality], dtype=torch.long)

    transmission_features = transmission_features.view(1, -1)
    action = q_agent.select_action(flatten_state(state_data))
    action_value = torch.argmax(torch.from_numpy(action).float()).item()
    reward = env.step(state_data, action_value)
    next_state = state_data  # Update this with your next state logic if needed

    # Update the Q-table
    q_agent.update(
        np.array([flatten_state(state_data)]),
        np.array(action),
        np.array([reward]),
        np.array([flatten_state(state_data)]),
        np.array([False]),
    )

    actions.append(action_value)
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
res.to_csv("res4.csv")


"""
Library code changes:
    C:/Users/Cheems/AppData/Local/Programs/Python/Python311/Lib/site-packages/torch/nn/modules/linear.py", line 114
    Add:
        input = input.to(self.weight.dtype)
"""
