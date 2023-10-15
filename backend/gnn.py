# import tensorflow as tf
# from tensorflow.keras.layers import Dense

# # Input data
# dataset = {
#     "num_vehicles": 5,
#     "current_vehicle": [0, 1],
#     "current_vehicle_speed": 65,
#     "vehicle_positions": [[1, 1], [2, 1], [3, 1], [4, 1]],
#     "speed": [70, 80, 66, 73],
#     "transmission_quality": [0.9, 0.8, 0.7, 0.85],
# }

# # Graph parameters
# num_features = 3
# num_nodes = dataset["num_vehicles"]
# nodes = range(num_nodes)
# edges = [(0, i) for i in nodes if i != 0]

# # Node features
# node_features = tf.constant(
#     [[dataset["current_vehicle"]] + dataset["vehicle_positions"]]
# )

# graph_layer = Dense(16, input_shape=(1, 4, 1))

# # Node embeddings
# node_embeddings = [graph_layer(node_features[:, i, :]) for i in nodes]

# # Edge features
# edge_features = tf.constant(dataset["transmission_quality"])
# edge_features = tf.reshape(edge_features, (1, -1, 2))

# # Edge embeddings
# edge_embeddings = graph_layer(edge_features)

# # Update transmission quality
# for i, edge in enumerate(edges):
#     u, v = edge
#     dist = tf.norm(node_embeddings[u] - node_embeddings[v])
#     dataset["transmission_quality"][i] = 1 / (1 + dist)

# for p in dataset["transmission_quality"]:
#     print(p)


import tensorflow as tf
from tensorflow.keras.layers import Dense


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

    for p in dataset["transmission_quality"]:
        print(float(p))


# Input data
dataset = {
    "num_vehicles": 8,
    "current_vehicle": [0, 1],
    "current_vehicle_speed": 65,
    "vehicle_positions": [[1, 1], [2, 1], [3, 1], [4, 1], [9, 1], [9, 1], [11, 1]],
    "speed": [70, 80, 66, 73, 80, 80, 90],
    "transmission_quality": [0.9, 0.8, 0.7, 0.85, 0.7, 0.9, 1.0],
}

update_transmission(dataset)
