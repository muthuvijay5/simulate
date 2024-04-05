import tensorflow as tf
from tensorflow.keras import models, layers
import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt


# dataset = pd.read_csv("train2.csv")
dataset = pd.read_csv("train.csv")
data = defaultdict(list)
cnn = {}


for ip, speed in zip(dataset["data"], dataset["speed"]):
    inp = eval(ip)
    data[len(inp)].append((inp, speed))

for d in data.values():
    print(d[0][0])
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(len(d[0][0]), 1)))
    model.add(layers.Conv1D(64, 3, activation="relu", padding="same"))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(128, 3, activation="relu", padding="same"))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1))

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    inp = []
    speed = []

    for i, j in d:
        inp.append(i)
        speed.append(j)

    history = model.fit(inp, speed, epochs=30)
    cnn[len(inp[0])] = model
    print(len(inp[0]))

    model.save(f"{len(d[0][0])}.h5")
    # Plot the accuracy graph
    # plt.plot(history.history['mae'])
    # plt.title('MAE')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.show()

    # # Plot the loss graph
    # plt.plot(history.history['loss'])
    # plt.title('Model Loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.show()

    # plot graph for 100 epochs
