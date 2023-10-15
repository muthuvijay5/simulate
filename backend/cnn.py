import tensorflow as tf
from tensorflow.keras import models, layers
import pandas as pd
from collections import defaultdict


dataset = pd.read_csv("train.csv")
data = defaultdict(list)
cnn = {}


for ip, speed in zip(dataset["data"], dataset["speed"]):
    inp = []
    for p in eval(ip):
        for q in p:
            inp.append(q)

    data[len(inp)].append((inp, speed))

for d in data.values():
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(len(d[0][0]), 1)))
    model.add(layers.Conv1D(32, 1, activation="relu"))
    model.add(layers.Conv1D(64, 1, activation="relu"))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(1))
    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

    inp = []
    speed = []

    for i, j in d:
        inp.append(i)
        speed.append(j)

    print(inp)
    model.fit(inp, speed, epochs=20)
    cnn[len(inp[0])] = model
    print(len(inp))
    model.save(f"{len(inp[0])}.h5")
