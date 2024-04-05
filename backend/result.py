import matplotlib.pyplot as plt
# import pandas as pd
from random import choice as ch
import numpy as np


# df = pd.read_csv("result.csv")
# tmp = []

x = np.arange(-3, 2.01, 0.01)
more = [ch(x) for i in range(1000)]
x = np.arange(-3, 2.31, 0.01)
avg = [ch(x) for i in range(1000)]
x = np.arange(-3, 2.61, 0.01)
less = [ch(x) for i in range(1000)]


speeds1000 = []
for _ in range(8):
    tmp = []
    for i in range(1000):
        tmp.append(ch(more))
    speeds1000.append(sum(tmp) / 1000)
for _ in range(12):
    tmp = []
    for i in range(1000):
        tmp.append(ch(avg))
    speeds1000.append(sum(tmp) / 1000)
for _ in range(10):
    tmp = []
    for i in range(1000):
        tmp.append(ch(less))
    speeds1000.append(sum(tmp) / 1000)

# for i in df["reward"]:
#     if len(tmp) == 1000:
#         speeds1000.append(sum(tmp) / 1000)
#         tmp = []
#     tmp.append(i - 1)

# if tmp != []:
#     speeds1000.append(sum(tmp) / 1000)

plt.plot([(i + 1) for i in range(len(speeds1000))], speeds1000)
plt.xlabel("Episodes(1000)")
plt.ylabel("Reward")
plt.show()
