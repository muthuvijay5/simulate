import matplotlib.pyplot as plt
from random import choice


d = []
s = 1
for i in range(10000):
    d.append(s)
    s -= 0.0001

d = d + d + d + d

s = -2
for i in range(1700):
    d.append(s)
    s -= 0.0001

x = [choice(d) for i in range(1000)]
y = [i + 1 for i in range(1000)]

plt.plot(y, x)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Rewards")
plt.show()
