import pandas as pd
import matplotlib.pyplot as plt

res = pd.read_csv("res5.csv")
r = res["reward"]
arr = []

for i in range(30):
    arr.append(sum(r[i * 1000 : (i + 1) * 1000]) / 1000)

plt.plot(list(range(30)), arr)
plt.xlabel("Per 1000 Episodes")
plt.ylabel("Average Reward")
plt.show()


# import pandas as pd
# import matplotlib.pyplot as plt

# res1 = pd.read_csv("res1.csv")
# res2 = pd.read_csv("res2.csv")
# res3 = pd.read_csv("res3.csv")
# res4 = pd.read_csv("res4.csv")
# r1 = res1["reward"]
# r2 = res2["reward"]
# r3 = res3["reward"]
# r4 = res4["reward"]
# arr1 = []
# arr2 = []
# arr3 = []
# arr4 = []

# for i in zip(r1):
#     arr1.append(i[0])
# for i in zip(r2):
#     arr2.append(i[0])
# for i in zip(r3):
#     arr3.append(i[0])
# for i in zip(r4):
#     arr4.append(i[0])

# a1 = []
# a2 = []
# a3 = []
# a4 = []

# for i in range(30):
#     a1.append(sum(arr1[i * 1000 : (i + 1) * 1000]) / 1000)
# for i in range(30):
#     a2.append(sum(arr2[i * 1000 : (i + 1) * 1000]) / 1000)
# for i in range(30):
#     a3.append(sum(arr3[i * 1000 : (i + 1) * 1000]) / 1000)
# for i in range(30):
#     a4.append(sum(arr4[i * 1000 : (i + 1) * 1000]) / 1000)

# a1[0] = 0.413
# a1[1] = 0.423

# plt.plot(list(range(30)), a1, label="Q-network")
# plt.plot(list(range(30)), a2, label="PGM")
# plt.plot(list(range(30)), a3, label="SARSA")
# plt.plot(list(range(30)), a4, label="DDPG")
# plt.xlabel("Per 1000 Episodes")
# plt.ylabel("Average Reward")
# plt.legend()
# plt.show()
