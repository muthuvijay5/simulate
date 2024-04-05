# import pandas as pd

# res = pd.read_csv("res1.csv")
# rewards = res["reward"]
# neg = 0
# pos = 0
# for i in rewards:
#     if i <= 0:
#         neg += 1
#     else:
#         pos += 1

# print(neg, pos)


import pandas as pd

res = pd.read_csv("res4.csv")
rewards = res["reward"]
r = []
for i in range(30000):
    if res["action"][i] == 3:
        r.append(rewards[i])
rewards = r
neg = 0
pos = 0
for i in rewards:
    if i <= 0:
        neg += 1
    else:
        pos += 1

print(neg, pos, pos + neg)
