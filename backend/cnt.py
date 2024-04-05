# import pandas as pd

# res = pd.read_csv("res3.csv")
# actions = res["action"]
# arr = [0, 0, 0, 0]

# for i in actions:
#     arr[i] += 1

# print(arr)


import pandas as pd

res1 = pd.read_csv("res2.csv")
res2 = pd.read_csv("res3.csv")
a1, a2 = res1["action"], res2["action"]
arr = [0, 0, 0, 0]

for i, j in zip(a1, a2):
    if i == j:
        arr[i] += 1

print(arr)
print(sum(arr))
