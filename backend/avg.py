import pandas as pd

res1 = pd.read_csv("res1.csv")
res2 = pd.read_csv("res2.csv")
res3 = pd.read_csv("res3.csv")
res4 = pd.read_csv("res4.csv")

print(sum(res1["reward"]) / 30000)
print(sum(res2["reward"]) / 30000)
print(sum(res3["reward"]) / 30000)
print(sum(res4["reward"]) / 30000)
