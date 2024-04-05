import pandas as pd

ds = pd.read_csv("obs.csv")
op = ds["obstacle_positions"]

for i in range(len(op)):
    x = eval(op[i])
    for j in range(len(x)):
        x[j] = [x[j][0], x[j][1], x[j][0] - 2, x[j][1] - 2]
    op[i] = x

n = pd.DataFrame()
n["obstacle_positions"] = op

n.to_csv("new.csv")
