import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from keras.models import load_model
import pandas as pd


df = pd.read_csv("train2.csv")
test_input = [eval(i) for i in df["data"]]
test_input = test_input[:10000]
test_speed = df["speed"][:10000]


ni = []
ns = []

for i in range(10000):
    if test_input[i][3] == 3:
        ni.append(test_input[i])
        ns.append(test_speed[i])

loaded_model = load_model("last.h5")
predictions = loaded_model.predict(ni)

# Flatten the predictions and test_speed arrays if needed
# predictions = np.ndarray.flatten(predictions)
# test_speed = np.ndarray.flatten(test_speed)

# Calculate R2
r2 = r2_score(ns, predictions)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(ns, predictions))

# Print the R2 and RMSE values
print("R-squared (R2):", r2)
print("Root Mean Squared Error (RMSE):", rmse)
print(mean_absolute_error(ns, predictions))