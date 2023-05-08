import pandas as pd
import numpy as np
from mlxtend.evaluate import permutation_test


df = pd.read_csv("scores.csv")


x = df["inf_only"]
y = df["inf_or_fin"]

# print(x)
# print(y)

p_value = permutation_test(x, y,
                           method='approximate',
                           num_rounds=100000,
                           seed=0)
print(p_value)

