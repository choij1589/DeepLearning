import numpy as np
import pandas as pd
from tools import decision_tree
from sklearn.metrics import mean_absolute_error

# get dataset
cols = [
    "CRIM",
    "ZN",
    "INDUS",
    "CHAS",
    "NOX",
    "RM",
    "AGE",
    "DIS",
    "RAD",
    "TAX",
    "PTRATIO",
    "B",
    "LSTAT",
    "MEDV",
]

original_data = pd.read_csv("data/housing_100x.csv", header=None, names=cols).iloc[
    1:, :
]
confused_data = pd.read_csv(
    "data/housing_100x_outlier_missing.csv", header=None, names=cols
).iloc[1:, :]

# preprocessing
# no cut for MEDV, we have to expect the value
lower_bounds = {
    "CRIM": 0.0,
    "ZN": 0.0,
    "INDUS": 0.0,
    "CHAS": 0,
    "NOX": 0.3,
    "RM": 4.0,
    "AGE": 0.0,
    "DIS": 1.0,
    "RAD": 0.0,
    "TAX": 100.0,
    "PTRATIO": 10.0,
    "B": 300.0,
    "LSTAT": 0.0,
    "MEDV": 0.0,
}
upper_bounds = {
    "CRIM": 20.0,
    "ZN": 100.0,
    "INDUS": 30.0,
    "CHAS": 1,
    "NOX": 0.9,
    "RM": 9.0,
    "AGE": 100.0,
    "DIS": 12.0,
    "RAD": 25.0,
    "TAX": 750.0,
    "PTRATIO": 25.0,
    "B": 420.0,
    "LSTAT": 40.0,
    "MEDV": 50.0,
}
fixed_outlier_data = confused_data.copy()

for feature in cols:
    for idx in confused_data.index:
        lower = lower_bounds[feature]
        upper = upper_bounds[feature]

        # CHAS should be 0 or 1, round the value
        # RAD should be an ingeger index between 0 and 24, change float to int
        if feature == "CHAS":
            fixed_outlier_data.loc[idx, "CHAS"] = int(
                round(confused_data.loc[idx, "CHAS"])
            )
        elif feature == "RAD":
            fixed_outlier_data.loc[idx, "RAD"] = int(confused_data.loc[idx, "RAD"])
        else:
            pass

        # -1 will be also fixed
        if not (
            lower <= confused_data.loc[idx, feature]
            and confused_data.loc[idx, feature] <= upper
        ):
            fixed_outlier_data.loc[idx, feature] = np.nan
fixed_missing_data = fixed_outlier_data.interpolate(method="linear")
preprocessed_data = fixed_missing_data.copy()  # will use this one

# Divide train and test dataset
shuffled_data = preprocessed_data.sample(frac=1).reset_index(drop=True)
train_ratio = 0.7
cut = int(len(shuffled_data) * train_ratio)
train_data = shuffled_data.loc[:cut, :].to_numpy()
test_data = shuffled_data.loc[cut:, :].to_numpy()

train_data_temp = train_data[0:5000, :]
test_data_temp = test_data[0:1000, :]
from itertools import combinations

combs = list(combinations(list(range(13)), 5))
labels = test_data_temp[:, -1]
for comb in combs:
    comb = list(comb)
    predicted = decision_tree(train_data_temp, test_data_temp, 5, 10, comb)
    scores = mean_absolute_error(labels, predicted)
    print(comb, round(scores, 2))
