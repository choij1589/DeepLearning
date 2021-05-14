import random
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error

# Split a dataset based on an attribute and an attribute value
# implement a split method that seperate the dataset into the left branch and the right branch
def branch_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] > value:
            right.append(row)
        else:
            left.append(row)
        # todo code
    return left, right


# implement the standard deviation method to select the splitting point
def std_index(groups, total):
    # get total
    # use MEDV for learning
    if len(groups[0]) == 0 or len(groups[1]) == 0:
        return 0.0
    left_and_right = np.array(groups[0][-1] + groups[1][-1])
    left = np.array(groups[0][-1])
    right = np.array(groups[1][-1])
    all_std = np.var(left_and_right) * len(left_and_right)
    left_std = np.var(left) * len(left)
    right_std = np.var(right) * len(right)
    s_total = all_std - (left_std + right_std)
    return s_total


# Select the best split point for a dataset
def get_split(dataset, c_idx=None):
    dt_length = len(dataset)
    b_index, b_value, b_score, b_groups = -1, -1.0, 0.0, None
    # loop over this column index to find split points
    if c_idx is None:
        c_idx = list(range(len(dataset[0]) - 1))
    for index in c_idx:
        for row in dataset:
            # get left | right at the current row-index
            groups = branch_split(index, row[index], dataset)
            std_score = std_index(groups, dt_length)
            matching_criteria = False
            if std_score > b_score:
                matching_criteria = True
            if matching_criteria:
                b_index, b_value, b_score, b_groups = (
                    index,
                    row[index],
                    std_score,
                    groups,
                )
    return {"index": b_index, "value": b_value, "groups": b_groups}


# Create a terminal node value
# generate the prediction result if the tree reaches to this leaf
def to_terminal(group):
    labels = [row[-1] for row in group]
    # implement the aggregation method => make prediction
    outputs = np.mean(labels)
    return outputs


# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth, c_idx=None):
    # remove selected index
    # I'm not sure it would work
    #print(c_idx, depth, node["index"])
    # if c_idx == None:
    #     Warning("조교야 정신차려")
    # if depth != max_depth:
    #    c_idx.remove(node['index'])

    left, right = node["groups"]
    del node["groups"]
    # check for a no split
    if not left or not right:
        node["left"] = node["right"] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node["left"], node["right"] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node["left"] = to_terminal(left)
    else:
        node["left"] = get_split(left, c_idx)
        split(node["left"], max_depth, min_size, depth + 1, c_idx)
    # process right child
    if len(right) <= min_size:
        node["right"] = to_terminal(right)
    else:
        node["right"] = get_split(right, c_idx)
        split(node["right"], max_depth, min_size, depth + 1, c_idx)


# Build a decision tree
def build_tree(train, max_depth, min_size, c_idx=None):
    root = get_split(train, c_idx)
    split(root, max_depth, min_size, 1, c_idx)
    return root


# Make a prediction with a decision tree
def predict(node, row):
    if row[node["index"]] < node["value"]:
        if isinstance(node["left"], dict):
            return predict(node["left"], row)
        else:
            return node["left"]
    else:
        if isinstance(node["right"], dict):
            return predict(node["right"], row)
        else:
            return node["right"]


# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size, c_idx=None):
    tree = build_tree(train, max_depth, min_size, c_idx)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return predictions


# random forest implementation
# 1. implement a shuffle method to shuffle train set
# 2. implement a feature random seletion method for each tree, by that it can split branches
# 3. implement an ensemble method to aggregate the predicted results


def random_forest(n_trees, train_set, test_set, max_depth, min_size, n_features=5):
    np.random.seed(12)
    predictions = []

    for _ in range(n_trees):
        features = list(range(13))
        random.shuffle(features)
        features = features[0:5]
        predictions.append(
            decision_tree(train_set, test_set, max_depth, min_size, features)
        )
    # todo code
    # hint: create multiple decision trees
    # aggregate their prediction results => final results
    predictions = np.mean(predictions, axis=0)
    return predictions

