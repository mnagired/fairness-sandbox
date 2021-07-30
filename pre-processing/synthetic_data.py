'''

this file generates synthetic data

'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

'''

Helper Functions

'''

def get_num_feats(ranges, num_types, n):
    total = []
    for i in range(len(ranges)):
        lower, upper = ranges[i]
        if num_types[i] == 1:
            arr = np.random.randint(low = lower, high = upper, size = (n,1))
        else:
            assert num_types[i] == 0, \
            "Error! num_types element must be 1 (integer) or 0 (float)"
            arr = np.random.uniform(low = lower, high = upper, size = (n,1))
        total.append(arr)
    return np.concatenate(total, axis = 1)

# create binary sensitive attribute
def get_sensitive_feat(n, r):
    num_minority = int(r * n)
    num_majority = n - num_minority

    minority = np.zeros((num_minority, 1))
    majority = np.ones((num_majority, 1))

    sens_feat = np.vstack((minority, majority))

    # shuffle so as to ensure randomness
    np.random.shuffle(sens_feat)

    return sens_feat

def get_cat_feats(num_cat_feats, cat_feats_levels, n):
    cat_feats = []
    for i in range(num_cat_feats):
        levels = cat_feats_levels[i]
        if levels < 2:
            raise ValueError("Categorical features must have at least 2 classes!")
        vals = np.arange(levels)
        cat = np.random.choice(vals, n, [0.5,0.5]).reshape(n, 1)
        cat_feats.append(cat)
    return np.hstack((cat_feats))

def distribution_plot(outcome_min = [], outcome_maj = [], outcome = [],
                      threshold_min = 0.5, threshold_maj = 0.5,
                      diff_dist = False):

    if diff_dist:

        plt.figure(figsize=(17,7))

        plt.subplot(1,2,1)
        plt.hist(outcome_min,label='continuous outcome',bins='auto')
        plt.axvline(threshold_min,color='red',label='threshold')
        plt.xlabel("Continuous Outcome")
        plt.ylabel("Number of Samples")
        plt.title("Minority")

        plt.subplot(1,2,2)
        plt.hist(outcome_maj,label='continuous outcome',bins='auto')
        plt.axvline(threshold_maj,color='red',label='threshold')
        plt.xlabel("Continuous Outcome")
        plt.ylabel("Number of Samples")
        plt.title("Majority")
        plt.show()

    else:

        plt.figure(figsize=(17,7))
        plt.subplot(1,2,1)
        plt.hist(outcome,label='continuous outcome',bins='auto')
        plt.axvline(threshold_min,color='red',label='threshold')
        plt.xlabel("Continuous Outcome")
        plt.ylabel("Number of Samples")
        plt.title("Distribution of Outcomes")
        plt.show()

def get_attribute_names(df, num_numerical_cols, num_cat_cols):
    col_names = []
    for i in range(num_numerical_cols):
        col_names.append('num' + str(i+1))
    for i in range(num_cat_cols):
        col_names.append('cat' + str(i+1))
    col_names.append('sens_feat')
    col_names.append('outcome')

    return col_names

# flip labels with probability eta
def flip_labels(df_synthetic, label_noise):
    labels = df_synthetic['outcome']

    for i in range(len(labels)):
        if random.uniform(0,1) <= label_noise:
            labels[i] = 1 if labels[i] == 0 else 0
    df_synthetic['outcome'] = labels

    return df_synthetic

'''

Main Function to Generate Data

'''

'''
Parameters:

    n is the total number of examples in the dataset

    num_numerical_feats is number of numerical features
        each numerical feature is drawn from a
        multivariate normal distribution

    ranges = list with range of values for each numerical feature
        each element of ranges is a range [x,y)
        default is [0,1)
        NOTE: len(ranges) == num_numerical_feats

    num_types: list of values such that
        num_types[i] = 1 if integer, 0 if float for numerical feature i
        default is all integers
        NOTE: len(num_types) == num_numerical_feats

    num_cat_feats is number of categorical features

    cat_levels is an array where each element is the number
        of levels for each categorical feature
        len(cat_levels) = num_cat_feats

    r is the proportion of examples in the minority group
        (1-r) is proportion of examples in majority group

    label_noise is in [0,1]

    diff_dist is true if minority and majority have different
        underlying sampling distributions

    show_vis displays the distribution of outcomes

'''

def get_synthetic_data(n, r, num_numerical_feats, num_cat_feats,
                       ranges = [], num_types = [],
                       cat_levels = [], label_noise = 0,
                       diff_dist = False, show_vis = False):

    assert 0 < r < 1, "R must be in [0,1]"
    num_min = int(n*r)
    num_maj = n - num_min

    cat_probs = list(np.multiply(np.ones(num_cat_feats),0.5))

    # numerical feature params
    if ranges == []:
        ranges = [(0,1) for i  in range(num_numerical_feats)]

    assert len(ranges) == num_numerical_feats, \
    "Error! len(ranges) != num_numerical_feats"

    if num_types == []:
        num_types = list(np.ones(num_numerical_feats))

    assert len(num_types) == num_numerical_feats, \
    "len(num_types) != num_numerical_feats"

    # generating the features

    num_features_min = get_num_feats(ranges, num_types, num_min)
    num_features_maj = get_num_feats(ranges, num_types, num_maj)
    num_features = np.concatenate((num_features_min, num_features_maj))

    # binary sensitive attribute, 0: minority, 1: majority
    sens_feat = get_sensitive_feat(r=r, n=n)

    assert len(cat_levels) == num_cat_feats, \
    "Each categorical feature must have a specification for its number of levels"
    cat_feats = get_cat_feats(num_cat_feats, cat_levels, n)

    # generating outcomes (continuous and binary)
    if diff_dist:
        # causal effect params
        effect_param_min = [0.5, -0.2, 0.1]
        effect_param_maj = [-0.7, 0.5, 1.5]
        outcome_continuous_min = 1/(1+np.exp(-np.matmul(num_features_min,effect_param_min))) # logit model + no added noise
        outcome_continuous_maj = 1/(1+np.exp(-np.matmul(num_features_maj,effect_param_maj)))
        outcome_binary_min = np.where(outcome_continuous_min >= 0.5, 1, 0) # logistic decision boundary
        outcome_binary_maj = np.where(outcome_continuous_maj >= 0.5, 1, 0)
        outcome_binary = np.hstack((outcome_binary_min, outcome_binary_maj)).reshape(n,1)
        if show_vis:
            distribution_plot(outcome_continuous_min, outcome_continuous_maj, diff_dist=True)
    else:
        effect_param = [0.5, -0.2, 0.1]
        outcome_continuous = 1/(1+np.exp(-np.matmul(num_features,effect_param))) # logit model + no added noise
        outcome_binary = np.where(outcome_continuous >= 0.5, 1, 0).reshape(n,1) # logistic decision boundary
        if show_vis:
            distribution_plot(outcome=outcome_continuous, diff_dist=False)


    temp_data = np.hstack((num_features, cat_feats, sens_feat, outcome_binary))
    np.random.shuffle(temp_data) # randomly shuffle the data

    df_synthetic = pd.DataFrame(temp_data)
    df_synthetic.columns = get_attribute_names(df_synthetic, num_numerical_feats, num_cat_feats)

    assert 0 <= label_noise < 1, "Label noise must be in [0, 1)"
    if label_noise != 0:
        df_synthetic = flip_labels(df_synthetic, label_noise)

    return df_synthetic
