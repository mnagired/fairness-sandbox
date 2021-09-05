'''
this file generates synthetic data
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

'''

r is the proportion of training examples in the minority group,

which means 1-r is proportion of examples in the majority group

eta is the probability of flipping the label

n is the number of training examples

beta is the probability of keeping a positively labeled example
from the minority class

NOTE: results can be replicated if and only if the following condition holds:

(1-r)(1-2*eta) + r((1-eta)*beta - eta) > 0

'''
def get_params(r = 1/3, eta = 1/4, n = 2000, beta = 0.5):
    return r, eta, n, beta


# check if above constraint holds
def check_constraints(r, eta, beta):
    first = (1-r)*(1-2*eta)
    second = r * ((1-eta)*beta - eta)
    res = first + second
    print("constraint: ", res)
    items = "r: " + str(r) + " eta: " + str(eta) + " beta: " + str(beta)
    print("yes!\n", items) if res > 0 else print("no\n", items)


'''

DATA GENERATION

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

def get_attribute_names(df, num_numerical_cols, num_cat_cols):
    col_names = []
    for i in range(num_numerical_cols):
        col_names.append('num' + str(i+1))
    for i in range(num_cat_cols):
        col_names.append('cat' + str(i+1))
    col_names.append('sens_feat')
    col_names.append('outcome')

    return col_names

'''

TRUE LABEL GENERATION

'''

# return labels from Bayes Optimal Classifier
def get_bayes_optimal_labels(features, effect_param):
    outcome_continuous = 1/(1+np.exp(-np.matmul(features, effect_param)))
    return np.where(outcome_continuous >= 0.5, 1, 0)

# flip labels with probability eta
def flip_labels(df_synthetic, eta):
    labels = df_synthetic['outcome']

    for i in range(len(labels)):
        if random.uniform(0,1) <= eta:
            labels[i] = 1 if labels[i] == 0 else 0
    df_synthetic['outcome'] = labels

    return df_synthetic


# ensure equal proportion of positive examples across minority and majority
def equal_base_rates(df_majority, df_minority):
    cols = df_majority.columns
    base_rate_maj = df_majority['outcome'].value_counts()[0] / len(df_majority)
    base_rate_min = df_minority['outcome'].value_counts()[0] / len(df_minority)

    X_maj_pos = df_majority[df_majority['outcome'] == 1].iloc[:, :].values
    X_maj_neg = df_majority[df_majority['outcome'] == 0].iloc[:, :].values

    diff = round(base_rate_maj,4) - round(base_rate_min,4)

    print(diff*100)

    count = 0

    if diff > 0:
        while(diff > 0.01):
            X_maj_neg = np.delete(X_maj_neg, 0, axis = 0)

            df_majority = pd.DataFrame(pd.DataFrame(np.vstack((X_maj_pos, X_maj_neg))))
            df_majority.columns = cols

            base_rate_maj = df_majority['outcome'].value_counts()[0] / len(df_majority)
            diff = round(base_rate_maj,4) - round(base_rate_min,4)
            count+=1

            # fail-safe
            if count > int(len(df_majority)/3): break
    else:
        diff = round(base_rate_min,4) - round(base_rate_maj,4)
        while(diff > 0.01):
            X_maj_pos = np.delete(X_maj_pos, 0, axis = 0)

            df_majority = pd.DataFrame(pd.DataFrame(np.vstack((X_maj_pos, X_maj_neg))))
            df_majority.columns = cols

            base_rate_maj = df_majority['outcome'].value_counts()[0] / len(df_majority)
            diff = round(base_rate_min,4) - round(base_rate_maj,4)
            count+=1

            # fail-safe
            if count > int(len(df_majority)/3): break

    total = np.vstack((df_majority, df_minority))

    # shuffle so as to ensure randomness
    np.random.shuffle(total)

    df_true = pd.DataFrame(pd.DataFrame(total))
    df_true.columns = cols

    print(diff*100)

    return df_true

def distribution_plot(outcome_min = [], outcome_maj = [],
                      threshold_min = 0.5, threshold_maj = 0.5):

    plt.figure(figsize=(17,7))

    sns.distplot(outcome_min, label = 'minority')
    sns.distplot(outcome_maj, label = 'majority')

    if threshold_maj == threshold_min:
        plt.axvline(threshold_min,color='red',label='threshold')
    else:
        plt.axvline(threshold_min,color='red',label='threshold_min')
        plt.axvline(threshold_maj,color='blue',label='threshold_maj')

    plt.legend()
    plt.show()


'''

create synthetic data with:
    logistic outcome model s.t. outcome = Indicator[logit(effect_param*features) >= 0.5]

create minority/majority groups according to r param

simulate Bayes Optimal Classifiers for minority and majority

flip labels according to eta param

ensure equal base rates (proportion of positive examples) across both groups

'''

'''
Parameters:

    n is the total number of examples in the dataset
    r is the proportion of examples in the minority group
        (1-r) is proportion of examples in majority group
    eta is the probability of flipping a label

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

    show_vis = True to see distribution of outcomes for minority and majority

'''

def get_synthetic_data(n, r, eta,
                       num_numerical_feats, num_cat_feats,
                       ranges = [], num_types = [],
                       cat_levels = [], show_vis = False):

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

    # causal effect params
    effect_param_min = [0.5, -0.2, 0.1]
    effect_param_maj = [-0.7, 0.5, 1.5]

    outcome_continuous_min = 1/(1+np.exp(-np.matmul(num_features_min,effect_param_min))) # logit model + no added noise
    outcome_continuous_maj = 1/(1+np.exp(-np.matmul(num_features_maj,effect_param_maj)))
    outcome_binary_min = np.where(outcome_continuous_min >= 0.5, 1, 0) # logistic decision boundary
    outcome_binary_maj = np.where(outcome_continuous_maj >= 0.5, 1, 0)
    outcome_binary = np.hstack((outcome_binary_min, outcome_binary_maj)).reshape(n,1)
    if show_vis:
        distribution_plot(outcome_continuous_min, outcome_continuous_maj)

    temp_data = np.hstack((num_features, cat_feats, sens_feat, outcome_binary))
    np.random.shuffle(temp_data) # randomly shuffle the data

    df_synthetic = pd.DataFrame(temp_data)
    df_synthetic.columns = get_attribute_names(df_synthetic, num_numerical_feats, num_cat_feats)

    df_majority = df_synthetic[df_synthetic['sens_feat'] == 1]
    df_minority = df_synthetic[df_synthetic['sens_feat'] == 0]

    assert 0 <= eta < 1, "Eta must be in [0, 1)"
    df_synthetic = flip_labels(df_synthetic, eta)

    return df_synthetic
