'''
this file generates synthetic data
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

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
    print("yes!", r, eta, beta) if res > 0 else print("no", r, eta, beta)


'''

TRUE LABEL GENERATION

'''

# create minority and majority groups
def get_cat_features(n, r):
    num_minority = int(r * n)
    num_majority = n - num_minority

    minority = np.zeros((num_minority, 1))
    majority = np.ones((num_majority, 1))

    cat_features = np.vstack((minority, majority))

    # shuffle so as to ensure randomness
    np.random.shuffle(cat_features)

    return cat_features

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
            df_majority.columns = ['num1','num2','num3','cat','outcome']

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
            df_majority.columns = ['num1','num2','num3','cat','outcome']

            base_rate_maj = df_majority['outcome'].value_counts()[0] / len(df_majority)
            diff = round(base_rate_min,4) - round(base_rate_maj,4)
            count+=1

            # fail-safe
            if count > int(len(df_majority)/3): break

    total = np.vstack((df_majority, df_minority))

    # shuffle so as to ensure randomness
    np.random.shuffle(total)

    df_true = pd.DataFrame(pd.DataFrame(total))
    df_true.columns = ['num1','num2','num3','cat','outcome']

    print(diff*100)

    return df_true

'''

create synthetic data with:
    3 numerical features (Gaussian), 1 categorical (sensitive attribute)
    logistic outcome model s.t. outcome = Indicator[logit(effect_param*features) >= 0.5]

create minority/majority groups according to r param

simulate Bayes Optimal Classifiers for minority and majority

flip labels according to eta param

ensure equal base rates (proportion of positive examples) across both groups

'''

def true_label_generation(r, eta, n):

    '''
    delete this variable to allow user to control percentage of positively labeled examples
    eg: let outcome_continuous >= 0.2 implies 80% positively labeled samples
    '''
    # causal effect params
    effect_param_min = [0.5, -0.2, 0.1]
    effect_param_maj = [-0.7, 0.5, 1.5]

    num_min = int(n*r)
    num_maj = n - num_min

    # required: len(cat_probabilities) = n_cat_features
    n_cat_features = 2
    cat_probabilities = [0.5, 0.5]

    # numerical feature params
    means = [0, 0, 0]
    cov_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    # features
    cat_features = get_cat_features(r=r, n=n)
    num_features_min = np.random.multivariate_normal(means, cov_matrix, num_min)
    num_features_maj = np.random.multivariate_normal(means, cov_matrix, num_maj)

    num_features = np.concatenate((num_features_min, num_features_maj))

    # outcomes
    outcome_continuous_min = 1/(1+np.exp(-np.matmul(num_features_min,effect_param_min))) # logit model + no added noise
    outcome_continuous_maj = 1/(1+np.exp(-np.matmul(num_features_maj,effect_param_maj))) # logit model + no added noise
    outcome_binary_min = get_bayes_optimal_labels(features=num_features_min, effect_param=effect_param_min)
    outcome_binary_maj = get_bayes_optimal_labels(features=num_features_maj, effect_param=effect_param_maj)

    outcome = np.hstack((outcome_binary_min,outcome_binary_maj)).reshape(n,1)
    temp_data = np.hstack((num_features,cat_features, outcome))
    np.random.shuffle(temp_data) # randomly shuffle the data

    df_synthetic = pd.DataFrame(temp_data)
    df_synthetic.columns = ['num1','num2','num3','cat','outcome']

    df_majority = df_synthetic[df_synthetic['cat'] == 1]
    df_minority = df_synthetic[df_synthetic['cat'] == 0]

    df_synthetic = flip_labels(df_synthetic, eta)

    df_majority = df_synthetic[df_synthetic['cat'] == 1]
    df_minority = df_synthetic[df_synthetic['cat'] == 0]

    # df_synthetic = equal_base_rates(df_majority, df_minority)

    return df_synthetic 
