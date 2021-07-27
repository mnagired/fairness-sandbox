'''

this file contains 6 different biases

'''

import random
import pandas as pd
import numpy as np

from numpy import percentile

# to avoid warning
pd.options.mode.chained_assignment = None

'''
Representation Bias

feat_conditions: conditions of data to undersample,
    input clauses as (df['column_name'] boolean) combined with & or |,
    e.g. (df['num1'] > 0) & (df['num1'] <= 1) & (df['cat1'] == 0)
beta: probability of deleting a sample

'''

def representation(df, feat_conditions, beta):

    df_bias = df.copy()
    drop_idx = []

    for i in df_bias.index[feat_conditions]:
        if random.uniform(0,1) <= beta: drop_idx.append(i)

    return df_bias.drop(drop_idx)

'''
Measurement Bias

Add noise to an attribute, either entirely or on various subgroups

Refer to numpy.random documentation(https://numpy.org/doc/1.16/reference/routines.random.html)
for types of sampling distributions for the noise.

Helper Functions:
1. Get unique values for categorical features
2. Get 5 number summary for numerical features

'''

# Precondition: feature MUST be categorical
def get_unique_cat(df, feature):
    arr = list(np.sort(df[feature].unique()))
    return arr

# Precondition: feature MUST be numerical
def get_summary_num(df, feature):
    res = dict()
    res['attribute'] = feature

    data_min, data_max = df[feature].min(), df[feature].max()
    res['min'] = data_min
    res['max'] = data_max

    quartiles = percentile(df[feature], [25,50,75])
    res['1st Quartile'] = quartiles[0]
    res['2nd Quartile'] = quartiles[1]
    res['3rd Quartile'] = quartiles[2]

    return res

def get_noise(df, feature, noise_dist, noise_dist_params):
    num_params = len(noise_dist_params)

    if num_params == 1:
        noise = noise_dist(noise_dist_params[0])
    elif num_params == 2:
        noise = noise_dist(noise_dist_params[0], noise_dist_params[1])
    elif num_params == 3:
        noise = noise_dist(noise_dist_params[0], noise_dist_params[1],
                           noise_dist_params[2])
    else:
        raise ValueError("check your parameters for the noise distribution!")

    return noise

# inject noise with probability noise_prob
def inject_noise_cat(df_train, df_noise, feature, noise_prob):
    # get unique values for feature (from training dataset)
    unique_vals = get_unique_cat(df_train, feature)
    for i in range(len(df_noise[feature])):
        # noise injection criteria
        if random.uniform(0,1) <= noise_prob:
            # perturb specific noise value
            df_noise[feature].iloc[i] = np.random.choice(unique_vals)

    return df_noise

# inject noise with probability noise_prob
def inject_noise_num(df_noise, feature, noise_prob,
                     noise_dist, noise_dist_params):
    for i in range(len(df_noise[feature])):
        # noise injection criteria
        if random.uniform(0,1) <= noise_prob:
            # perturb specific noise value
            df_noise[feature].iloc[i] += get_noise(df_noise, feature,
                                      noise_dist, noise_dist_params)

    return df_noise

def total_measurement(df_train, df, feature, feature_type, noise_prob,
                      noise_dist, noise_dist_params, subgroups):
    err_msg = "Error! feature_type must be either numeric or categorical"
    noise = get_noise(df, feature, noise_dist, noise_dist_params)

    if feature_type == 'categorical':
        return inject_noise_cat(df_train, df, feature, noise_prob)
    else:
        assert feature_type == 'numeric', err_msg
        return inject_noise_num(df, feature, noise_prob,
                                noise_dist, noise_dist_params)

# NOTE: we assume that you have provided the proper parameters for each noise distribution call
#       for each subgroup (if applicable)
def subgroup_measurement(df_train, df, feature, feature_type, noise_prob,
                         noise_dist, noise_dist_params, subgroups):
    err_msg = "Must have as many noise_dist_param lists as number of categories for feature"
    err_msg_2 = "Error! feature_type must be either numeric or categorical"

    if feature_type == 'numeric':
        for i in range(len(subgroups)):
            lower, upper = subgroups[i]
            # isolate subgroup
            df_noise = df[(df[feature] >= lower) & (df[feature] <= upper)]
            # apply noise to subgroup
            df_noise = inject_noise_num(df_noise, feature, noise_prob,
                                        noise_dist, noise_dist_params[i])
            # modify subgroup in original dataframe
            df[(df[feature] >= lower) & (df[feature] <= upper)] = df_noise
    else:
        # maybe randomly select from get_unique_cat for the specific groups?
        assert feature_type == 'categorical', err_msg_2
        for i in range(len(subgroups)):
            # isolate subgroup
            df_noise = df[df[feature] == subgroups[i]]
            # apply noise to subgroup
            df_noise = inject_noise_cat(df_train, df_noise, feature, noise_prob)
            # modify subgroup in original dataframe
            df[df[feature] == subgroups[i]] = df_noise

    return df

'''

feature: must be a column in the dataframe
feature_type: numeric or categorical
noise_dist: distribution function from np.random library
    e.g. np.random.beta, np.random.logistic, etc.

noise_dist_params: parameters for noise distribution
    e.g. if noise_dist == np.random.normal
        then noist_dist_params could be [0, 0.1], i.e. noise w/ mean 0 and sigma 0.1
    NOTE 1: make sure number of parameters is consistent with function documentation
            EXCLUDING the final param (length of column)
    NOTE 2: if using subgroups, noise_dist_params[i] should be a list of the parameters
            for the noise distribution of subgroup[i]

noise_type: 0 if noise applied to entire attribute,
            1 if noise applied to subgroups

noise_prob: probability of applying noise to an example

subgroups: subgroups for feature
    if feature_type == numerical:
        subgroups[i] is a range of values (x, y) for subgroup i
    if feature_type == categorical:
        subgroups[i] is a single numerical value for subgroup i
            (refer to OHE if needed)

'''
def measurement(df, feature, feature_type, noise_dist = np.random.normal,
                noise_prob = 0.25, noise_dist_params = [0, 1],
                noise_type = 0, subgroups = []):

    assert noise_type in [0,1], "noise_type must be 0 or 1, see comments!"
    assert feature in list(df.columns), "feature must be a column in the dataframe!"

    df_bias = df.copy()

    if noise_type == 0:
        df_bias = total_measurement(df, df_bias, feature, feature_type, noise_prob,
                                             noise_dist, noise_dist_params, subgroups)
    else:
        df_bias = subgroup_measurement(df, df_bias, feature, feature_type, noise_prob,
                                       noise_dist, noise_dist_params, subgroups)


    return df_bias


'''
Omitted Variable Bias

Note: if you choose to remove the sensitive feature, you will no longer be able to impose a fairness intervention!
Resulting comparisons will simply be between regular ml models trained with and without the sensitive attribute.

'''

# must input a Dataset object
def omitted_variable(datasets, df, short_name, col_to_del, is_sens_attr = False):
    assert col_to_del in list(df.columns), "Column to delete must be a column in the dataframe!"
    assert short_name in datasets.keys(), "Dataset with that short name doesn't exist!"

    if is_sens_attr:
        datasets[short_name].has_sens_attr = False

    return df.drop(col_to_del, axis = 1)


'''
Label Noise Bias

add noise to labels for a specific subset of the data
    (conditioned on another feature or subgroup of another feature)


feature: must be a column in the dataframe
feature_type: numeric or categorical
subgroup_val: subgroup value for feature
    if subgroup_type == numerical:
        subgroup is a range of values (x, y)
    if subgroup_type == categorical:
        subgroup is a single numerical value
            (refer to OHE if needed)

label_noise: flip labels with probability label_noise

'''

def label_noise(df, feature, feature_type, subgroup_val, label_noise):

    assert feature in list(df.columns), "feature must be a column in the dataframe!"

    err_msg = "Error! feature_type must be either numeric or categorical"

    df_bias = df.copy()

    if feature_type == 'numeric':
        lower, upper = subgroup_val
        df_bias = df_bias[(df_bias[feature] >= lower) &
                          (df_bias[feature] <= upper)]

    else:
        assert feature_type == 'categorical', err_msg
        df_bias = df_bias[df_bias[feature] == subgroup_val]

    labels = list(df_bias['outcome'])

    for i in range(len(labels)):
        if random.uniform(0,1) <= label_noise:
            labels[i] = 1 if labels[i] == 0 else 0
    df_bias['outcome'] = labels

    if feature_type == 'numeric':
        lower, upper = subgroup_val
        df[(df[feature] >= lower) & (df[feature] <= upper)] = df_bias
    else:
        df[df[feature] == subgroup_val] = df_bias

    return df


'''
Over-Sampling Majority Class

Note: you can either choose to randomly over-sample existing examples or
      generate new samples by interpolation using SMOTE and ADASYN

Parameters:

    maj_val: value of sens_attr which indicates majority class
    min_val: value of sens_attr which indicates minority class
    sens_attr: sensitive attribute
    over_amt: amount of over-sampling to be applied to majority
        e.g. over_amt = 2 means twice as many samples in majority

'''

def random_over_sampling(df_train, sens_attr,
                         maj_val, min_val, over_amt = 2):
    df_majority = df_train[df_train[sens_attr] == maj_val]
    df_minority = df_train[df_train[sens_attr] == min_val]

    df_oversampled = df_majority.sample(int(over_amt)*len(df_majority), replace = True)

    # combine oversampled and original majority class to create dataset
    df_concat = pd.concat([df_oversampled,df_minority])

    return df_concat.sample(frac=1) # reshuffle rows of dataframe randomly


from imblearn.over_sampling import *
# to avoid warning
pd.options.mode.chained_assignment = None

'''

Parameters:

    maj_val: value of sens_attr which indicates majority class
    min_val: value of sens_attr which indicates minority class
    sens_attr: sensitive attribute
    over_amt: amount of over-sampling to be applied to majority
        e.g. over_amt = 2 means twice as many samples in majority
    type: if 1 then SMOTE, if 2 then ADASYN

'''

def over_sampling(df_train, sens_attr,
          maj_val, min_val, over_amt = 2, type = 1):

    assert type in [1,2], "Type must be 1 or 2, see comments!"

    cols = df_train.columns

    df_majority_X = df_train[df_train[sens_attr] == maj_val].drop('outcome', axis = 1)
    df_majority_y = df_train[df_train[sens_attr] == maj_val]['outcome']
    df_minority_X = df_train[df_train[sens_attr] == min_val].drop('outcome', axis = 1)
    df_minority_y = df_train[df_train[sens_attr] == min_val]['outcome']

    over_sample_amt = int(len(df_majority_X) * over_amt)

    # make original minority class into majority with label 0
    df_minority_flipped = df_minority_X.sample(over_sample_amt, replace = True)
    df_minority_flipped['outcome'] = np.zeros((len(df_minority_flipped),1))

    # make original majority have all label 0 (so it's the minority now)
    df_majority_X['outcome'] = np.ones((len(df_majority_X),1))
    df_majority = df_majority_X

    df_total = pd.concat([df_minority_flipped, df_majority])

    X_total = df_total.iloc[:, :-1].values
    y_total = df_total.iloc[:, -1].values

    if type == 1:
        over_sampler = SMOTE(random_state = 42, sampling_strategy = 'minority')
    else:
        over_sampler = ADASYN(random_state = 42, sampling_strategy = 'minority')

    X_total_resampled, y_total_resampled = over_sampler.fit_resample(X_total, y_total)

    df_res = pd.DataFrame(X_total_resampled)
    df_res['outcome'] = y_total_resampled
    df_res.columns = cols

    df_oversampled = df_res[df_res[sens_attr] == maj_val]
    oversampled_labels = df_majority_y.sample(len(df_oversampled), replace = True).values

    labels = oversampled_labels

    df_oversampled['outcome'] = labels

    # combine oversampled and original majority class to create dataset
    df_concat = pd.concat([df_oversampled,df_minority])

    return df_concat.sample(frac=1) # reshuffle rows of dataframe randomly


'''

Under-Sampling Minority Class

Note 1: you will need to input $\beta$, which is the probability of deleting
        an example from the minority class. For example, if $\beta = 0.25$
        then each example in the training data will be deleted with probability $0.25$,
        which will result in approximately $25\%$ of the total minority class examples being deleted.

Note 2: this method is equivalent to using representation bias on the minority

'''

'''

This function performs the under-sampling bias injection

'''
def under_sample(df_minority, beta):
    X_min = df_minority.iloc[:, :].values
    cols = df_minority.columns

    # delete each example with probability beta
    for i in range(len(X_min)):
        if random.uniform(0,1) <= beta:
            X_min = np.delete(X_min, 0, axis=0)

    df_minority = pd.DataFrame(pd.DataFrame(X_min))
    df_minority.columns = cols
    return df_minority


'''

Parameters:

    beta: probability of deleting example from minority
    sens_attr: sensitive attribute
    maj_val: value of sens_attr which indicates majority class
    min_val: value of sens_attr which indicates minority class

'''
def under_sampling(df_train, beta, sens_attr,
                    maj_val, min_val):
    df_majority = df_train[df_train[sens_attr] == maj_val]
    df_minority = df_train[df_train[sens_attr] == min_val]

    df_total = df_majority
    df_undersampled = under_sample(df_minority, beta)

    # combine undersampled and original majority class to create dataset
    df_concat = pd.concat([df_total,df_undersampled])

    return df_concat.sample(frac=1) # reshuffle rows of dataframe randomly
