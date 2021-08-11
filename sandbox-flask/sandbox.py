# Start: Packages for Fairness
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_curve, auc
from collections import Counter

import fairlearn
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import *
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from fairlearn.preprocessing import CorrelationRemover

# import aif360

import imblearn
from imblearn.over_sampling import *
from imblearn.under_sampling import *

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

import copy
import json
# End: Packages for Fairness

# For synthetic data
from synthetic_data import get_synthetic_data

# For bias injection
from biases import *
from biases import representation

from contextlib import contextmanager
import logging
import os

from flask import current_app, g

class Dataset:
    def __init__(self, short_name = '', path = '', cat_cols = [], num_cols = [],
                 sens_attr = '', has_sens_attr = True,
                 sep = '', synthetic = False):
        self.short_name = short_name
        self.path = path
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.has_sens_attr = has_sens_attr
        if has_sens_attr:
            self.sens_attr = sens_attr
        if not synthetic:
            self.df = pd.read_csv(path, sep = sep)

# each dataset is a dictionary where keys = short name, values = Dataset object

# Defining global variable
datasets = dict()
X_true = None
y_true = None
X_bias_true = None
y_bias_true = None
y_pred_truth = None
y_pred_mitigated_true = None
y_pred_mitigated_bias = None
y_pred_mitigated_bias_on_true = None
classifier_true = None
classifier_bias = None
df_sens = None # Consider not using global variable
y_pred_mitigated_true = None
y_pred_mitigated_bias = None
y_pred_mitigated_bias_on_true = None

curr_df = None
X_bias = None
y_bias = None
df_train = None
df_test = None
y_pred_bias = None
y_pred_bias_on_true = None
sens_feat_true = None
sens_feat_bias = None
biases = None
df_minority = None
df_bias = None
currentBias = None
# X_train = None # potentially not needed
# y_train = None # potentially not needed


##### Helper function #####

def add_bias(bias_func, short_name):
    global biases
    biases[short_name] = bias_func

def add_dataset(dataset):
    global datasets # referencing the global variable datasets
    if not isinstance(dataset, Dataset):
        current_app.logger.info("Error! Please enter a valid Dataset object")
    else:
        if dataset.short_name not in datasets.keys():
            datasets[dataset.short_name] = dataset

def threshold(df, g_1=0.3, g_2=0.3, g_3=0.4, threshold=11):
    """
    Added "pass/fail" to make problem binary classification
    """
    assert g_1 + g_2 + g_3 == 1, "The sum of the percentages should be 1"
    assert 0 < threshold < 20, "Threshold needs to be between 0 and 20"
    df['pass'] = df.apply(lambda row: 1
                                 if g_1*row['G1'] + g_2*row['G2'] + g_3*row['G3'] >= threshold
                                 else 0, axis=1)

# OHE categorical features (prompt for user's choice here?)

# get indices of categorical columns
def get_cat_cols(dataset):
    df = dataset.df
    res = []
    for col in dataset.cat_cols:
        res.append(df.columns.get_loc(col))
    return res


'''
    train_ratio: is the proportion of data examples in the training set
        (1-train_ratio is proportion in unbiased testing set)
'''
def train_test_split(df, train_ratio = 0.5):
    
    df_train = df.loc[range(0,int(len(df)*train_ratio)), :]
    df_test = df.loc[range(int(len(df)*train_ratio)+1, len(df)), :]
    
    return df_train, df_test

'''

This function separates the minority and majority classes

Parameters:
    
    sens_attr: sensitive attribute
    maj_val: value of sens_attr which indicates majority class
    min_val: value of sens_attr which indicates minority class

'''
def get_maj_min(df, sens_attr, maj_val, min_val):
    assert sens_attr in list(df.columns), "Sensitive attribute must be a column in the dataframe!"
    df_majority = df[df[sens_attr] == maj_val]
    df_minority = df[df[sens_attr] == min_val]
    
    return df_majority, df_minority

# if verbose, shows "Finished iteration: ... "
# if apply_fairness, uses fairness intervention
def tradeoff_visualization(bias_amts, classifier, X_true, y_true, 
                           df_train, sensitive_feature = "cat",
                           is_synthetic = False,
                           apply_fairness = False, verbose = False):
    
    accuracy_on_true = []
    accuracy_on_biased = []
    accuracy_on_true_mitigated = []
    accuracy_on_biased_mitigated = []
    
    count = 0

    for bias in bias_amts:
        
        df_train_copy = df_train.copy()
        
        if currentBias == 'omitted_variable':
            df_bias = biases['omitted_variable'](datasets, df_train, 'synthetic', 'num1', is_sens_attr=False)
        elif currentBias == 'random_over_sampling':
            df_bias = biases['random_over_sampling'](df_train, 'sens_feat', 1, 0, 2)
        elif currentBias == 'over_sampling':
            df_bias = biases['over_sampling'](df_train, df_minority, 'sens_feat', 1, 0, 2, type=2)
        elif currentBias == 'label_noise':
            df_bias = biases['label_noise'](df_train, 'sens_feat', 'categorical', 1, 0.2)
        elif currentBias == 'measurement':
            df_bias = biases['measurement'](df_train, 'cat2', 'categorical', noise_prob=1, noise_type=1, subgroups=[2])
        else:
            df_bias = biases['representation'](df_train, (df_train['num1'] > 0) & (df_train['cat1'] == 0), 0.5)
        df_sens = df_bias[sensitive_feature]

        # format data
        X_bias = df_bias.iloc[:, :-1].values
        y_bias = df_bias.iloc[:, -1].values
        
        if not is_synthetic:
            # OHE
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), cat_cols)], remainder='passthrough')
            X_bias_true = np.array(ct.fit_transform(X_bias))
        else:
            X_bias_true = X_bias
        
        y_bias_true = df_bias.iloc[:, -1].values
        
        classifier_bias = classifier.fit(X_bias_true, y_bias_true)
        
        if apply_fairness:
            constraint = DemographicParity()
            classifier_mitigated_bias = ExponentiatedGradient(classifier_bias, constraint)
            classifier_mitigated_bias.fit(X_bias_true, y_bias_true, sensitive_features = df_sens)
            
            # testing on biased data WITH fairness intervention
            y_pred_mitigated_bias = classifier_mitigated_bias.predict(X_bias_true)
            
            # testing on GT data WITH fairness intervention
            y_pred_mitigated_bias_on_true = classifier_mitigated_bias.predict(X_true)
        
        # testing on biased data withOUT fairness intervention
        y_pred_bias = classifier_bias.predict(X_bias_true)
        
        # testing on GT data withOUT fairness intervention
        y_pred_bias_on_true = classifier_bias.predict(X_true)

        # model performance
        
        if apply_fairness:
            # on biased data
            acc_bias_mitigated = accuracy_score(y_pred=y_pred_mitigated_bias, y_true=y_bias_true)
            accuracy_on_biased_mitigated.append(acc_bias_mitigated)
            # on GT data
            acc_bias_mitigated_on_true = accuracy_score(y_pred=y_pred_mitigated_bias_on_true, y_true=y_true)
            accuracy_on_true_mitigated.append(acc_bias_mitigated_on_true)
        
        # on biased data
        acc_bias = accuracy_score(y_pred=y_pred_bias, y_true=y_bias_true)
        accuracy_on_biased.append(acc_bias)
        # on GT data
        acc_bias_on_true = accuracy_score(y_pred=y_pred_bias_on_true, y_true=y_true)
        accuracy_on_true.append(acc_bias_on_true)
        
        if verbose:
            print("Finished Iteration: ", count)
            count +=1

    return bias_amts, accuracy_on_biased, accuracy_on_true, \
           accuracy_on_biased_mitigated, accuracy_on_true_mitigated


##### End Helper Function #####

def setup_old():
    global datasets, X_true, y_true, curr_df
    current_app.logger.info('Adding dataset')
    # example - adding a dataset
    path_adult_income = 'datasets/adult.csv'
    cat_cols = ['workclass', 'education','marital-status', 'occupation', 'relationship', 'race',
                'gender', 'native-country','income']
    num_cols = ['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    adult_income = Dataset('adult_income', path_adult_income, cat_cols, num_cols)

    add_dataset(adult_income)

    # TODO - add more datasets

    cat = ['school', 'sex', 'address','famsize','Pstatus','Mjob','Fjob','reason',
       'guardian','schoolsup','famsup','paid', 'activities','nursery','higher', 'internet','romantic']
    num = ['age', 'Medu', 'Fedu','traveltime','studytime','failures', 'famrel',
        'freetime','goout','Dalc','Walc','health','absences','G1', 'G2', 'G3']

    add_dataset(Dataset("student_mat", path='datasets/student-mat.csv', cat_cols=cat, num_cols=num))
    add_dataset(Dataset("student_por", path='datasets/student-por.csv', cat_cols=cat, num_cols=num))

    current_app.logger.info('dataset added')

    # take a peek at the first few data points
    df_por = datasets['student_por'].df
    df_por.head()
    current_app.logger.info(df_por.head(1))

    #generate_pnp(datasetMath)
    threshold(df_por, threshold=14)
    # df_por['pass'].value_counts()

    sens_attrs = [df_por['sex'], df_por['address']]

    # for reference
    # current_app.logger.info(datasets['student_por'].cat_cols)
    # current_app.logger.info(datasets['student_por'].num_cols)

    # pre-process data
    X_true, y_true = dataPreprocess()
    curr_df = df_por

    df_synthetic = get_synthetic_data(n=1000, r = 0.25, num_numerical_feats=3, num_cat_feats=2,
                                  ranges = [(1, 100), (0, 10), (-10, 25)],
                                  num_types=(0, 1, 1),
                                  cat_levels=[2,3], diff_dist=True, label_noise = 0.1)

    # add to dictionary of datasets
    path_synthetic = 'datasets/synthetic_data.csv'
    df_synthetic.to_csv(path_synthetic)
    add_dataset(Dataset(short_name="synthetic", path=path_synthetic, cat_cols=[], num_cols=[], synthetic=True, sens_attr = "sens_feat"))
    current_app.logger.info(datasets['synthetic'].head())
    
    return

def setup():
    global datasets, X_true, y_true, df_train, df_test, curr_df, df_minority
    df_synthetic = get_synthetic_data(n=1000, r = 0.25, num_numerical_feats=3, num_cat_feats=2,
                                  ranges = [(1, 100), (0, 10), (-10, 25)],
                                  num_types=(0, 1, 1),
                                  cat_levels=[2,3], diff_dist=True, label_noise = 0.1)

    # add to dictionary of datasets
    path_synthetic = 'datasets/synthetic_data.csv'
    df_synthetic.to_csv(path_synthetic)
    add_dataset(Dataset(short_name="synthetic", path=path_synthetic, cat_cols=[], num_cols=[], synthetic=True, sens_attr = "sens_feat"))
    curr_df = df_synthetic
    current_app.logger.info(df_synthetic.head())

    # Potentially splitting these
    df_train, df_test = train_test_split(curr_df)
    df_majority, df_minority = get_maj_min(df_train, 'sens_feat', 1, 0)

    X_train = df_train.iloc[:, :-1].values
    y_train = df_train.iloc[:, -1].values
    X_true = df_test.iloc[:, :-1].values
    y_true = df_test.iloc[:, -1].values

    sens_attrs_true = [df_test[datasets['synthetic'].sens_attr]]

    return

def plotCounts(featureName):
    if featureName in curr_df.columns:
        results = curr_df[featureName].value_counts(normalize=True)
        current_app.logger.info(results)
        return results.to_json()
    else:
        current_app.logger.info("Error! Please enter a valid feature.")

def dataPreprocess():
    # To-do: pass in dataset instead
    df_por = datasets['student_por'].df

    # format data
    X = df_por.iloc[:, :-2].values
    y = df_por.iloc[:, -1].values

    cat_cols = get_cat_cols(datasets['student_por'])
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), cat_cols)], remainder='passthrough')
    X = np.array(ct.fit_transform(X))

    X_test = df_por.iloc[:, :-2]
    X_dum = pd.get_dummies(X_test)

    X_true = X
    y_true = df_por['pass']
    return (X_true, y_true)

def injectBias_old():
    global datasets, X_bias_true, y_bias_true, df_sens
    df_por = datasets['student_por'].df
    
    # separate based on protected attribute
    sens_attrs = [df_por['sex'], df_por['address']]
    sens_values = sens_attrs[1].unique()
    
    df_favored = df_por[df_por['address'] == 'U']
    df_unfavored = df_por[df_por['address'] == 'R']
    X_true, y_true = dataPreprocess()

    # under-sampling process
    df_undersampled = df_unfavored.sample(n=190, random_state=42)

    #print(df_favored.shape, df_unfavored.shape, df_undersampled.shape)

    # combine undersampled and original favored class to create dataset
    df_concat = pd.concat([df_favored,df_undersampled])
    df_concat.shape

    # for fairness measures later
    df_sens = df_concat['address']

    # format data
    X_bias = df_concat.iloc[:, :-2].values
    #print(X_undersampled.shape)
    y_bias = df_concat.iloc[:, -1].values

    # OHE
    cat_cols = get_cat_cols(datasets['student_por'])
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), cat_cols)], remainder='passthrough')
    X_bias_true = np.array(ct.fit_transform(X_bias))
    y_bias_true = df_concat['pass']


    # Post-injection visualization
    favored = len(df_favored)
    true_unfavored = len(df_por[df_por['address'] == 'R'])
    bias_unfavored = len(df_undersampled)

    x_vals = ['Favored', "Unfavored"]
    y_vals_true = [favored, true_unfavored]
    y_vals_bias = [favored, bias_unfavored]

    plt.subplot(1,2,1)
    plt.bar(x_vals, y_vals_true)
    plt.title("Ground Truth")
    plt.ylabel("Count")

    plt.subplot(1,2,2)
    plt.bar(x_vals, y_vals_bias)
    plt.title("Under-Sampling")
    plt.ylim([0,500])

    # plt.show()
    plt.savefig('client/public/img/figure.png')
    plt.close()
    return "./img/figure.png"

def injectBias(selectedBias):
    global datasets, X_bias, y_bias, df_sens, biases, df_bias, currentBias

    biases = dict()

    add_bias(under_sampling, 'under_sampling')
    add_bias(omitted_variable, 'omitted_variable')
    add_bias(random_over_sampling, 'random_over_sampling')
    add_bias(over_sampling, 'over_sampling')
    add_bias(label_noise, 'label_noise')
    add_bias(measurement, 'measurement')
    add_bias(representation, 'representation')

    if selectedBias == 'omitted_variable':
        df_bias = biases['omitted_variable'](datasets, df_train, 'synthetic', 'num1', is_sens_attr=False)
    elif selectedBias == 'random_over_sampling':
        df_bias = biases['random_over_sampling'](df_train, 'sens_feat', 1, 0, 2)
    elif selectedBias == 'over_sampling':
        df_bias = biases['over_sampling'](df_train, df_minority, 'sens_feat', 1, 0, 2, type=2)
    elif selectedBias == 'label_noise':
        df_bias = biases['label_noise'](df_train, 'sens_feat', 'categorical', 1, 0.2)
    elif selectedBias == 'measurement':
        df_bias = biases['measurement'](df_train, 'cat2', 'categorical', noise_prob=1, noise_type=1, subgroups=[2])
    else:
        df_bias = biases['representation'](df_train, (df_train['num1'] > 0) & (df_train['cat1'] == 0), 0.5)
    currentBias = selectedBias

    # for fairness measures later
    if datasets['synthetic'].has_sens_attr:
        df_sens = df_bias[datasets['synthetic'].sens_attr]

    # format data
    X_bias = df_bias.iloc[:, :-1].values
    y_bias = df_bias.iloc[:, -1].values

    return "success" #temp


def trainModel_old():
    global datasets, X_bias_true, y_bias_true, y_pred_truth, classifier_true, classifier_bias

    classifier = DecisionTreeClassifier(min_samples_leaf = 10, max_depth = 4)

    classifier_true = classifier.fit(X_true, y_true)
    y_pred_truth = classifier_true.predict(X_true)

    classifier_bias = classifier.fit(X_bias_true, y_bias_true)
    y_pred_bias = classifier_bias.predict(X_bias_true)
    y_pred_bias_on_true = classifier_bias.predict(X_true)
    current_app.logger.info("Training finished.")

    # To-do: update return method
    result_text = f"Accuracy of Ground Truth Model on Ground Truth Data: {accuracy_score(y_pred_truth, y_true)} | " \
        f"Accuracy of Biased Model on Biased Data: {accuracy_score(y_pred_bias, y_bias_true)} | " \
        f"Accuracy of Biased Model on Ground Truth Data: {accuracy_score(y_pred_bias_on_true, y_true)}"
    return result_text

def trainModel():
    global datasets, y_pred_truth, classifier_true, classifier_bias, y_pred_bias, y_pred_bias_on_true, sens_feat_true, sens_feat_bias

    classifier = LogisticRegression(random_state = 42)

    classifier_true = classifier.fit(X_true, y_true)
    y_pred_truth = classifier_true.predict(X_true)

    classifier_bias = classifier.fit(X_bias, y_bias)
    y_pred_bias = classifier_bias.predict(X_bias)
    y_pred_bias_on_true = classifier_bias.predict(X_true)

    sens_feat_true = df_test['sens_feat']
    sens_feat_bias = df_sens

    results = {
        "Accuracy of Ground Truth Model on Ground Truth Data": accuracy_score(y_pred_truth, y_true),
        "Accuracy of Biased Model on Biased Data": accuracy_score(y_pred_bias, y_bias),
        "Accuracy of Biased Model on Ground Truth Data": accuracy_score(y_pred_bias_on_true, y_true)
    }
    current_app.logger.info(results)
    # gm_true = MetricFrame(metrics=accuracy_score,y_true=y_true, y_pred=y_pred_truth, sensitive_features = sens_feat_true)
    # current_app.logger.info("Overall Accuracy: ", gm_true.overall)
    # current_app.logger.info("Group Accuracy : ", gm_true.by_group[0])
    # current_app.logger.info("Group Accuracy : ", gm_true.by_group[1])

    return json.dumps(results)

def fairnessIntervention_old():
    global y_pred_mitigated_true, y_pred_mitigated_bias, y_pred_mitigated_bias_on_true
    df_por = datasets['student_por'].df
    sens_attrs = [df_por['sex'], df_por['address']]

    np.random.seed(0)
    constraint = DemographicParity()
    mitigator_true = ExponentiatedGradient(classifier_true, constraint)
    mitigator_true.fit(X_true, y_true, sensitive_features = sens_attrs[1])
    y_pred_mitigated_true = mitigator_true.predict(X_true)
    constraint = DemographicParity()
    mitigator_bias = ExponentiatedGradient(classifier_bias, constraint)
    mitigator_bias.fit(X_bias_true, y_bias_true, sensitive_features = df_sens)
    y_pred_mitigated_bias = mitigator_bias.predict(X_bias_true)
    y_pred_mitigated_bias_on_true = mitigator_bias.predict(X_true)
    
    result_text = f"Accuracy of Ground Truth Model + Fairness Intervention on Ground Truth Data: {accuracy_score(y_pred_mitigated_true, y_true)} | " \
        f"Accuracy of Biased Model + Fairness Intervention on Ground Truth Data: {accuracy_score(y_pred_mitigated_bias_on_true, y_true)} "
    # result_text = "success"
    return result_text

def fairnessIntervention():
    global y_pred_mitigated_true, y_pred_mitigated_bias, y_pred_mitigated_bias_on_true

    # remover = CorrelationRemover(sensitive_feature_ids=[5])
    remover_df = CorrelationRemover(sensitive_feature_ids=['sens_feat'])

    # X_train_corr = remover.fit_transform(X_train)
    df_corr = pd.DataFrame(remover_df.fit_transform(df_train))

    df_temp = df_train.drop('sens_feat', axis=1)
    df_corr.columns = df_temp.columns

    # Exponentiated Gradient
    constraint = DemographicParity()
    mitigator_bias = ExponentiatedGradient(classifier_bias, constraint)
    mitigator_bias.fit(X_bias, y_bias, sensitive_features = sens_feat_bias)
    y_pred_mitigated_bias_on_true = mitigator_bias.predict(X_true)

    current_app.logger.info("Accuracy of Biased Model + Fairness Intervention on Ground Truth Data: ",
        accuracy_score(y_pred_mitigated_bias_on_true, y_true))

    result_text = "success"
    return result_text

def fairnessTradeoff():
    classifier = LogisticRegression()

    bias_amts = np.divide(list(range(10,-1,-1)),10)

    bias_amts, accuracy_on_biased, accuracy_on_true, \
            accuracy_on_biased_mitigated, accuracy_on_true_mitigated = \
    tradeoff_visualization(bias_amts, classifier, X_true, y_true,
                        df_train, "sens_feat", is_synthetic=True,
                        apply_fairness=True, verbose=True)

    current_app.logger.info(accuracy_on_biased)

    results = {
        "Bias Amounts": bias_amts.tolist(),
        "Tested On Biased Data + No Fairness Intervention": accuracy_on_biased,
        "Tested On Biased Data + Fairness Intervention": accuracy_on_biased_mitigated,
        "Tested On Ground Truth + No Fairness Intervention": accuracy_on_true,
        "Tested On Ground Truth + Fairness Intervention": accuracy_on_true_mitigated
    }

    current_app.logger.info(results)

    result_text = "success"
    return json.dumps(results)

# def fairnessTradeoff():
#     classifier = DecisionTreeClassifier(min_samples_leaf = 10, max_depth = 4)
#     bias_amts, dataset_size_true, dataset_size_bias, accuracy_on_biased, accuracy_on_true, eod_on_biased, eod_on_true = tradeoff_visualization(classifier, False, False)
#     accuracy_visualizations(bias_amts, dataset_size_true, dataset_size_bias, accuracy_on_biased, accuracy_on_true, eod_on_biased, eod_on_true, False)
#     # fairness_visualizations(bias_amts, eod_on_true, eod_on_biased, False)
#     return "./img/figure.png"

# def fairnessTradeoff2():
#     classifier = DecisionTreeClassifier(min_samples_leaf = 10, max_depth = 4)
#     bias_amts, dataset_size_true, dataset_size_bias, accuracy_on_biased, accuracy_on_true, eod_on_biased, eod_on_true = tradeoff_visualization(classifier, True, False)
#     accuracy_visualizations(bias_amts, dataset_size_true, dataset_size_bias, accuracy_on_biased, accuracy_on_true, eod_on_biased, eod_on_true, True)
#     # fairness_visualizations(bias_amts, eod_on_true, eod_on_biased, False)
#     return "./img/figure.png"