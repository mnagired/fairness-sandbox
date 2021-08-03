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
from fairlearn.metrics import *
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
# import aif360

import imblearn
from imblearn.over_sampling import *
from imblearn.under_sampling import *

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

import copy
# End: Packages for Fairness


from contextlib import contextmanager
import logging
import os

from flask import current_app, g

class Dataset:
    def __init__(self, short_name = '', path = '', cat_cols = [], num_cols = []):
        self.short_name = short_name
        self.path = path
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.df = pd.read_csv(path, sep = ';')

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

# get indices of categorical columns
def get_cat_cols(dataset):
    df = dataset.df
    res = []
    for col in dataset.cat_cols:
        res.append(df.columns.get_loc(col))
    return res

# if verbose, shows "Finished iteration: ... "
# if apply_fairness, uses fairness intervention
def tradeoff_visualization(classifier, apply_fairness = False, verbose = False):
    
    bias_amts = list(range(0,200,10))
    accuracy_on_true = []
    accuracy_on_biased = []
    accuracy_on_true_mitigated = []
    accuracy_on_biased_mitigated = []
    eod_on_true = []
    eod_on_biased = []
    dataset_size_true = np.full(shape=len(bias_amts), fill_value= X_true.shape[0]).tolist()
    dataset_size_bias = []
    table = []

    classifier_true = classifier.fit(X_true, y_true)
    y_pred_truth = classifier_true.predict(X_true)

    # Todo: check if set as global variable
    df_por = datasets['student_por'].df
    df_favored = df_por[df_por['address'] == 'U']
    df_unfavored = df_por[df_por['address'] == 'R']
    cat_cols = get_cat_cols(datasets['student_por'])

    df_undersampled = df_unfavored.sample(n=len(df_unfavored), random_state=42)

    for i in range(20):
        # under-sampling process
        if i == 0:
            df_undersampled = df_undersampled.sample(n=len(df_undersampled), random_state=42)
        else:
            df_undersampled = df_undersampled.sample(n=len(df_undersampled)-10, random_state=42)

        # combine undersampled and original favored class to create dataset
        df_concat = pd.concat([df_favored,df_undersampled])
        df_concat.shape
        df_sens = df_concat['address']

        # format data
        X_bias = df_concat.iloc[:, :-2].values
        y_bias = df_concat.iloc[:, -1].values

        # OHE
        ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), cat_cols)], remainder='passthrough')
        X_bias_true = np.array(ct.fit_transform(X_bias))
        y_bias_true = df_concat['pass']

        dataset_size_bias.append(X_bias_true.shape[0])
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

        # fairness performance (TODO)
        '''
        eod_true = equalized_odds_difference(y_true=y_bias_true, y_pred = y_pred_bias, sensitive_features=df_sens)
        eod_on_true.append(eod_true)

        eod_bias_on_true = equalized_odds_difference(y_true=y_true, y_pred = y_pred_bias_on_true, sensitive_features=sens_attrs[1])
        eod_on_biased.append(eod_bias_on_true)
        '''

        # table visualization 
        table_elem = [i*10, acc_bias, acc_bias_on_true]
        table.append(table_elem)
        
        if verbose:
            print("Finished Iteration: ", len(df_concat))

    return bias_amts, dataset_size_true, dataset_size_bias, accuracy_on_biased, accuracy_on_true, eod_on_biased, eod_on_true


def accuracy_visualizations(bias_amts, dataset_size_true, dataset_size_bias,
                            accuracy_on_biased = [], accuracy_on_true = [],
                            accuracy_on_biased_mitigated = [],
                            accuracy_on_true_mitigated = [], fairness = False):
    
    if fairness:
        plt.figure(figsize=(17,7))

        plt.subplot(1,2,1)
        plt.plot(bias_amts, accuracy_on_true_mitigated, label = 'Ground Truth')
        plt.plot(bias_amts, accuracy_on_biased_mitigated, label = 'Biased Data')
        plt.xlabel("Amount of Bias (number of minority samples removed)")
        plt.ylabel("Accuracy Score")
        plt.axhline(y=accuracy_score(y_pred_truth, y_true), color = "green", label = "Ground Truth Model Accuracy", alpha = 0.5)
        plt.title("Biased Model Accuracy")
        plt.ylim(0.92, 0.99)
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(bias_amts, dataset_size_true, label = 'Ground Truth')
        plt.plot(bias_amts, dataset_size_bias, label = 'Biased Data')
        plt.xlabel("Amount of Bias (number of minority samples removed)")
        plt.ylabel("Dataset Size")
        plt.legend()

        # plt.show()
        plt.savefig('client/public/img/figure.png')
        plt.close()
        
    else:
        plt.figure(figsize=(17,7))

        plt.subplot(1,2,1)
        plt.plot(bias_amts, accuracy_on_true, label = 'Ground Truth')
        plt.plot(bias_amts, accuracy_on_biased, label = 'Biased Data')
        plt.xlabel("Amount of Bias (number of minority samples removed)")
        plt.ylabel("Accuracy Score")
        plt.axhline(y=accuracy_score(y_pred_truth, y_true), color = "green", label = "Ground Truth Model Accuracy", alpha = 0.5)
        plt.title("Biased Model Accuracy")
        plt.ylim(0.92, 0.99)
        plt.legend()

        plt.subplot(1,2,2)
        plt.plot(bias_amts, dataset_size_true, label = 'Ground Truth')
        plt.plot(bias_amts, dataset_size_bias, label = 'Biased Data')
        plt.xlabel("Amount of Bias (number of minority samples removed)")
        plt.ylabel("Dataset Size")
        plt.legend()

        # plt.show()
        plt.savefig('client/public/img/figure.png')
        plt.close()

def fairness_visualizations(bias_amts, eod_on_true = [], eod_on_biased = [],
                           eod_on_biased_mitigated = [], eod_on_true_mitigated = [],
                           fairness = False):
    if fairness:
        plt.plot(bias_amts, eod_on_true_mitigated, label = 'Ground Truth')
        plt.plot(bias_amts, eod_on_biased_mitigated, label = 'Biased Data')
        plt.xlabel("Amount of Bias (number of minority samples removed)")
        plt.ylabel("Equalized Odds Difference")
        plt.legend()
        plt.title("Biased Model Equalized Odds Difference")
        # plt.show()
        plt.savefig('client/public/img/figure2.png')
        plt.close()
    else:
        plt.plot(bias_amts, eod_on_true, label = 'Ground Truth')
        plt.plot(bias_amts, eod_on_biased, label = 'Biased Data')
        plt.xlabel("Amount of Bias (number of minority samples removed)")
        plt.ylabel("Equalized Odds Difference")
        plt.legend()
        plt.title("Biased Model Equalized Odds Difference")
        # plt.show()
        plt.savefig('client/public/img/figure2.png')
        plt.close()

def setup():
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
    return

def test_plot(featureName):
    current_app.logger.info(f'plotting for {featureName}')
    plot_counts(datasets['student_por'].df, featureName)
    return "./img/figure.png"

def plotBefore():
    return('Ground Truth Label Distribution \n{}'.format(Counter(datasets['student_por'].df['pass'])))
    # return str(pd.value_counts(datasets['student_por'].df['pass'], sort=True))
    plot_counts_data(datasets['student_por'].df['pass'])
    # current_app.logger.info(datasets['student_por'].df['pass'].count())
    current_app.logger.info('hello')
    return "./img/figure.png"

def plot_counts_data(data):
    count = pd.value_counts(data, sort = True)
    fig = count.plot(kind = 'bar', rot = 0)
    fig.figure.savefig('client/public/img/figure.png')
    plt.close()

def plotCounts(featureName):
    if featureName in curr_df.columns:
        temp = curr_df[featureName].value_counts(normalize=True)
        # fig.figure.savefig('client/public/img/figure.png')
        current_app.logger.info(temp)
        return temp.to_json()
    else:
        current_app.logger.info("Error! Please enter a valid feature.")

# Old. Using matplotlib
def plot_counts(df, attr):
    if attr in df.columns:
        fig = df[attr].value_counts(normalize=True).plot.barh()
        fig.figure.savefig('client/public/img/figure.png')
        current_app.logger.info("Plot success!")
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

def injectBias():
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

def trainModel():
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

def fairnessIntervention():
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

def fairnessTradeoff():
    classifier = DecisionTreeClassifier(min_samples_leaf = 10, max_depth = 4)
    bias_amts, dataset_size_true, dataset_size_bias, accuracy_on_biased, accuracy_on_true, eod_on_biased, eod_on_true = tradeoff_visualization(classifier, False, False)
    accuracy_visualizations(bias_amts, dataset_size_true, dataset_size_bias, accuracy_on_biased, accuracy_on_true, eod_on_biased, eod_on_true, False)
    # fairness_visualizations(bias_amts, eod_on_true, eod_on_biased, False)
    return "./img/figure.png"

def fairnessTradeoff2():
    classifier = DecisionTreeClassifier(min_samples_leaf = 10, max_depth = 4)
    bias_amts, dataset_size_true, dataset_size_bias, accuracy_on_biased, accuracy_on_true, eod_on_biased, eod_on_true = tradeoff_visualization(classifier, True, False)
    accuracy_visualizations(bias_amts, dataset_size_true, dataset_size_bias, accuracy_on_biased, accuracy_on_true, eod_on_biased, eod_on_true, True)
    # fairness_visualizations(bias_amts, eod_on_true, eod_on_biased, False)
    return "./img/figure.png"