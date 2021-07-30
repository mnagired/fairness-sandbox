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
# import aif360

import imblearn
from imblearn.over_sampling import *
from imblearn.under_sampling import *

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

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

datasets = dict()

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

def setup():
    global datasets
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
    current_app.logger.info("hello")
    current_app.logger.info(df_por.head(1))

    #generate_pnp(datasetMath)
    threshold(df_por, threshold=14)
    # df_por['pass'].value_counts()

    sens_attrs = [df_por['sex'], df_por['address']]

    # for reference
    # current_app.logger.info(datasets['student_por'].cat_cols)
    # current_app.logger.info(datasets['student_por'].num_cols)


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

def test_plot(featureName):
    current_app.logger.info(f'plotting for {featureName}')
    plot_counts(datasets['student_por'].df, featureName)
    return "./img/figure.png"

def plotCounts(featureName):
    if attr in df.columns:
        temp = df[attr].value_counts(normalize=True)
        # fig.figure.savefig('client/public/img/figure.png')
        current_app.logger.info(temp)
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
    # df_por = datasets['student_por'].df
    X_true, y_true = dataPreprocess()

    # Over-Sampling with SMOTE
    smote = SMOTE()
    X_smote, y_smote = smote.fit_resample(X_true, y_true)

    # Under-Sampling with Repeated Edited Nearest Neighbours
    # renn = RepeatedEditedNearestNeighbours(sampling_strategy="majority", max_iter=3, n_neighbors=3)
    # X_renn, y_renn = renn.fit_resample(X_true, y_true)

    # # print("Over-Sampling with SMOTE\n")
    # print('Ground Truth Label Distribution \n{}'.format(Counter(y_true)))
    text = "Over-Sampling with SMOTE\n"
    text += 'Biased Data Label Distribution \n{}'.format(Counter(y_smote))
    return text

def injectBiasUnder():
    df_por = datasets['student_por'].df
    X_true, y_true = dataPreprocess()

    # Under-Sampling with Repeated Edited Nearest Neighbours
    renn = RepeatedEditedNearestNeighbours(sampling_strategy="majority", max_iter=3, n_neighbors=3)
    X_renn, y_renn = renn.fit_resample(X_true, y_true)

    # # print("Over-Sampling with SMOTE\n")
    # print('Ground Truth Label Distribution \n{}'.format(Counter(y_true)))
    text = "Under-Sampling with Repeated Edited Nearest Neighbors\n"
    text += 'Biased Data Label Distribution \n{}'.format(Counter(y_renn))
    return text