'''

this file contains 6 different biases

'''

import random
import pandas
import numpy as np

'''

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
