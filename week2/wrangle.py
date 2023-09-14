import pandas as pd 
import numpy as np

def rename_columns(df):
    # columns to lower case
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    string_columns = list(df.dtypes[df.dtypes == 'object'].index)

    for col in string_columns:
        df[col] = df[col].str.lower().str.replace(' ', '_')
    
    return df

def split_data(df):
    '''
    splits dataframe into train, validate, test
    where train is 60%, validate - 20%,test - 20% 
    '''
    # make the split reproducible
    np.random.seed(2912)

    n = len(df)

    # 20% for validate, 20% for test and 60% for train
    n_val = int(0.2 * n)
    n_test = int(0.2 * n)
    n_train = n - (n_val + n_test)

    # shuffle values to make the split random
    idx = np.arange(n)
    np.random.shuffle(idx)

    # rearrange the values inside the dataframe
    df_shuffled = df.iloc[idx]

    # split with slicing the data
    df_train = df_shuffled.iloc[:n_train].copy().reset_index(drop=True)
    df_val = df_shuffled.iloc[n_train:n_train+n_val].copy().reset_index(drop=True)
    df_test = df_shuffled.iloc[n_train+n_val:].copy().reset_index(drop=True)

    return df_train, df_val, df_test
    