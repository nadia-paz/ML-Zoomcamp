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
    Returns dataframes: df_train, df_val, df_test
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

def get_target_vars(df_train, df_val, df_test):
    ''' 
    Separates the target variable from train, validate and test data sets.
    Delets the column with target variable from all 3 data sets.
    Returns numpy arrays: y_train, y_val, y_tes
    '''
    # separate the target variable and make its logarithmic transformation
    y_train = np.log1p(df_train.msrp.values)
    y_val = np.log1p(df_val.msrp.values)
    y_test = np.log1p(df_test.msrp.values)

    # drop the target variable from the data sets
    del df_train['msrp']
    del df_val['msrp']
    del df_test['msrp']

    return y_train, y_val, y_test

def get_target_var(df: pd.DataFrame, target: str):
    ''' 
    Parameters:
        df -> dataframe
        target -> string, name of the column with the target variable
    Separates the target variable from a data set.
    Delets a column with a target variable from it.
    Returns: 
        numpy array
    '''
    # separate the target variable and make its logarithmic transformation
    y = np.log1p(df[target].values)


    # drop the target variable from the data sets
    del df[target]

    return y
        