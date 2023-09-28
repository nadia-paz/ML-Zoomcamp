import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

def get_telco_data():
    data = '../data/telco.csv'
    df = pd.read_csv(data)
    # 'coerce' fills errors with Null values
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    # fill nulls with zeros
    df.TotalCharges = df.TotalCharges.fillna(0)
    # rename columns
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    # drop customer id
    del df['customerid']
    # get names of string columns
    string_cols = df.dtypes[df.dtypes == 'object'].index
    # bring all values in string columns to lower case
    for col in string_cols:
        df[col] = df[col].str.lower().str.replace(' ', '_')
    # from yes/no to 1/0
    df.churn = (df.churn == 'yes').astype('uint8')

    return df

def split_telco_data(df=get_telco_data(), seed=42, explore=True):
    df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=seed)
    if explore:
        return df_train_full
    else:
        df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=seed)
        # get y arrays
        y_train = df_train.churn.values
        y_val = df_val.churn.values
        y_test = df_test.churn.values
        # delete target var from data sets
        del df_train['churn']
        del df_val['churn']
        del df_test['churn']

        return df_train, df_val, df_test, y_train, y_val, y_test
