import pandas as pd 
import numpy as np 

from scipy import stats

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
        return df_train_full.reset_index(drop=True)
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

        return df_train.reset_index(drop=True), \
               df_val.reset_index(drop=True), \
               df_test.reset_index(drop=True), y_train, y_val, y_test

def get_numerical():
    return ['tenure', 'monthlycharges', 'totalcharges']

def get_categorical():
    cat = ['gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod'] 

    return cat

def get_p_values(df: pd.DataFrame, cat_vars: list[str], alpha:float = 0.1):
    '''
    Performs Chi-squared test on categorical features of the dataframe.
    Parameters:
    df: data frame
    cat_vars: list of strings with names of categorical columns
    alpha: float, alpha value, default 0.1 for confidence interval 99%
    Returns a data frame with p_values of all categorical variables and their significance result
    '''

    #dictionary to hold names of the column and a p_value assotiated with it
    p_v = {}
    alpha = alpha
    #for every column in category variables run a chi2 test
    for col in cat_vars:
        #create a crosstable
        observed = pd.crosstab(df[col], df.churn)
        #run a chi squared test fot categorical data
        test = stats.chi2_contingency(observed)
        p_value = test[1]
        #add the result to the dictionary
        p_v[col] = p_value
        
        #transform a dictionary to Series and then to Data Frame
        p_values = pd.Series(p_v).reset_index()
        p_values.rename(columns = {'index':'Feature', 0:'P_value'}, inplace = True)
        p_values = p_values.sort_values(by='P_value')

        #add the column that shows if the result is significant
        p_values['is_significant'] = p_values['P_value'] < alpha
    
    return p_values
